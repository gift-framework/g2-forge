"""
Trainer: Main training loop for G₂ metric construction

Handles:
- Curriculum learning with phase-based scheduling
- Optimization with learning rate scheduling
- Checkpointing and model saving
- Metrics tracking and logging

Universal - works for any G₂ manifold configuration!
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm.auto import tqdm
import json
import time

from ..utils.config import G2ForgeConfig
from ..core.operators import (
    build_levi_civita_sparse_7d,
    compute_exterior_derivative,
    hodge_star_3,
    compute_coclosure,
    reconstruct_metric_from_phi,
)
from ..core.losses import CompositeLoss
from ..networks import (
    create_phi_network_from_config,
    create_harmonic_networks_from_config,
)
from ..manifolds import create_manifold


class Trainer:
    """
    Main trainer for G₂ metric construction.

    Implements curriculum learning, optimization, and checkpointing.
    Auto-adapts to any topology from config.
    """

    def __init__(
        self,
        config: G2ForgeConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        verbose: bool = True
    ):
        """
        Initialize trainer.

        Args:
            config: G2ForgeConfig with complete setup
            device: Device to use ('cuda' or 'cpu')
            verbose: Print progress
        """
        self.config = config
        self.device = torch.device(device)
        self.verbose = verbose

        if self.verbose:
            print(f"Initializing Trainer on {self.device}")
            print(f"Manifold: {config.manifold.type}")
            print(f"Topology: b₂={config.manifold.topology.b2}, b₃={config.manifold.topology.b3}")

        # Create manifold
        self.manifold = create_manifold(config.manifold)

        # Create networks
        self.phi_network = create_phi_network_from_config(config).to(self.device)
        self.h2_network, self.h3_network = create_harmonic_networks_from_config(config)
        self.h2_network = self.h2_network.to(self.device)
        self.h3_network = self.h3_network.to(self.device)

        if self.verbose:
            print(f"Networks created:")
            print(f"  Phi: {self.phi_network.count_parameters():,} params")
            print(f"  H²: {self.h2_network.count_parameters():,} params (n_forms={self.h2_network.n_forms})")
            print(f"  H³: {self.h3_network.count_parameters():,} params (n_forms={self.h3_network.n_forms})")

        # Create optimizer
        all_params = (
            list(self.phi_network.parameters()) +
            list(self.h2_network.parameters()) +
            list(self.h3_network.parameters())
        )
        self.optimizer = AdamW(
            all_params,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay
        )

        # Create scheduler
        self.scheduler = self._create_scheduler()

        # Create loss function
        self.loss_fn = CompositeLoss(
            topology=config.manifold.topology,
            manifold=self.manifold,
            assoc_cycles=self.manifold.get_associative_cycles(),
            coassoc_cycles=self.manifold.get_coassociative_cycles()
        ).to(self.device)

        # Build Levi-Civita (once, reuse)
        self.eps_indices, self.eps_signs = build_levi_civita_sparse_7d()
        self.eps_indices = self.eps_indices.to(self.device)
        self.eps_signs = self.eps_signs.to(self.device)

        # Metrics tracking
        self.metrics_history = []
        self.start_epoch = 0

        # Checkpointing
        self.checkpoint_dir = Path(config.checkpointing.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision (AMP) setup
        self.use_amp = config.training.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.verbose and self.use_amp:
            print(f"Mixed-precision (AMP) enabled: ~2x speedup expected")

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.training.warmup_epochs
        )

        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.total_epochs - self.config.training.warmup_epochs,
            eta_min=self.config.training.lr_min
        )

        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.config.training.warmup_epochs]
        )

        return scheduler

    def _get_current_phase(self, epoch: int):
        """Get current curriculum phase."""
        for phase_name, phase_config in self.config.training.curriculum.items():
            if phase_config.epoch_range[0] <= epoch < phase_config.epoch_range[1]:
                return phase_name, phase_config
        # Return last phase if beyond all ranges
        last_phase = list(self.config.training.curriculum.items())[-1]
        return last_phase

    def train_step(self, epoch: int) -> Dict[str, Any]:
        """
        Single training step with optional mixed-precision (AMP).

        Args:
            epoch: Current epoch

        Returns:
            metrics: Dict of metrics for this step
        """
        # Get curriculum phase
        phase_name, phase_config = self._get_current_phase(epoch)
        grid_n = phase_config.grid_n
        loss_weights = phase_config.loss_weights

        # Sample coordinates
        coords = self.manifold.sample_coordinates(
            n_samples=self.config.training.batch_size,
            grid_n=grid_n,
            device=str(self.device)
        )
        coords.requires_grad_(True)

        # Forward pass with autocast for mixed precision
        with autocast(enabled=self.use_amp):
            # Forward pass
            phi_tensor = self.phi_network.get_phi_tensor(coords)
            h2_forms = self.h2_network(coords)
            h3_forms = self.h3_network(coords)

            # Compute geometric quantities
            dphi = compute_exterior_derivative(
                phi_tensor,
                coords,
                subsample_factor=1  # Can subsample for speed
            )

            metric = reconstruct_metric_from_phi(phi_tensor)

            star_phi = hodge_star_3(
                phi_tensor,
                metric,
                self.eps_indices,
                self.eps_signs
            )

            dstar_phi = compute_coclosure(
                star_phi,
                coords,
                subsample_factor=self.config.training.subsample_coclosure
            )

            # Region weights (for TCS)
            region_weights = self.manifold.get_region_weights(coords)

            # Compute loss
            total_loss, components = self.loss_fn(
                phi=phi_tensor,
                dphi=dphi,
                dstar_phi=dstar_phi,
                star_phi=star_phi,
                metric=metric,
                harmonic_h2=h2_forms,
                harmonic_h3=h3_forms,
                region_weights=region_weights,
                loss_weights=loss_weights,
                epoch=epoch
            )

        # Backward pass with gradient scaling for AMP
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()

        # Unscale before clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.phi_network.parameters()) +
            list(self.h2_network.parameters()) +
            list(self.h3_network.parameters()),
            self.config.training.grad_clip
        )

        # Optimizer step with scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # Metrics
        metrics = {
            'epoch': epoch,
            'phase': phase_name,
            'loss': total_loss.item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            **components
        }

        return metrics

    def train(
        self,
        num_epochs: Optional[int] = None,
        resume: bool = False
    ) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs (default: from config)
            resume: Resume from checkpoint

        Returns:
            results: Training results and final metrics
        """
        if num_epochs is None:
            num_epochs = self.config.training.total_epochs

        if resume:
            self.load_checkpoint()

        if self.verbose:
            print(f"\nStarting training for {num_epochs} epochs")
            print(f"From epoch {self.start_epoch} to {num_epochs}")
            print("="*60)

        start_time = time.time()

        # Training loop
        pbar = tqdm(
            range(self.start_epoch, num_epochs),
            desc="Training",
            disable=not self.verbose
        )

        for epoch in pbar:
            metrics = self.train_step(epoch)
            self.metrics_history.append(metrics)

            # Update progress bar
            if self.verbose:
                pbar.set_postfix({
                    'phase': metrics['phase'][:10],
                    'loss': f"{metrics['loss']:.2e}",
                    'tors_cl': f"{metrics['torsion_closure']:.2e}",
                    'rank_h2': f"{metrics['rank_h2']}/{self.h2_network.n_forms}",
                    'rank_h3': f"{metrics['rank_h3']}/{self.h3_network.n_forms}"
                })

            # Logging
            if epoch % 100 == 0 and self.verbose:
                print(f"\nEpoch {epoch}/{num_epochs} [{metrics['phase']}]")
                print(f"  Loss: {metrics['loss']:.6f}")
                print(f"  Torsion closure: {metrics['torsion_closure']:.2e}")
                print(f"  Torsion coclosure: {metrics['torsion_coclosure']:.2e}")
                print(f"  Rank H²: {metrics['rank_h2']}/{self.h2_network.n_forms}")
                print(f"  Rank H³: {metrics['rank_h3']}/{self.h3_network.n_forms}")
                print(f"  LR: {metrics['lr']:.2e}")

            # Checkpointing
            if (epoch + 1) % self.config.checkpointing.interval == 0:
                self.save_checkpoint(epoch, metrics)

        training_time = time.time() - start_time

        # Final checkpoint
        final_metrics = self.metrics_history[-1]
        self.save_checkpoint(num_epochs - 1, final_metrics, prefix='final')

        if self.verbose:
            print("\n" + "="*60)
            print(f"Training completed in {training_time/3600:.2f} hours")
            print(f"Final torsion closure: {final_metrics['torsion_closure']:.2e}")
            print(f"Final rank H²: {final_metrics['rank_h2']}/{self.h2_network.n_forms}")
            print(f"Final rank H³: {final_metrics['rank_h3']}/{self.h3_network.n_forms}")
            print("="*60)

        results = {
            'num_epochs': num_epochs,
            'training_time_hours': training_time / 3600,
            'final_metrics': final_metrics,
            'metrics_history': self.metrics_history
        }

        # Save results
        with open(self.checkpoint_dir / 'results.json', 'w') as f:
            # Remove metrics_history for JSON (too large)
            results_json = {k: v for k, v in results.items() if k != 'metrics_history'}
            json.dump(results_json, f, indent=2)

        return results

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict,
        prefix: str = 'checkpoint'
    ):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'config': self.config.to_dict(),
            'phi_network': self.phi_network.state_dict(),
            'h2_network': self.h2_network.state_dict(),
            'h3_network': self.h3_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),  # AMP scaler state
            'metrics': metrics,
            'metrics_history': self.metrics_history
        }

        path = self.checkpoint_dir / f'{prefix}_epoch_{epoch}.pt'
        torch.save(checkpoint, path)

        if self.verbose and epoch % 500 == 0:
            print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: Optional[Path] = None):
        """Load checkpoint."""
        if path is None:
            # Find latest checkpoint
            checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if not checkpoints:
                print("No checkpoint found")
                return
            path = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))

        if self.verbose:
            print(f"Loading checkpoint: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.phi_network.load_state_dict(checkpoint['phi_network'])
        self.h2_network.load_state_dict(checkpoint['h2_network'])
        self.h3_network.load_state_dict(checkpoint['h3_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        # Load AMP scaler state if available
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.metrics_history = checkpoint.get('metrics_history', [])

        if self.verbose:
            print(f"Resumed from epoch {self.start_epoch}")


__all__ = ['Trainer']
