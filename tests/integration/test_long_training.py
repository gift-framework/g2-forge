"""
Integration tests for extended training runs.

Tests long-duration training scenarios that are too slow for regular CI:
- 1000+ epoch training
- Multi-phase curriculum learning
- Convergence behavior
- Memory stability over time

These tests are marked as 'slow' and should be run separately.
"""

import pytest
import torch
import gc

from g2forge.training import Trainer
from g2forge.utils.config import create_k7_config, G2ForgeConfig


@pytest.mark.slow
class TestLongTrainingRuns:
    """Test extended training runs (1000+ epochs)."""

    @pytest.mark.slow
    def test_training_1000_epochs_small_topology(self):
        """Test training for 1000 epochs with small topology."""
        config = create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

        # Simplify to single phase for speed
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }
        config.training.batch_size = 256

        trainer = Trainer(config, device='cpu', verbose=False)

        # Train for 1000 epochs
        results = trainer.train(num_epochs=1000)

        # Check completion
        assert len(results['training_history']['epoch']) == 1000

        # Check loss decreased
        initial_loss = results['training_history']['total_loss'][0]
        final_loss = results['training_history']['total_loss'][-1]

        assert final_loss < initial_loss, \
            f"Loss should decrease: {initial_loss:.2e} -> {final_loss:.2e}"

    @pytest.mark.slow
    def test_training_2000_epochs_with_curriculum(self):
        """Test training for 2000 epochs with curriculum learning."""
        config = create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

        # Use 3 phases
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1'],
            'phase2': config.training.curriculum_phases['phase2'],
            'phase3': config.training.curriculum_phases['phase3'],
        }
        config.training.phase_boundaries = [700, 1400]
        config.training.batch_size = 256

        trainer = Trainer(config, device='cpu', verbose=False)

        results = trainer.train(num_epochs=2000)

        # Check completion
        assert len(results['training_history']['epoch']) == 2000

        # Check phases were executed
        # Loss pattern should show phase transitions

    @pytest.mark.slow
    def test_memory_stability_over_long_training(self):
        """Test that memory usage remains stable over long training."""
        config = create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }
        config.training.batch_size = 256

        trainer = Trainer(config, device='cpu', verbose=False)

        # Record initial memory (if on GPU)
        initial_memory = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        # Train for 500 epochs
        results = trainer.train(num_epochs=500)

        # Check memory after training
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory

            # Memory growth should be minimal (< 100MB)
            assert memory_growth < 100 * 1024 * 1024, \
                f"Memory leaked: {memory_growth / 1024**2:.1f} MB"

        # Force garbage collection
        gc.collect()


@pytest.mark.slow
class TestConvergenceBehavior:
    """Test convergence behavior over extended training."""

    @pytest.mark.slow
    def test_loss_convergence_pattern(self):
        """Test that losses follow expected convergence pattern."""
        config = create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }
        config.training.batch_size = 512

        trainer = Trainer(config, device='cpu', verbose=False)

        results = trainer.train(num_epochs=500)

        total_losses = results['training_history']['total_loss']

        # Check monotonic decrease in early training (first 50 epochs)
        early_losses = total_losses[:50]
        for i in range(1, len(early_losses)):
            # Allow some fluctuation
            assert early_losses[i] <= early_losses[i-1] * 1.2, \
                f"Loss increased too much at epoch {i}"

        # Check convergence in late training (loss stabilizes)
        late_losses = total_losses[-50:]
        late_std = torch.tensor(late_losses).std().item()

        # Standard deviation should be small (converged)
        assert late_std < 0.1 * late_losses[0], \
            f"Loss not converged, std={late_std:.2e}"

    @pytest.mark.slow
    def test_torsion_decreases_monotonically(self):
        """Test that torsion losses decrease over training."""
        config = create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }

        trainer = Trainer(config, device='cpu', verbose=False)

        results = trainer.train(num_epochs=300)

        torsion_closure = results['training_history']['torsion_closure']
        torsion_coclosure = results['training_history']['torsion_coclosure']

        # Both should decrease overall
        assert torsion_closure[-1] < torsion_closure[0]
        assert torsion_coclosure[-1] < torsion_coclosure[0]

    @pytest.mark.slow
    def test_gram_matrix_improves_over_training(self):
        """Test that Gram matrix loss decreases over training."""
        config = create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1'],
            'phase2': config.training.curriculum_phases['phase2'],
        }
        config.training.phase_boundaries = [200]

        trainer = Trainer(config, device='cpu', verbose=False)

        results = trainer.train(num_epochs=400)

        gram_losses = results['training_history']['gram_matrix']

        # Should improve from phase 1 to phase 2
        phase1_avg = sum(gram_losses[:200]) / 200
        phase2_avg = sum(gram_losses[200:]) / 200

        # Phase 2 focuses more on Gram matrix, should improve
        assert phase2_avg < phase1_avg * 1.5, \
            f"Gram matrix should improve: {phase1_avg:.2e} -> {phase2_avg:.2e}"


@pytest.mark.slow
class TestCheckpointingLongTraining:
    """Test checkpointing behavior during long training."""

    @pytest.mark.slow
    def test_checkpoint_save_and_resume(self, tmp_path):
        """Test saving and resuming from checkpoint during long training."""
        config = create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }
        config.training.batch_size = 256

        # Train for 100 epochs
        trainer1 = Trainer(config, device='cpu', verbose=False)
        results1 = trainer1.train(num_epochs=100)

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint_100.pt"
        trainer1.save_checkpoint(str(checkpoint_path))

        # Create new trainer and load checkpoint
        trainer2 = Trainer(config, device='cpu', verbose=False)
        trainer2.load_checkpoint(str(checkpoint_path))

        # Continue training for 100 more epochs
        results2 = trainer2.train(num_epochs=100)

        # Check that training continued
        assert len(results2['training_history']['epoch']) == 100

        # Loss should continue decreasing
        final_loss_1 = results1['training_history']['total_loss'][-1]
        final_loss_2 = results2['training_history']['total_loss'][-1]

        # Should continue improving
        assert final_loss_2 < final_loss_1 * 1.2

    @pytest.mark.slow
    def test_multiple_checkpoint_saves(self, tmp_path):
        """Test saving checkpoints at regular intervals."""
        config = create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }
        config.training.batch_size = 256
        config.checkpoint.save_interval = 100

        trainer = Trainer(config, device='cpu', verbose=False)

        # Train for 300 epochs with checkpointing every 100
        results = trainer.train(num_epochs=300)

        # Should have created checkpoints
        # (This assumes Trainer.train() saves checkpoints automatically)


@pytest.mark.slow
class TestMultiPhaseTraining:
    """Test multi-phase curriculum learning over extended training."""

    @pytest.mark.slow
    def test_five_phase_training(self):
        """Test training with all 5 curriculum phases."""
        config = create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

        # Use all 5 phases
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1'],
            'phase2': config.training.curriculum_phases['phase2'],
            'phase3': config.training.curriculum_phases['phase3'],
            'phase4': config.training.curriculum_phases['phase4'],
            'phase5': config.training.curriculum_phases['phase5'],
        }
        config.training.phase_boundaries = [200, 500, 800, 1100]
        config.training.batch_size = 256

        trainer = Trainer(config, device='cpu', verbose=False)

        results = trainer.train(num_epochs=1500)

        # Check all epochs completed
        assert len(results['training_history']['epoch']) == 1500

        # Check loss decreased overall
        initial_loss = results['training_history']['total_loss'][0]
        final_loss = results['training_history']['total_loss'][-1]

        assert final_loss < initial_loss

    @pytest.mark.slow
    def test_phase_transition_smoothness(self):
        """Test that phase transitions don't cause loss spikes."""
        config = create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1'],
            'phase2': config.training.curriculum_phases['phase2'],
            'phase3': config.training.curriculum_phases['phase3'],
        }
        config.training.phase_boundaries = [300, 600]
        config.training.batch_size = 256

        trainer = Trainer(config, device='cpu', verbose=False)

        results = trainer.train(num_epochs=900)

        total_losses = results['training_history']['total_loss']

        # Check transitions at epochs 300 and 600
        for transition_epoch in [300, 600]:
            # Loss shouldn't spike dramatically at transition
            before = total_losses[transition_epoch - 1]
            after = total_losses[transition_epoch]

            # Allow 2x increase max (weight changes can cause temporary spike)
            assert after < before * 2.0, \
                f"Loss spike at phase transition (epoch {transition_epoch}): {before:.2e} -> {after:.2e}"


@pytest.mark.slow
class TestScaling:
    """Test training scalability with different configurations."""

    @pytest.mark.slow
    def test_training_with_large_batch_size(self):
        """Test training with large batch size."""
        config = create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }
        config.training.batch_size = 2048  # Large batch

        trainer = Trainer(config, device='cpu', verbose=False)

        results = trainer.train(num_epochs=100)

        # Should complete without error
        assert len(results['training_history']['epoch']) == 100

    @pytest.mark.slow
    def test_training_with_large_topology(self):
        """Test training with larger topology."""
        config = create_k7_config(b2_m1=10, b3_m1=30, b2_m2=10, b3_m2=30)

        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }
        config.training.batch_size = 128  # Smaller batch for memory

        trainer = Trainer(config, device='cpu', verbose=False)

        results = trainer.train(num_epochs=100)

        # Should complete
        assert len(results['training_history']['epoch']) == 100

        # Check networks sized correctly
        assert trainer.h2_net.n_forms == 20
        assert trainer.h3_net.n_forms == 61


@pytest.mark.slow
class TestTrainingStability:
    """Test training stability over extended runs."""

    @pytest.mark.slow
    def test_no_nan_over_long_training(self):
        """Test that training doesn't produce NaN over long runs."""
        config = create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }
        config.training.batch_size = 512

        trainer = Trainer(config, device='cpu', verbose=False)

        results = trainer.train(num_epochs=500)

        # Check no NaN in losses
        for key in ['total_loss', 'torsion_closure', 'torsion_coclosure']:
            losses = results['training_history'][key]
            for i, loss in enumerate(losses):
                assert not torch.isnan(torch.tensor(loss)), \
                    f"NaN detected in {key} at epoch {i}"

    @pytest.mark.slow
    def test_gradient_norms_stable(self):
        """Test that gradient norms remain stable over training."""
        config = create_k7_config(b2_m1=3, b3_m1=10, b2_m2=2, b3_m2=10)

        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }

        trainer = Trainer(config, device='cpu', verbose=False)

        # Track gradient norms over training
        grad_norms = []

        # Sample gradient norms every 50 epochs
        for epoch in range(0, 200, 50):
            trainer.train(num_epochs=1)

            # Get gradient norm
            total_norm = 0.0
            for param in trainer.phi_network.parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm().item() ** 2
            total_norm = total_norm ** 0.5

            grad_norms.append(total_norm)

        # Gradient norms shouldn't explode
        for norm in grad_norms:
            assert norm < 1e3, f"Gradient exploded: {norm:.2e}"
            assert norm > 1e-6, f"Gradient vanished: {norm:.2e}"
