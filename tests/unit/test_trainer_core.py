"""
Unit tests for Trainer core methods, especially the train() method.

Tests the main training loop behavior at unit level, complementing
integration tests with focused verification of:
- Metrics history accumulation
- Phase transitions
- Gradient accumulation
- Result structure
- Early stopping behavior
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

from g2forge.training import Trainer
from g2forge.utils import G2ForgeConfig, TopologyConfig


class TestTrainerTrainMethod:
    """Tests for Trainer.train() method behavior."""

    def test_train_returns_complete_results_dict(self, small_topology_config):
        """Test that train() returns all expected keys in results dict."""
        config = small_topology_config
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }

        trainer = Trainer(config, device='cpu', verbose=False)
        results = trainer.train(num_epochs=5)

        # Check required keys
        assert 'final_metrics' in results
        assert 'training_history' in results
        assert 'config' in results

        # Check final_metrics structure
        final_metrics = results['final_metrics']
        assert 'total_loss' in final_metrics
        assert 'torsion_closure' in final_metrics
        assert 'torsion_coclosure' in final_metrics

        # Check training_history structure
        history = results['training_history']
        assert 'epoch' in history
        assert 'total_loss' in history
        assert len(history['epoch']) == 5
        assert len(history['total_loss']) == 5

    def test_train_accumulates_metrics_history(self, small_topology_config):
        """Test that training history is correctly accumulated over epochs."""
        config = small_topology_config
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }

        trainer = Trainer(config, device='cpu', verbose=False)
        results = trainer.train(num_epochs=10)

        history = results['training_history']

        # Check all epochs recorded
        assert len(history['epoch']) == 10
        assert history['epoch'] == list(range(10))

        # Check all loss components recorded
        assert 'total_loss' in history
        assert 'torsion_closure' in history
        assert 'torsion_coclosure' in history
        assert 'gram_matrix' in history
        assert 'volume' in history

        # Check all have same length
        for key in history:
            assert len(history[key]) == 10, f"{key} has wrong length"

        # Check values are numeric
        for loss_val in history['total_loss']:
            assert isinstance(loss_val, float)
            assert not torch.isnan(torch.tensor(loss_val))

    def test_train_phase_transitions(self, small_topology_config):
        """Test curriculum phase transitions occur at correct epochs."""
        config = small_topology_config

        # Set up two phases with clear boundary
        from g2forge.utils.config import CurriculumPhaseConfig

        phase1 = CurriculumPhaseConfig(
            grid_n=10,
            loss_weights={
                'torsion_closure': 5.0,
                'torsion_coclosure': 5.0,
                'gram_matrix': 0.1,
                'volume': 0.5,
            }
        )

        phase2 = CurriculumPhaseConfig(
            grid_n=15,
            loss_weights={
                'torsion_closure': 1.0,
                'torsion_coclosure': 1.0,
                'gram_matrix': 5.0,  # Much higher in phase 2
                'volume': 0.5,
            }
        )

        config.training.curriculum_phases = {
            'phase1': phase1,
            'phase2': phase2,
        }
        config.training.phase_boundaries = [5]  # Switch at epoch 5

        trainer = Trainer(config, device='cpu', verbose=False)

        # Test phase detection
        assert trainer._get_current_phase(0) == 'phase1'
        assert trainer._get_current_phase(4) == 'phase1'
        assert trainer._get_current_phase(5) == 'phase2'
        assert trainer._get_current_phase(10) == 'phase2'

    def test_train_losses_decrease(self, small_topology_config):
        """Test that losses generally decrease over training."""
        config = small_topology_config
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }

        trainer = Trainer(config, device='cpu', verbose=False)
        results = trainer.train(num_epochs=20)

        history = results['training_history']
        total_losses = history['total_loss']

        # Check first half average vs second half average
        first_half_avg = sum(total_losses[:10]) / 10
        second_half_avg = sum(total_losses[10:]) / 10

        # Loss should decrease (or at least not increase significantly)
        assert second_half_avg <= first_half_avg * 1.1, \
            f"Loss increased: {first_half_avg:.2e} -> {second_half_avg:.2e}"

    def test_train_handles_multiple_phases(self, small_topology_config):
        """Test training with multiple curriculum phases."""
        config = small_topology_config

        # Keep first 3 phases
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1'],
            'phase2': config.training.curriculum_phases['phase2'],
            'phase3': config.training.curriculum_phases['phase3'],
        }
        config.training.phase_boundaries = [5, 10]  # Two boundaries

        trainer = Trainer(config, device='cpu', verbose=False)
        results = trainer.train(num_epochs=15)

        # Should complete without error
        assert len(results['training_history']['epoch']) == 15

    def test_train_gradient_updates_weights(self, small_topology_config):
        """Test that training actually updates network weights."""
        config = small_topology_config
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }

        trainer = Trainer(config, device='cpu', verbose=False)

        # Store initial weights
        initial_phi_weights = {
            name: param.clone().detach()
            for name, param in trainer.phi_network.named_parameters()
        }

        # Train for a few epochs
        trainer.train(num_epochs=5)

        # Check weights changed
        weights_changed = False
        for name, param in trainer.phi_network.named_parameters():
            if not torch.allclose(param, initial_phi_weights[name], atol=1e-8):
                weights_changed = True
                break

        assert weights_changed, "Network weights did not update during training"

    def test_train_respects_device_placement(self, small_topology_config):
        """Test that training respects device configuration."""
        config = small_topology_config
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }

        trainer = Trainer(config, device='cpu', verbose=False)
        results = trainer.train(num_epochs=3)

        # Check networks are on CPU
        for param in trainer.phi_network.parameters():
            assert param.device.type == 'cpu'

    @pytest.mark.parametrize("num_epochs", [1, 5, 10])
    def test_train_completes_requested_epochs(self, small_topology_config, num_epochs):
        """Test that training runs for exactly the requested number of epochs."""
        config = small_topology_config
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }

        trainer = Trainer(config, device='cpu', verbose=False)
        results = trainer.train(num_epochs=num_epochs)

        assert len(results['training_history']['epoch']) == num_epochs


class TestTrainerInternalMethods:
    """Tests for Trainer internal helper methods."""

    def test_get_current_phase_single_phase(self, small_topology_config):
        """Test phase detection with single phase."""
        config = small_topology_config
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }
        config.training.phase_boundaries = []

        trainer = Trainer(config, device='cpu', verbose=False)

        assert trainer._get_current_phase(0) == 'phase1'
        assert trainer._get_current_phase(100) == 'phase1'

    def test_get_current_phase_multiple_phases(self, small_topology_config):
        """Test phase detection with multiple phases."""
        config = small_topology_config
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1'],
            'phase2': config.training.curriculum_phases['phase2'],
            'phase3': config.training.curriculum_phases['phase3'],
        }
        config.training.phase_boundaries = [10, 20]

        trainer = Trainer(config, device='cpu', verbose=False)

        # Check boundaries
        assert trainer._get_current_phase(0) == 'phase1'
        assert trainer._get_current_phase(9) == 'phase1'
        assert trainer._get_current_phase(10) == 'phase2'
        assert trainer._get_current_phase(19) == 'phase2'
        assert trainer._get_current_phase(20) == 'phase3'
        assert trainer._get_current_phase(100) == 'phase3'

    def test_optimizer_configuration(self, small_topology_config):
        """Test that optimizer is configured correctly."""
        config = small_topology_config
        trainer = Trainer(config, device='cpu', verbose=False)

        # Check optimizer type
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

        # Check learning rate
        assert trainer.optimizer.param_groups[0]['lr'] == config.training.learning_rate

        # Check weight decay
        assert trainer.optimizer.param_groups[0]['weight_decay'] == config.training.weight_decay

    def test_scheduler_configuration(self, small_topology_config):
        """Test that learning rate scheduler is configured correctly."""
        config = small_topology_config
        config.training.warmup_epochs = 5

        trainer = Trainer(config, device='cpu', verbose=False)

        # Should have a scheduler
        assert hasattr(trainer, 'scheduler')
        assert trainer.scheduler is not None


class TestTrainerMetricsComputation:
    """Tests for metrics computation during training."""

    def test_metrics_include_all_loss_components(self, small_topology_config):
        """Test that metrics include all individual loss components."""
        config = small_topology_config
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }

        trainer = Trainer(config, device='cpu', verbose=False)
        results = trainer.train(num_epochs=3)

        history = results['training_history']

        # Check all expected loss components
        expected_components = [
            'total_loss',
            'torsion_closure',
            'torsion_coclosure',
            'gram_matrix',
            'volume',
        ]

        for component in expected_components:
            assert component in history, f"Missing loss component: {component}"

    def test_final_metrics_computed(self, small_topology_config):
        """Test that final metrics are computed at end of training."""
        config = small_topology_config
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }

        trainer = Trainer(config, device='cpu', verbose=False)
        results = trainer.train(num_epochs=5)

        final_metrics = results['final_metrics']

        # Check structure
        assert isinstance(final_metrics, dict)
        assert len(final_metrics) > 0

        # Check key metrics
        assert 'total_loss' in final_metrics
        assert 'torsion_closure' in final_metrics


class TestTrainerEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_train_with_very_small_batch(self, small_topology_config):
        """Test training with very small batch size."""
        config = small_topology_config
        config.training.batch_size = 10  # Very small
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }

        trainer = Trainer(config, device='cpu', verbose=False)
        results = trainer.train(num_epochs=3)

        # Should complete without error
        assert len(results['training_history']['epoch']) == 3

    def test_train_preserves_config(self, small_topology_config):
        """Test that training results include original config."""
        config = small_topology_config
        config.training.curriculum_phases = {
            'phase1': config.training.curriculum_phases['phase1']
        }

        trainer = Trainer(config, device='cpu', verbose=False)
        results = trainer.train(num_epochs=3)

        # Config should be in results
        assert 'config' in results
        assert results['config'] == config
