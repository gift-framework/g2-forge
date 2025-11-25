"""
Command-line interface for g2-forge.

Usage:
    g2-forge train --config config.yaml
    g2-forge info
    g2-forge validate checkpoint.pt
"""

import argparse
import sys
from pathlib import Path


def train_command(args):
    """Run training from config file."""
    from .utils.config import G2ForgeConfig
    from .training import Trainer

    print(f"Loading config from: {args.config}")
    config = G2ForgeConfig.from_yaml(Path(args.config))

    device = args.device or ('cuda' if __import__('torch').cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    trainer = Trainer(config, device=device, verbose=True)

    if args.resume:
        trainer.load_checkpoint()

    results = trainer.train(num_epochs=args.epochs)
    print(f"\nTraining complete! Final torsion: {results['final_metrics']['torsion_closure']:.2e}")


def info_command(args):
    """Print g2-forge info."""
    from . import __version__
    import torch

    print(f"g2-forge v{__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"\nFor documentation: https://gift-framework.github.io/g2-forge/")


def validate_command(args):
    """Validate a checkpoint."""
    import torch
    from .utils.config import G2ForgeConfig

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Config version: {checkpoint['config'].get('version', 'unknown')}")

    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"\nFinal metrics:")
        print(f"  Loss: {metrics.get('loss', 'N/A'):.6f}")
        print(f"  Torsion closure: {metrics.get('torsion_closure', 'N/A'):.2e}")
        print(f"  Torsion coclosure: {metrics.get('torsion_coclosure', 'N/A'):.2e}")
        print(f"  Rank H²: {metrics.get('rank_h2', 'N/A')}")
        print(f"  Rank H³: {metrics.get('rank_h3', 'N/A')}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='g2-forge',
        description='Universal Neural Construction of G₂ Holonomy Metrics'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a G₂ metric model')
    train_parser.add_argument('--config', '-c', required=True, help='Path to config YAML file')
    train_parser.add_argument('--device', '-d', help='Device (cuda/cpu)')
    train_parser.add_argument('--epochs', '-e', type=int, help='Number of epochs (override config)')
    train_parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')

    # Info command
    info_parser = subparsers.add_parser('info', help='Print g2-forge info')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a checkpoint')
    validate_parser.add_argument('checkpoint', help='Path to checkpoint file')

    args = parser.parse_args()

    if args.command == 'train':
        train_command(args)
    elif args.command == 'info':
        info_command(args)
    elif args.command == 'validate':
        validate_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
