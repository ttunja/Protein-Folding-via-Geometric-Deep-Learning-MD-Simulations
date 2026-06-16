"""Parse training logs and generate experiment comparison plots."""

import re
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_train_log(log_path: str) -> dict:
    """Extract metrics from a training log file."""
    metrics = {
        'epoch_losses': [],
        'val_losses': [],
        'lr_history': [],
        'batch_losses': [],   # per-1000-batch summed losses
        'batch_baseline': [], # per-1000-batch baseline losses
        'timestamps': [],
        'start_time': None,
        'end_time': None,
    }

    with open(log_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        ts_match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        timestamp = ts_match.group(1) if ts_match else None

        if 'Training started' in line and timestamp:
            metrics['start_time'] = timestamp

        if 'Epoch loss=' in line:
            val = float(re.search(r'Epoch loss=([\d.]+)', line).group(1))
            metrics['epoch_losses'].append(val)
            if timestamp:
                metrics['end_time'] = timestamp

        if 'Validation loss=' in line:
            val = float(re.search(r'Validation loss=([\d.]+)', line).group(1))
            metrics['val_losses'].append(val)

        if 'LR: ' in line:
            val = float(re.search(r'LR: ([\d.e+-]+)', line).group(1))
            metrics['lr_history'].append(val)

        if 'Summed loss equals' in line and 'Baseline' not in line:
            val = float(re.search(r'Summed loss equals ([\d.]+)', line).group(1))
            metrics['batch_losses'].append(val)

        if 'Summed Baseline loss equals' in line:
            val = float(re.search(r'Summed Baseline loss equals ([\d.]+)', line).group(1))
            metrics['batch_baseline'].append(val)

    return metrics


def load_experiment_registry(registry_path: str) -> dict:
    """Load or create the experiment registry."""
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            return json.load(f)
    return {'experiments': {}}


def save_experiment_registry(registry: dict, registry_path: str):
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)


def register_experiment(registry_path: str, name: str, log_path: str,
                        config_path: str = None, description: str = ''):
    """Register an experiment in the tracker."""
    registry = load_experiment_registry(registry_path)
    registry['experiments'][name] = {
        'log_path': log_path,
        'config_path': config_path,
        'description': description,
        'registered_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    save_experiment_registry(registry, registry_path)
    print(f'Registered experiment: {name}')


def plot_comparison(registry_path: str, output_dir: str, experiment_names: list = None):
    """Generate comparison plots for registered experiments."""
    registry = load_experiment_registry(registry_path)
    os.makedirs(output_dir, exist_ok=True)

    experiments = {}
    for name, info in registry['experiments'].items():
        if experiment_names and name not in experiment_names:
            continue
        if not os.path.exists(info['log_path']):
            print(f'Warning: log not found for {name}: {info["log_path"]}')
            continue
        experiments[name] = parse_train_log(info['log_path'])

    if not experiments:
        print('No experiments to plot.')
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(experiments), 10)))

    # --- 1. Epoch train loss ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, m) in enumerate(experiments.items()):
        if m['epoch_losses']:
            epochs = range(1, len(m['epoch_losses']) + 1)
            ax.plot(epochs, m['epoch_losses'], '-o', color=colors[i], label=name, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss (sum)')
    ax.set_title('Training Loss per Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'epoch_train_loss.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- 2. Validation loss ---
    fig, ax = plt.subplots(figsize=(10, 6))
    has_val = False
    for i, (name, m) in enumerate(experiments.items()):
        if m['val_losses']:
            has_val = True
            epochs = range(1, len(m['val_losses']) + 1)
            ax.plot(epochs, m['val_losses'], '-o', color=colors[i], label=name, markersize=4)
    if has_val:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss per Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, 'epoch_val_loss.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- 3. Batch-level training loss (smoothed) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, m) in enumerate(experiments.items()):
        if m['batch_losses']:
            x = np.arange(len(m['batch_losses']))
            y = np.array(m['batch_losses'])
            ax.plot(x, y, color=colors[i], alpha=0.3, linewidth=0.5)
            # smoothed (rolling average, window=10)
            if len(y) > 10:
                kernel = np.ones(10) / 10
                y_smooth = np.convolve(y, kernel, mode='valid')
                ax.plot(np.arange(len(y_smooth)) + 5, y_smooth, color=colors[i], label=name, linewidth=1.5)
            else:
                ax.plot(x, y, color=colors[i], label=name)
    ax.set_xlabel('Batch checkpoint (x1000)')
    ax.set_ylabel('Summed Loss (per 1000 batches)')
    ax.set_title('Training Loss (batch-level)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'batch_train_loss.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- 4. Loss vs baseline ratio ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, m) in enumerate(experiments.items()):
        if m['batch_losses'] and m['batch_baseline']:
            n = min(len(m['batch_losses']), len(m['batch_baseline']))
            ratio = np.array(m['batch_losses'][:n]) / (np.array(m['batch_baseline'][:n]) + 1e-8)
            ax.plot(range(n), ratio, color=colors[i], alpha=0.3, linewidth=0.5)
            if n > 10:
                kernel = np.ones(10) / 10
                ratio_smooth = np.convolve(ratio, kernel, mode='valid')
                ax.plot(np.arange(len(ratio_smooth)) + 5, ratio_smooth, color=colors[i], label=name, linewidth=1.5)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='baseline (zero pred)')
    ax.set_xlabel('Batch checkpoint (x1000)')
    ax.set_ylabel('Loss / Baseline Loss')
    ax.set_title('Loss Ratio vs Zero-Prediction Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'loss_vs_baseline.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- 5. LR schedule ---
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (name, m) in enumerate(experiments.items()):
        if m['lr_history']:
            ax.plot(range(1, len(m['lr_history']) + 1), m['lr_history'], '-o', color=colors[i], label=name, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, 'lr_schedule.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Summary table ---
    print(f'\n{"Experiment":<30} {"Epochs":>7} {"Final Train":>12} {"Best Val":>10} {"Final LR":>10}')
    print('-' * 75)
    for name, m in experiments.items():
        n_ep = len(m['epoch_losses'])
        final_train = f"{m['epoch_losses'][-1]:.1f}" if m['epoch_losses'] else '-'
        best_val = f"{min(m['val_losses']):.4f}" if m['val_losses'] else '-'
        final_lr = f"{m['lr_history'][-1]:.1e}" if m['lr_history'] else '-'
        print(f'{name:<30} {n_ep:>7} {final_train:>12} {best_val:>10} {final_lr:>10}')

    print(f'\nPlots saved to {output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment tracking and comparison')
    sub = parser.add_subparsers(dest='command')

    reg = sub.add_parser('register', help='Register an experiment')
    reg.add_argument('--name', required=True)
    reg.add_argument('--log', required=True)
    reg.add_argument('--config', default=None)
    reg.add_argument('--desc', default='')

    plot = sub.add_parser('plot', help='Plot experiment comparisons')
    plot.add_argument('--experiments', nargs='*', default=None)

    sub.add_parser('list', help='List registered experiments')

    args = parser.parse_args()

    registry_path = os.path.join(os.path.dirname(__file__), '../../docs/experiments.json')
    output_dir = os.path.join(os.path.dirname(__file__), '../../docs/plots')

    if args.command == 'register':
        register_experiment(registry_path, args.name, args.log, args.config, args.desc)
    elif args.command == 'plot':
        plot_comparison(registry_path, output_dir, args.experiments)
    elif args.command == 'list':
        reg = load_experiment_registry(registry_path)
        for name, info in reg['experiments'].items():
            print(f"  {name}: {info.get('description', '')} ({info.get('registered_at', '')})")
    else:
        parser.print_help()
