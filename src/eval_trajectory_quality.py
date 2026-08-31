"""Trajectory quality evaluation: bond lengths, collisions, and the
fraction-of-native-contacts (Q) distribution against reference MD.

Two figures are produced per protein:
  - quality_<protein>.png:        bond length and collision rate, single
                                  short rollout from the K=1 gru_only model
                                  (Section 6.4 in the thesis).
  - q_distribution_<protein>.png: Q histogram, three long rollouts from the
                                  K=5 multi-step model vs MD reference
                                  (Section 6.4 in the thesis).

Usage:
  python eval_trajectory_quality.py --protein chignolin
  python eval_trajectory_quality.py --protein chignolin --skip_q
  python eval_trajectory_quality.py --protein chignolin --skip_quality
  python eval_trajectory_quality.py --protein all
"""

import argparse
import os
import sys
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.utils import (compute_weighted_rgs, compute_com, compute_distance_matrix,
                          compute_ramachandran_angles, compute_dihedral_angle)
from preprocessing.preprocessing_utils import load_data
from train import seed_everything
from eval_folding import (
    PROTEIN_CONFIGS as FOLDING_CONFIGS,
    build_batch,
    get_native_cg,
    compute_native_contacts,
    find_unfolded_start,
)


PROTEIN_CONFIGS = {
    'chignolin': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Chignolin/protein_train',
        'model_path': '/home/ubuntu/models/PROTEIN-FOLDING-MODELS/Chignolin/chignolin_gru_only/model_best',
        'multistep_path': '/home/ubuntu/models/PROTEIN-FOLDING-MODELS/Chignolin/chignolin_gru_multistep/model_best',
        'test_trajs': [f'traj_{i}' for i in range(38, 48)],
        'n_beads': 10,
    },
    'trp_cage': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Trp-cage/protein_train',
        'model_path': '/home/ubuntu/models/PROTEIN-FOLDING-MODELS/Trp-cage/trp_cage_gru_only/model_best',
        'multistep_path': '/home/ubuntu/models/PROTEIN-FOLDING-MODELS/Trp-cage/trp_cage_gru_multistep/model_best',
        'test_trajs': [f'traj_{i}' for i in range(83, 93)],
        'n_beads': 20,
    },
    'bba': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/BBA/protein_train',
        'model_path': '/home/ubuntu/models/PROTEIN-FOLDING-MODELS/BBA/bba_gru_only/model_best',
        'multistep_path': None,
        'test_trajs': [f'traj_{i}' for i in range(89, 99)],
        'n_beads': 28,
    },
    'villin': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Villin/protein_train',
        'model_path': '/home/ubuntu/models/PROTEIN-FOLDING-MODELS/Villin/villin_gru_only/model_best',
        'multistep_path': None,
        'test_trajs': [f'traj_{i}' for i in range(44, 54)],
        'n_beads': 35,
    },
    # unseen proteins - auto-discover trajectories since we never train on these
    'ntl9': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/NTL9/protein_train',
        'model_path': '/home/ubuntu/models/PROTEIN-FOLDING-MODELS/Combined/combined_gru_multistep/model_best',
        'test_trajs': None,
        'n_beads': 39,
    },
    'ww': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/WW/protein_train',
        'model_path': '/home/ubuntu/models/PROTEIN-FOLDING-MODELS/Combined/combined_gru_multistep/model_best',
        'test_trajs': None,
        'n_beads': 35,
    },
    'protein_b': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Protein_B/protein_train',
        'model_path': '/home/ubuntu/models/PROTEIN-FOLDING-MODELS/Combined/combined_gru_multistep/model_best',
        'test_trajs': None,
        'n_beads': 47,
    },
}


def get_cg_trajectory(data_dir: str, traj_name: str, n_beads: int,
                      start: int = 0, end: int = 1000):
    """Load a trajectory and coarse-grain it. Returns [n_beads, n_frames, 3]."""
    tp = os.path.join(data_dir, traj_name)
    position = np.array(load_data(['position'], f'{tp}/position.h5')[0], dtype=np.float32)
    residue_num = load_data(['residue_num'], f'{tp}/residue_num.h5')[0]
    cluster = torch.tensor(residue_num - min(residue_num), dtype=torch.long)
    cg_pos, _ = compute_com(torch.tensor(position[start:end]).permute(1, 0, 2),
                            torch.ones(position.shape[1]), cluster)
    return np.array(cg_pos)  # [n_beads, n_frames, 3]


def get_model_trajectory(model_path: str, data_dir: str, traj_name: str,
                         start_frame: int, n_beads: int, rollout_length: int = 2000,
                         max_vel: float = 3.0, seed: int = 42):
    """Generate a trajectory from the model. Returns [n_beads, n_frames, 3]."""
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()

    tp = os.path.join(data_dir, traj_name)
    position = np.array(load_data(['position'], f'{tp}/position.h5')[0], dtype=np.float32)
    ptype = load_data(['particle_type'], f'{tp}/ptype.h5')[0]
    bonds = load_data(['bond_indices'], f'{tp}/bond.h5')[0]
    rgs_data = load_data(['rgs'], f'{tp}/rgs.h5')[0]
    residue_num = load_data(['residue_num'], f'{tp}/residue_num.h5')[0]
    cluster = torch.tensor(residue_num - min(residue_num), dtype=torch.long)

    sel = np.arange(start_frame, start_frame + 31)
    rev_bonds = np.concatenate([bonds[:, 1:], bonds[:, :1]], axis=1)
    ab = torch.tensor(np.concatenate([bonds, rev_bonds], axis=0), dtype=torch.long)
    row, col = ab[:, 0], ab[:, 1]
    boundary = cluster[row] != cluster[col]
    cg_bonds = torch.unique(
        torch.cat([cluster[row[boundary]][:, None], cluster[col[boundary]][:, None]], dim=1), dim=0)

    batch = {
        'position': torch.tensor(position[sel[:-1]], dtype=torch.float32).permute(1, 0, 2),
        'target': torch.tensor(position[sel[-1]], dtype=torch.float32),
        'particle_types': torch.tensor(ptype, dtype=torch.long),
        'n_particle': torch.tensor([ptype.shape[0]], dtype=torch.long),
        'bonds': ab, 'n_bond': torch.tensor([ab.shape[0]], dtype=torch.long),
        'rgs': torch.tensor(rgs_data[sel[:-1]], dtype=torch.float32).unsqueeze(0),
        'residue_num': torch.tensor(residue_num, dtype=torch.long),
        'n_keypoint': torch.tensor([n_beads], dtype=torch.long),
        'cluster': cluster, 'cg_bonds': cg_bonds,
        'n_cg_bond': torch.tensor([len(cg_bonds)], dtype=torch.long),
    }

    seed_everything(seed)
    with torch.no_grad():
        out = model.simulate(batch, rollout_length, save_positions=True, deter=False,
                            save_frequency=1, disable_bar=True, max_vel=max_vel)
    return np.array(out['rollout_u_pos'][:n_beads])  # [n_beads, n_frames, 3]


def compute_bond_lengths(cg_positions):
    """Compute consecutive bead-bead distances over time.
    cg_positions: [n_beads, n_frames, 3]
    Returns: [n_bonds, n_frames] array of distances."""
    n_beads = cg_positions.shape[0]
    bond_lengths = []
    for i in range(n_beads - 1):
        d = np.sqrt(((cg_positions[i] - cg_positions[i + 1]) ** 2).sum(axis=-1))
        bond_lengths.append(d)
    return np.array(bond_lengths)


def compute_collision_rate(cg_positions, threshold: float = 3.0):
    """Compute fraction of bead pairs closer than threshold per frame.
    cg_positions: [n_beads, n_frames, 3]"""
    n_beads, n_frames, _ = cg_positions.shape
    n_pairs = n_beads * (n_beads - 1) / 2
    collisions = []
    for t in range(n_frames):
        dm = np.sqrt(((cg_positions[:, t:t+1] - cg_positions[:, t:t+1].transpose(1, 0, 2)) ** 2).sum(-1))
        # count pairs with d < threshold, excluding diagonal and i>j
        n_close = np.sum(np.triu(dm < threshold, k=1))
        collisions.append(n_close / n_pairs)
    return np.array(collisions)


def evaluate_protein(protein_name: str, output_dir: str, rollout_length: int = 2000,
                     max_vel: float = 3.0):
    """Full trajectory quality evaluation for one protein."""
    cfg = PROTEIN_CONFIGS[protein_name]
    n_beads = cfg['n_beads']
    logging.info(f'\n{"="*60}')
    logging.info(f'  {protein_name} ({n_beads} beads) - trajectory quality')
    logging.info(f'{"="*60}')

    # auto-discover trajectories if not specified
    test_trajs = cfg['test_trajs']
    if test_trajs is None:
        test_trajs = sorted([d for d in os.listdir(cfg['data_dir'])
                            if os.path.isdir(os.path.join(cfg['data_dir'], d)) and d.startswith('traj_')])

    # collect MD reference data from multiple test trajectories
    md_phi_all, md_psi_all = [], []
    md_bond_lengths_all = []
    md_collision_all = []

    for traj_name in test_trajs[:5]:
        cg_pos = get_cg_trajectory(cfg['data_dir'], traj_name, n_beads, start=30, end=530)
        phi, psi = compute_ramachandran_angles(cg_pos)
        md_phi_all.extend(phi)
        md_psi_all.extend(psi)
        bl = compute_bond_lengths(cg_pos)
        md_bond_lengths_all.append(bl.flatten())
        cr = compute_collision_rate(cg_pos)
        md_collision_all.extend(cr)

    md_phi = np.array(md_phi_all)
    md_psi = np.array(md_psi_all)
    md_bonds = np.concatenate(md_bond_lengths_all)
    md_collisions = np.array(md_collision_all)

    # generate predicted trajectory
    logging.info('  Generating model trajectory...')
    pred_pos = get_model_trajectory(cfg['model_path'], cfg['data_dir'],
                                    test_trajs[0], start_frame=100,
                                    n_beads=n_beads, rollout_length=rollout_length,
                                    max_vel=max_vel)
    pred_phi, pred_psi = compute_ramachandran_angles(pred_pos)
    pred_bonds = compute_bond_lengths(pred_pos).flatten()
    pred_collisions = compute_collision_rate(pred_pos)

    # print stats. The extremes are reported alongside the means because the
    # distributions are right-skewed, so the mean alone hides where the model
    # and the reference actually part company.
    logging.info(f'\n  Consecutive bead distance:')
    for name, b in (('MD   ', md_bonds), ('Model', pred_bonds)):
        counts, edges = np.histogram(b.ravel(), bins=60)
        peak = edges[np.argmax(counts)]
        logging.info(f'    {name}: mean={b.mean():.2f} ± {b.std():.2f} Å  '
                     f'peak={peak:.2f}  min={b.min():.2f}  max={b.max():.2f}')

    logging.info(f'\n  Collision rate (d < 3Å):')
    for name, c in (('MD   ', md_collisions), ('Model', pred_collisions)):
        logging.info(f'    {name}: mean={c.mean():.4f}  max={c.max():.4f}')

    logging.info(f'\n  Ramachandran angles:')
    logging.info(f'    MD:    φ mean={md_phi.mean():.1f}°, ψ mean={md_psi.mean():.1f}°')
    logging.info(f'    Model: φ mean={np.array(pred_phi).mean():.1f}°, ψ mean={np.array(pred_psi).mean():.1f}°')

    # ---- PLOTS ----
    os.makedirs(output_dir, exist_ok=True)
    from utils.plot_style import apply_thesis_style, smooth, THESIS_PALETTE, MD_COLOR
    apply_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    md_bonds_mean = float(np.mean(md_bonds))
    md_bonds_std = float(np.std(md_bonds))

    # (a) Bond length distributions
    ax = axes[0]
    bins = np.linspace(min(md_bonds.min(), pred_bonds.min()),
                       max(md_bonds.max(), pred_bonds.max()), 70)
    ax.hist(md_bonds, bins=bins, alpha=0.55, color=MD_COLOR, density=True,
            label='MD reference', edgecolor=MD_COLOR, linewidth=0.4)
    ax.hist(pred_bonds, bins=bins, alpha=0.55, color=THESIS_PALETTE[0], density=True,
            label='Model', edgecolor=THESIS_PALETTE[0], linewidth=0.4)
    ax.axvline(md_bonds_mean, color=MD_COLOR, ls='--', lw=1.4,
               label=f'MD mean ({md_bonds_mean:.2f} Å)')
    ax.set_xlabel('Consecutive bead distance (Å)')
    ax.set_ylabel('Density')
    # not "bond length": a bead is a residue centre of mass, not a backbone atom,
    # so this distance is longer and more variable than a covalent bond
    ax.set_title('(a) Consecutive bead distance')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.25)

    # (b) Collision rate over time, smoothed
    ax = axes[1]
    md_coll_mean = float(md_collisions.mean())
    md_coll_std = float(md_collisions.std())
    n_steps = len(pred_collisions)
    x = np.arange(n_steps) * 200e-6  # μs
    pred_coll = np.asarray(pred_collisions, dtype=float)
    pred_smooth = smooth(pred_coll, frac=0.02)

    # Show raw spikes more prominently so peaks above the smoothed line are visible.
    ax.plot(x, pred_coll, color=THESIS_PALETTE[0], alpha=0.35, lw=0.8,
            label='Model (raw)', zorder=1)
    ax.plot(x, pred_smooth, color=THESIS_PALETTE[0], alpha=1.0, lw=2.0,
            label='Model (smoothed)', zorder=3)
    ax.axhline(md_coll_mean, color=MD_COLOR, ls='--', lw=1.4,
               label=f'MD mean ({md_coll_mean:.3f})')
    ax.fill_between(x,
                    max(0.0, md_coll_mean - md_coll_std),
                    md_coll_mean + md_coll_std,
                    color=MD_COLOR, alpha=0.12, label='MD ±1σ', zorder=0)
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Collision rate (fraction of bead pairs)')
    # name the threshold in the title: at 5 Å the backbone neighbours dominate
    ax.set_title('(b) Bead collision rate ($d < 3$ Å)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=max(pred_coll.max() * 1.05, 0.1))

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'quality_{protein_name}.png'),
                bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  Plot saved to {output_dir}/quality_{protein_name}.png')


def _md_q_series(data_dir, traj_name, native_dm, n_frames):
    """Q time series for an MD trajectory. n_frames=0 means use the whole trajectory."""
    tp = os.path.join(data_dir, traj_name)
    full = np.array(load_data(['position'], f'{tp}/position.h5')[0], dtype=np.float32)
    if n_frames > 0:
        position = full[:n_frames + 30]
    else:
        position = full
    residue_num = load_data(['residue_num'], f'{tp}/residue_num.h5')[0]
    cluster = torch.tensor(residue_num - min(residue_num), dtype=torch.long)
    cg_pos, _ = compute_com(torch.tensor(position).permute(1, 0, 2),
                            torch.ones(position.shape[1]), cluster)
    start = 30
    end = min(position.shape[0], 30 + n_frames) if n_frames > 0 else position.shape[0]
    qs = [compute_native_contacts(cg_pos[:, t], native_dm) for t in range(start, end)]
    return np.array(qs)


def _model_q_series(model_path, batch, native_dm, n_steps, seed, tail):
    """Q time series for an autoregressive model rollout.

    If `tail > 0`, only the last `tail` frames are returned; otherwise the
    whole rollout (all `n_steps` predicted frames) is used.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    b = {k: v.clone().to(device) for k, v in batch.items()}
    seed_everything(seed)
    with torch.no_grad():
        out = model.simulate(b, n_steps, save_positions=True, deter=False,
                             save_frequency=1, disable_bar=True, max_vel=3.0)
    pred = out['rollout_u_pos'].cpu()
    n_beads = batch['n_keypoint'].item()
    pred = pred[:n_beads]
    start = max(0, pred.shape[1] - tail) if tail > 0 else 0
    qs = [compute_native_contacts(pred[:, t], native_dm) for t in range(start, pred.shape[1])]
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return np.array(qs)


def evaluate_q_distribution(protein_name: str, output_dir: str,
                             n_rollouts: int = 1, rollout_steps: int = 10000,
                             rollout_tail: int = 0, md_trajs: int = 3,
                             md_frames: int = 0, model_path: str = None):
    """Q-distribution figure: K=5 multi-step model vs MD reference."""
    cfg = PROTEIN_CONFIGS[protein_name]
    n_beads = cfg['n_beads']
    mpath = model_path or cfg.get('multistep_path')
    if not mpath or not os.path.exists(mpath):
        logging.warning(f'  No multistep model for {protein_name} ({mpath}); '
                        'skipping Q-distribution.')
        return

    logging.info(f'\n{"="*60}')
    logging.info(f'  {protein_name} ({n_beads} beads) - Q distribution')
    logging.info(f'{"="*60}')

    fcfg = FOLDING_CONFIGS[protein_name]
    _, native_dm, _ = get_native_cg(fcfg['native_pdb'])

    test_trajs = cfg['test_trajs'] or sorted(
        d for d in os.listdir(cfg['data_dir'])
        if os.path.isdir(os.path.join(cfg['data_dir'], d)) and d.startswith('traj_'))

    md_q = []
    for traj in test_trajs[:md_trajs]:
        q = _md_q_series(cfg['data_dir'], traj, native_dm, md_frames)
        md_q.append(q)
        logging.info(f'  MD {traj}: Q mean {q.mean():.3f} (n={len(q)})')
    md_q_all = np.concatenate(md_q)

    # Start each model rollout from the same conformation that the corresponding
    # MD trajectory starts from. This makes the comparison apples-to-apples:
    # both distributions cover the same total simulated time from the same
    # initial configurations, so any divergence is the model's, not the
    # starting condition's.
    plot_trajs = test_trajs[:md_trajs]
    seeds = [1 + i for i in range(n_rollouts)]
    model_q = []
    for traj in plot_trajs:
        batch = build_batch(cfg['data_dir'], traj, 0, n_beads)
        for s in seeds:
            q = _model_q_series(mpath, batch, native_dm, rollout_steps, s, rollout_tail)
            model_q.append(q)
            logging.info(f'  model {traj} seed {s}: Q mean {q.mean():.3f} (n={len(q)})')
    model_q_all = np.concatenate(model_q)

    md_fold_pct = (md_q_all >= 0.7).mean() * 100
    model_fold_pct = (model_q_all >= 0.7).mean() * 100
    logging.info(f'  MD %Q>=0.7 = {md_fold_pct:.1f}% ; Model %Q>=0.7 = {model_fold_pct:.1f}%')

    from utils.plot_style import apply_thesis_style, MD_COLOR
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1.0, 26)
    # weight each frame so the bars are *fraction of frames per bin*; this makes
    # the bars within one distribution sum to 1.0 and read as percentages.
    md_w = np.full_like(md_q_all, 1.0 / len(md_q_all))
    model_w = np.full_like(model_q_all, 1.0 / len(model_q_all))
    ax.hist(md_q_all, bins=bins, weights=md_w, color=MD_COLOR, alpha=0.55,
            label=f'MD reference (n={len(md_q_all)})')
    ax.hist(model_q_all, bins=bins, weights=model_w, color='#ff7f0e', alpha=0.55,
            label=f'Model (n={len(model_q_all)})', edgecolor='#ff7f0e', linewidth=0.4)
    ax.axvline(0.7, color='black', ls=':', lw=1.2, alpha=0.6,
               label='Folded threshold ($Q=0.7$)')
    ax.set_xlabel('Fraction of native contacts $Q$')
    ax.set_ylabel('Fraction of frames')
    ax.set_title('Distribution of $Q$: model vs. MD reference')
    ax.set_xlim(0, 1.02)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f'q_distribution_{protein_name}.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  Plot saved to {out_path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(
        description='Trajectory quality (bonds + collisions) and Q distribution')
    parser.add_argument('--protein', type=str, required=True,
                        choices=list(PROTEIN_CONFIGS.keys()) + ['all'])
    parser.add_argument('--model_path', type=str, default=None,
                        help='Override the K=1 quality model_path')
    parser.add_argument('--multistep_path', type=str, default=None,
                        help='Override the K=5 multistep model used for Q distribution')
    parser.add_argument('--rollout_length', type=int, default=2000)
    parser.add_argument('--max_vel', type=float, default=3.0)
    parser.add_argument('--skip_quality', action='store_true',
                        help='Skip the bond/collision figure')
    parser.add_argument('--skip_q', action='store_true',
                        help='Skip the Q-distribution figure')
    parser.add_argument('--output_dir', type=str,
                        default='/home/ubuntu/code/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations/docs/plots')
    args = parser.parse_args()

    proteins = list(PROTEIN_CONFIGS.keys()) if args.protein == 'all' else [args.protein]
    os.makedirs(args.output_dir, exist_ok=True)
    for p in proteins:
        if args.model_path is not None:
            PROTEIN_CONFIGS[p]['model_path'] = args.model_path
        if not args.skip_quality:
            if os.path.exists(PROTEIN_CONFIGS[p]['model_path']):
                evaluate_protein(p, args.output_dir, args.rollout_length, args.max_vel)
            else:
                logging.warning(f'Quality model not found for {p}: '
                                f'{PROTEIN_CONFIGS[p]["model_path"]}')
        if not args.skip_q:
            evaluate_q_distribution(p, args.output_dir,
                                    model_path=args.multistep_path)
