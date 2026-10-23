"""Trajectory quality evaluation: bond lengths, collisions, and the
fraction-of-native-contacts (Q) distribution against reference MD.

Two evaluations per protein:
  - quality:        bond length and collision rate, single short rollout
                    from the K=1 gru_only model (Section 6.4 in the thesis).
  - Q distribution: three long rollouts from the K=5 multi-step model vs
                    MD reference (Section 6.4 in the thesis).

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

from utils.utils import (compute_weighted_rgs, compute_com, compute_distance_matrix,
                          compute_ramachandran_angles, compute_dihedral_angle)
from preprocessing.preprocessing_utils import load_data
from train import seed_everything
from eval_folding import (
    PROTEIN_CONFIGS as FOLDING_CONFIGS,
    build_batch,
    cg_traj,
    load_traj,
    native_reference,
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
    data = load_traj(tp, np.s_[start:end], _center(data_dir))
    return np.array(cg_traj(data))  # [n_beads, n_frames, 3]


def _center(data_dir: str) -> bool:
    """True if data_dir belongs to a protein centered at load time."""
    return any(c['data_dir'] == data_dir and c.get('center', False)
               for c in FOLDING_CONFIGS.values())


def get_model_trajectory(model_path: str, data_dir: str, traj_name: str,
                         start_frame: int, n_beads: int, rollout_length: int = 2000,
                         max_vel: float = 3.0, seed: int = 42):
    """Generate a trajectory from the model. Returns [n_beads, n_frames, 3]."""
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()

    batch = build_batch(data_dir, traj_name, start_frame, n_beads, _center(data_dir))

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


def evaluate_protein(protein_name: str, rollout_length: int = 2000,
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


def _md_q_series(data_dir, traj_name, native_dm, n_frames, bead_idx):
    """Q time series for an MD trajectory. n_frames=0 means use the whole trajectory."""
    tp = os.path.join(data_dir, traj_name)
    frames = np.s_[:n_frames + 30] if n_frames > 0 else None
    cg_pos = cg_traj(load_traj(tp, frames, _center(data_dir)))[bead_idx]
    start = 30
    qs = [compute_native_contacts(cg_pos[:, t], native_dm) for t in range(start, cg_pos.shape[1])]
    return np.array(qs)


def _model_q_series(model_path, batch, native_dm, n_steps, seed, tail, bead_idx):
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
    pred = pred[:n_beads][bead_idx]
    start = max(0, pred.shape[1] - tail) if tail > 0 else 0
    qs = [compute_native_contacts(pred[:, t], native_dm) for t in range(start, pred.shape[1])]
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return np.array(qs)


def evaluate_q_distribution(protein_name: str, n_rollouts: int = 1,
                             rollout_steps: int = 10000, rollout_tail: int = 0,
                             md_trajs: int = 3, md_frames: int = 0,
                             model_path: str = None):
    """Q distribution: K=5 multi-step model vs MD reference."""
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

    _, native_dm, _, bead_idx = native_reference(protein_name)

    test_trajs = cfg['test_trajs'] or sorted(
        d for d in os.listdir(cfg['data_dir'])
        if os.path.isdir(os.path.join(cfg['data_dir'], d)) and d.startswith('traj_'))

    md_q = []
    for traj in test_trajs[:md_trajs]:
        q = _md_q_series(cfg['data_dir'], traj, native_dm, md_frames, bead_idx)
        md_q.append(q)
        logging.info(f'  MD {traj}: Q mean {q.mean():.3f} (n={len(q)})')
    md_q_all = np.concatenate(md_q)

    # Start each model rollout from the same conformation that the corresponding
    # MD trajectory starts from. This makes the comparison apples-to-apples:
    # both distributions cover the same total simulated time from the same
    # initial configurations, so any divergence is the model's, not the
    # starting condition's.
    start_trajs = test_trajs[:md_trajs]
    seeds = [1 + i for i in range(n_rollouts)]
    model_q = []
    for traj in start_trajs:
        batch = build_batch(cfg['data_dir'], traj, 0, n_beads, _center(cfg['data_dir']))
        for s in seeds:
            q = _model_q_series(mpath, batch, native_dm, rollout_steps, s, rollout_tail,
                                bead_idx)
            model_q.append(q)
            logging.info(f'  model {traj} seed {s}: Q mean {q.mean():.3f} (n={len(q)})')
    model_q_all = np.concatenate(model_q)

    md_fold_pct = (md_q_all >= 0.7).mean() * 100
    model_fold_pct = (model_q_all >= 0.7).mean() * 100
    logging.info(f'  MD %Q>=0.7 = {md_fold_pct:.1f}% ; Model %Q>=0.7 = {model_fold_pct:.1f}%')


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
                        help='Skip the bond/collision evaluation')
    parser.add_argument('--skip_q', action='store_true',
                        help='Skip the Q-distribution evaluation')
    args = parser.parse_args()

    proteins = list(PROTEIN_CONFIGS.keys()) if args.protein == 'all' else [args.protein]
    for p in proteins:
        if args.model_path is not None:
            PROTEIN_CONFIGS[p]['model_path'] = args.model_path
        if not args.skip_quality:
            if os.path.exists(PROTEIN_CONFIGS[p]['model_path']):
                evaluate_protein(p, args.rollout_length, args.max_vel)
            else:
                logging.warning(f'Quality model not found for {p}: '
                                f'{PROTEIN_CONFIGS[p]["model_path"]}')
        if not args.skip_q:
            evaluate_q_distribution(p, model_path=args.multistep_path)
