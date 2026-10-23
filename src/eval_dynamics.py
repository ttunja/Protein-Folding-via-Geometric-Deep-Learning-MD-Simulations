"""Long-timescale dynamics analysis: folding events, transition counts, state populations.

Runs extended rollouts and analyzes folding/unfolding kinetics by tracking
RMSD and native contact fraction (Q) over time.

Reference timescales from Lindorff-Larsen et al. (2011), Science 334, 517:
  Chignolin:  0.6 μs folding time  →  3,000 frames at 200ps/frame
  Trp-cage:   14 μs               → 70,000 frames
  BBA:        18 μs               → 90,000 frames
  Villin:     2.8 μs              → 14,000 frames

Usage:
  python eval_dynamics.py --protein chignolin --rollout_length 10000
  python eval_dynamics.py --protein villin --rollout_length 20000
"""

import argparse
import os
import logging
import numpy as np
import torch

from utils.utils import compute_weighted_rgs, compute_com, compute_distance_matrix
from utils.utils import compute_ramachandran_angles
from preprocessing.preprocessing_utils import load_data
from train import seed_everything
import eval_folding
from eval_folding import native_reference, check_reference, load_traj, cg_traj


# reference folding times from Lindorff-Larsen et al. 2011
REFERENCE = {
    'chignolin': {'fold_time_us': 0.6, 'ca_rmsd_folded': 1.0, 'pdb': 'CLN025',
                  'rmsd_threshold': 2.0, 'q_threshold': 0.7},
    'trp_cage':  {'fold_time_us': 14.0, 'ca_rmsd_folded': 1.4, 'pdb': '2JOF',
                  'rmsd_threshold': 3.0, 'q_threshold': 0.7},
    'bba':       {'fold_time_us': 18.0, 'ca_rmsd_folded': 1.6, 'pdb': '1FME',
                  'rmsd_threshold': 3.5, 'q_threshold': 0.7},
    'villin':    {'fold_time_us': 2.8, 'ca_rmsd_folded': 1.3, 'pdb': '2F4K',
                  'rmsd_threshold': 4.0, 'q_threshold': 0.7},
    'ntl9':      {'fold_time_us': 1.5, 'ca_rmsd_folded': 1.5, 'pdb': '2HBA',
                  'rmsd_threshold': 4.5, 'q_threshold': 0.7},
    'ww':        {'fold_time_us': 21.0, 'ca_rmsd_folded': 1.3, 'pdb': '2F21',
                  'rmsd_threshold': 4.0, 'q_threshold': 0.7},
    'protein_b': {'fold_time_us': 3.5, 'ca_rmsd_folded': 1.6, 'pdb': '1PRB',
                  'rmsd_threshold': 5.0, 'q_threshold': 0.7},
}

# native structures and centering come from eval_folding.PROTEIN_CONFIGS
PROTEIN_CONFIGS = {
    'chignolin': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Chignolin/protein_train',
        'model_path': '/home/ubuntu/models/PROTEIN-FOLDING-MODELS/Chignolin/chignolin_gru_only/model_best',
        'test_trajs': [f'traj_{i}' for i in range(38, 48)],
        'n_beads': 10,
    },
    'trp_cage': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Trp-cage/protein_train',
        'model_path': '/home/ubuntu/models/PROTEIN-FOLDING-MODELS/Trp-cage/trp_cage_gru_only/model_best',
        'test_trajs': [f'traj_{i}' for i in range(83, 98)],
        'n_beads': 20,
    },
    'bba': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/BBA/protein_train',
        'model_path': '/home/ubuntu/models/PROTEIN-FOLDING-MODELS/BBA/bba_gru_only/model_best',
        'test_trajs': [f'traj_{i}' for i in range(89, 105)],
        'n_beads': 28,
    },
    'villin': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Villin/protein_train',
        'model_path': '/home/ubuntu/models/PROTEIN-FOLDING-MODELS/Villin/villin_gru_only/model_best',
        'test_trajs': [f'traj_{i}' for i in range(44, 58)],
        'n_beads': 35,
    },
    # unseen proteins - auto-discover trajectories
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

FRAME_DT_PS = 200  # each frame = 200 ps


def compute_rmsd(pos: torch.Tensor, ref: torch.Tensor) -> float:
    pos, ref = pos.float(), ref.float()
    pos_c = pos - pos.mean(0, keepdim=True)
    ref_c = ref - ref.mean(0, keepdim=True)
    H = pos_c.T @ ref_c
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    R = Vt.T @ torch.diag(torch.tensor([1., 1., torch.sign(d)])) @ U.T
    return torch.sqrt(((R @ pos_c.T).T - ref_c).pow(2).sum(-1).mean()).item()


def compute_Q(pos: torch.Tensor, native_dm: torch.Tensor,
              cutoff: float = 8.0, margin: float = 2.0) -> float:
    """Fraction of native contacts."""
    n = pos.shape[0]
    native_contacts = []
    for i in range(n):
        for j in range(i + 4, n):
            if native_dm[i, j] < cutoff:
                native_contacts.append((i, j))
    if not native_contacts:
        return 1.0
    dm = compute_distance_matrix(pos)
    formed = sum(1 for i, j in native_contacts if dm[i, j] < cutoff + margin)
    return formed / len(native_contacts)


def _center(protein_name):
    return eval_folding.PROTEIN_CONFIGS[protein_name].get('center', False)


def _test_trajs(cfg):
    return eval_folding.list_trajs(cfg['data_dir'], cfg['test_trajs'])


def find_unfolded_start(protein_name, n_beads):
    """Find the most unfolded 30-frame window."""
    cfg = PROTEIN_CONFIGS[protein_name]
    test_trajs = _test_trajs(cfg)
    best_rg, best_traj, best_start = 0, test_trajs[0], 0
    for traj_name in test_trajs[:10]:
        tp = os.path.join(cfg['data_dir'], traj_name)
        if not os.path.exists(f'{tp}/position.h5'):
            continue
        cg_pos = cg_traj(load_traj(tp, np.s_[:1000], _center(protein_name)))
        rgs = np.array([compute_weighted_rgs(cg_pos[:, t], torch.ones(n_beads),
                        torch.tensor([n_beads])).item() for t in range(cg_pos.shape[1])])
        for s in range(len(rgs) - 30):
            avg = rgs[s:s + 30].mean()
            if avg > best_rg:
                best_rg, best_traj, best_start = avg, traj_name, s
    return best_traj, best_start, best_rg


def count_transitions(states, min_duration=5):
    """Count folding/unfolding transitions.
    states: array of 'F' (folded) or 'U' (unfolded) per frame.
    min_duration: ignore state visits shorter than this (noise filter)."""
    # smooth: require min_duration consecutive frames to count as state change
    smoothed = []
    current_state = states[0]
    count = 1
    for s in states[1:]:
        if s == current_state:
            count += 1
        else:
            if count >= min_duration:
                smoothed.extend([current_state] * count)
            else:
                smoothed.extend([current_state] * count)  # keep but mark as transient
            current_state = s
            count = 1
    smoothed.extend([current_state] * count)

    # count transitions on smoothed signal
    n_fold, n_unfold = 0, 0
    in_state = smoothed[0]
    state_start = 0
    transitions = []
    for i in range(1, len(smoothed)):
        if smoothed[i] != in_state:
            duration = i - state_start
            if duration >= min_duration:
                if in_state == 'U' and smoothed[i] == 'F':
                    n_fold += 1
                    transitions.append(('fold', state_start, i))
                elif in_state == 'F' and smoothed[i] == 'U':
                    n_unfold += 1
                    transitions.append(('unfold', state_start, i))
            in_state = smoothed[i]
            state_start = i

    return n_fold, n_unfold, transitions


def get_md_reference(protein_name, bead_idx, native_cg, native_dm, n_frames=5000):
    """Compute RMSD and Q traces from reference MD trajectories."""
    cfg = PROTEIN_CONFIGS[protein_name]
    n_scored = len(bead_idx)
    md_rmsds, md_qs, md_rgs = [], [], []
    for traj_name in _test_trajs(cfg)[:3]:
        tp = os.path.join(cfg['data_dir'], traj_name)
        if not os.path.exists(f'{tp}/position.h5'):
            continue
        cg_pos = cg_traj(load_traj(tp, np.s_[:n_frames], _center(protein_name)))[bead_idx]
        for t in range(cg_pos.shape[1]):
            md_rmsds.append(compute_rmsd(cg_pos[:, t], native_cg))
            md_rgs.append(compute_weighted_rgs(cg_pos[:, t], torch.ones(n_scored),
                          torch.tensor([n_scored])).item())
            if t % 10 == 0:
                md_qs.append(compute_Q(cg_pos[:, t], native_dm))
    return md_rmsds, md_qs, md_rgs


def run_dynamics_analysis(protein_name: str, rollout_length: int = 10000,
                          n_rollouts: int = 3, max_vel: float = 3.0):
    """Full long-timescale dynamics analysis."""
    cfg = PROTEIN_CONFIGS[protein_name]
    ref = REFERENCE[protein_name]
    n_beads = cfg['n_beads']
    rmsd_thresh = ref['rmsd_threshold']
    q_thresh = ref['q_threshold']

    sim_time_us = rollout_length * FRAME_DT_PS / 1e6
    fold_time_us = ref['fold_time_us']

    logging.info(f'\n{"="*70}')
    logging.info(f'  {protein_name} - long-timescale dynamics analysis')
    logging.info(f'  {rollout_length} steps = {sim_time_us:.1f} μs '
                 f'(ref folding time: {fold_time_us} μs)')
    logging.info(f'  Folded: RMSD ≤ {rmsd_thresh} Å AND Q ≥ {q_thresh}')
    logging.info(f'{"="*70}')

    native_cg, native_dm, n_beads, bead_idx = native_reference(protein_name)
    n_scored = len(bead_idx)
    check_reference(protein_name)

    # find unfolded start
    traj_name, start_frame, start_rg = find_unfolded_start(protein_name, n_beads)
    logging.info(f'  Starting from {traj_name}@{start_frame}, Rg≈{start_rg:.1f} Å')

    # load model
    model = torch.load(cfg['model_path'], map_location='cpu', weights_only=False)
    model.eval()
    batch = eval_folding.build_batch(cfg['data_dir'], traj_name, start_frame, n_beads,
                                     _center(protein_name))

    # get MD reference
    logging.info('  Computing MD reference...')
    md_rmsds, md_qs, md_rgs = get_md_reference(
        protein_name, bead_idx, native_cg, native_dm, n_frames=min(rollout_length, 5000))

    md_states = ['F' if r <= rmsd_thresh else 'U' for r in md_rmsds]
    md_folded_frac = sum(1 for s in md_states if s == 'F') / len(md_states)
    md_folds, md_unfolds, _ = count_transitions(md_states)
    logging.info(f'  MD reference: {md_folded_frac*100:.0f}% folded, '
                 f'{md_folds} folding + {md_unfolds} unfolding events')

    # run rollouts
    all_results = []
    for r in range(n_rollouts):
        seed_everything(r * 42 + 1)
        b = {k: v.clone() for k, v in batch.items()}
        logging.info(f'\n  Rollout {r} ({rollout_length} steps)...')
        with torch.no_grad():
            out = model.simulate(b, rollout_length, save_positions=True, deter=False,
                                save_frequency=1, disable_bar=True, max_vel=max_vel)
        pred = out['rollout_u_pos'][:n_beads][bead_idx]

        rmsds, rg_vals, q_vals = [], [], []
        for t in range(pred.shape[1]):
            rmsds.append(compute_rmsd(pred[:, t], native_cg))
            rg_vals.append(compute_weighted_rgs(
                pred[:, t], torch.ones(n_scored), torch.tensor([n_scored])).item())
            if t % 5 == 0:
                q_vals.append(compute_Q(pred[:, t], native_dm))
            else:
                q_vals.append(q_vals[-1] if q_vals else 0.5)

        # classify frames
        states = []
        for t in range(len(rmsds)):
            if rmsds[t] <= rmsd_thresh and q_vals[t] >= q_thresh:
                states.append('F')
            else:
                states.append('U')

        folded_frac = sum(1 for s in states if s == 'F') / len(states)
        n_fold, n_unfold, transitions = count_transitions(states)

        # first folding time
        first_fold_frame = None
        for i, s in enumerate(states):
            if s == 'F':
                first_fold_frame = i
                break
        first_fold_us = first_fold_frame * FRAME_DT_PS / 1e6 if first_fold_frame else None

        result = {
            'rmsds': rmsds, 'rg_vals': rg_vals, 'q_vals': q_vals,
            'states': states, 'folded_frac': folded_frac,
            'n_fold': n_fold, 'n_unfold': n_unfold,
            'transitions': transitions, 'first_fold_us': first_fold_us,
            'min_rmsd': min(rmsds),
        }
        all_results.append(result)

        ff_str = f'{first_fold_us:.2f} μs' if first_fold_us else 'never'
        logging.info(f'    folded {folded_frac*100:.0f}% | {n_fold} fold + {n_unfold} unfold events | '
                     f'first fold: {ff_str} | min RMSD: {min(rmsds):.2f} Å')

    # Ramachandran from the generated trajectory (use first rollout)
    pred_first = out['rollout_u_pos'][:n_beads]
    pred_phi, pred_psi = compute_ramachandran_angles(np.array(pred_first))

    # MD Ramachandran from test trajectories
    md_phi_all, md_psi_all = [], []
    for traj_name in _test_trajs(cfg)[:3]:
        tp = os.path.join(cfg['data_dir'], traj_name)
        if not os.path.exists(f'{tp}/position.h5'):
            continue
        cg_pos = cg_traj(load_traj(tp, np.s_[:2000], _center(protein_name)))
        phi, psi = compute_ramachandran_angles(np.array(cg_pos))
        md_phi_all.extend(phi)
        md_psi_all.extend(psi)

    # summary
    logging.info(f'\n  {"="*50}')
    logging.info(f'  SUMMARY: {protein_name}')
    logging.info(f'  {"="*50}')
    logging.info(f'  Simulation: {rollout_length} frames = {sim_time_us:.1f} μs')
    logging.info(f'  Ref folding time: {fold_time_us} μs')
    for i, res in enumerate(all_results):
        ff_str = f'{res["first_fold_us"]:.2f} μs' if res['first_fold_us'] else 'never'
        logging.info(f'  R{i}: {res["folded_frac"]*100:.0f}% folded, '
                     f'{res["n_fold"]}F/{res["n_unfold"]}U transitions, '
                     f'first fold: {ff_str}, min RMSD: {res["min_rmsd"]:.2f} Å')
    logging.info(f'  MD ref: {md_folded_frac*100:.0f}% folded, '
                 f'{md_folds}F/{md_unfolds}U transitions')

    return all_results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Long-timescale dynamics analysis')
    parser.add_argument('--protein', type=str, required=True,
                        choices=list(PROTEIN_CONFIGS.keys()) + ['all'])
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model_best file (overrides per-protein default)')
    parser.add_argument('--rollout_length', type=int, default=10000,
                        help='Number of rollout steps (each = 200ps)')
    parser.add_argument('--n_rollouts', type=int, default=3)
    parser.add_argument('--max_vel', type=float, default=3.0)
    args = parser.parse_args()

    proteins = list(PROTEIN_CONFIGS.keys()) if args.protein == 'all' else [args.protein]
    for p in proteins:
        # override model_path if provided via CLI
        if args.model_path is not None:
            PROTEIN_CONFIGS[p]['model_path'] = args.model_path
        if os.path.exists(PROTEIN_CONFIGS[p]['model_path']):
            run_dynamics_analysis(p, args.rollout_length, args.n_rollouts,
                                args.max_vel)
        else:
            logging.warning(f'Model not found for {p}: {PROTEIN_CONFIGS[p]["model_path"]}')
