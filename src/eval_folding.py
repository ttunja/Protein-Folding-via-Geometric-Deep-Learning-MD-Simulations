"""Folding evaluation: RMSD to native + fraction of native contacts (Q).

Runs stochastic rollouts from unfolded starting conformations and evaluates
whether the model drives the protein toward the native folded state.

Uses literature-grounded RMSD thresholds for each protein:
  Chignolin:  folded ≤ 2.0 Å,  borderline ≤ 3.0 Å
  Trp-cage:   folded ≤ 3.0 Å,  borderline ≤ 4.5 Å
  BBA:        folded ≤ 3.5 Å,  borderline ≤ 5.0 Å
  Villin:     folded ≤ 4.0 Å,  borderline ≤ 6.0 Å

References:
  - Lindorff-Larsen et al. (2011), Science: fast-folding protein benchmarks
  - nRMSD ≈ 0.6-0.7 Å/√N_res for 1-bead CG models

Usage:
  python eval_folding.py --protein chignolin --model_path /path/to/model_best
  python eval_folding.py --protein all --model_dir /path/to/models
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

from utils.utils import compute_weighted_rgs, compute_com, compute_distance_matrix
from preprocessing.preprocessing_utils import load_data
from train import seed_everything
import prody


# literature-grounded thresholds per protein
FOLDING_THRESHOLDS = {
    'chignolin': {'folded': 2.0, 'borderline': 3.0, 'n_beads': 10},
    'trp_cage':  {'folded': 3.0, 'borderline': 4.5, 'n_beads': 20},
    'bba':       {'folded': 3.5, 'borderline': 5.0, 'n_beads': 28},
    'villin':    {'folded': 4.0, 'borderline': 6.0, 'n_beads': 35},
    # unseen proteins for zero-shot generalization tests
    # thresholds: ~0.6-0.7 Å/√N_res for 1-bead CG, rounded up
    'ntl9':      {'folded': 4.5, 'borderline': 6.5, 'n_beads': 39},
    'ww':        {'folded': 4.0, 'borderline': 6.0, 'n_beads': 35},
    'protein_b': {'folded': 5.0, 'borderline': 7.0, 'n_beads': 47},
}

PROTEIN_CONFIGS = {
    'chignolin': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Chignolin/protein_train',
        'native_pdb': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Chignolin/folded/2rvd.pdb',
        'test_trajs': [f'traj_{i}' for i in range(38, 48)],
    },
    'trp_cage': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Trp-cage/protein_train',
        'native_pdb': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Trp-cage/topology.pdb',
        'test_trajs': [f'traj_{i}' for i in range(83, 98)],
    },
    'bba': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/BBA/protein_train',
        'native_pdb': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/BBA/topology.pdb',
        'test_trajs': [f'traj_{i}' for i in range(89, 105)],
    },
    'villin': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Villin/protein_train',
        'native_pdb': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Villin/topology.pdb',
        'test_trajs': [f'traj_{i}' for i in range(44, 58)],
    },
    # unseen proteins - use all trajectories as "test" since we never train on them
    'ntl9': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/NTL9/protein_train',
        'native_pdb': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/NTL9/topology.pdb',
        'test_trajs': None,  # auto-discover
    },
    'ww': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/WW/protein_train',
        'native_pdb': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/WW/topology.pdb',
        'test_trajs': None,
    },
    'protein_b': {
        'data_dir': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Protein_B/protein_train',
        'native_pdb': '/home/ubuntu/data/PROTEIN-FOLDING-DATA/Protein_B/topology.pdb',
        'test_trajs': None,
    },
}


def compute_rmsd(pos: torch.Tensor, ref: torch.Tensor) -> float:
    """RMSD after optimal Kabsch superposition."""
    pos, ref = pos.float(), ref.float()
    pos_c = pos - pos.mean(0, keepdim=True)
    ref_c = ref - ref.mean(0, keepdim=True)
    H = pos_c.T @ ref_c
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(Vt.T @ U.T)
    R = Vt.T @ torch.diag(torch.tensor([1., 1., torch.sign(d)])) @ U.T
    return torch.sqrt(((R @ pos_c.T).T - ref_c).pow(2).sum(-1).mean()).item()


def compute_native_contacts(pos: torch.Tensor, native_dm: torch.Tensor,
                            cutoff: float = 8.0, margin: float = 2.0) -> float:
    """Fraction of native contacts (Q).

    A native contact: beads i,j with |i-j|>3 and d_native(i,j) < cutoff.
    Contact is formed if d(i,j) < cutoff + margin in the given frame.
    """
    n = pos.shape[0]
    native_contacts = []
    for i in range(n):
        for j in range(i + 4, n):
            if native_dm[i, j] < cutoff:
                native_contacts.append((i, j))
    if len(native_contacts) == 0:
        return 1.0
    dm = compute_distance_matrix(pos)
    formed = sum(1 for i, j in native_contacts if dm[i, j] < cutoff + margin)
    return formed / len(native_contacts)


def get_native_cg(pdb_path: str):
    """Load PDB, coarse-grain to residue level, return CG positions and distance matrix."""
    topo = prody.parsePDB(pdb_path)
    native_pos = torch.tensor(topo.getCoords(), dtype=torch.float32).unsqueeze(1)
    native_cluster = torch.tensor(topo.getResnums() - topo.getResnums().min(), dtype=torch.long)
    native_cg, _ = compute_com(native_pos, torch.ones(native_pos.shape[0]), native_cluster)
    native_cg = native_cg[:, 0, :]
    native_dm = compute_distance_matrix(native_cg)
    n_beads = native_cg.shape[0]
    return native_cg, native_dm, n_beads


def find_unfolded_start(data_dir: str, test_trajs: list, n_beads: int, top_k: int = 5):
    """Find the most unfolded 30-frame windows in the test trajectories."""
    if test_trajs is None:
        # auto-discover all trajectories in the data dir
        test_trajs = sorted([d for d in os.listdir(data_dir)
                            if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('traj_')])
    candidates = []
    for traj_name in test_trajs:
        tp = os.path.join(data_dir, traj_name)
        if not os.path.exists(f'{tp}/position.h5'):
            continue
        position = np.array(load_data(['position'], f'{tp}/position.h5')[0], dtype=np.float32)
        residue_num = load_data(['residue_num'], f'{tp}/residue_num.h5')[0]
        cluster = torch.tensor(residue_num - min(residue_num), dtype=torch.long)
        cg_pos, _ = compute_com(torch.tensor(position).permute(1, 0, 2),
                                torch.ones(position.shape[1]), cluster)
        # compute Rg for all frames
        rgs = np.array([
            compute_weighted_rgs(cg_pos[:, t], torch.ones(n_beads), torch.tensor([n_beads])).item()
            for t in range(min(1000, cg_pos.shape[1]))  # sample first 1000 for speed
        ])
        # find 30-frame window with highest avg Rg
        for s in range(0, len(rgs) - 30):
            avg_rg = rgs[s:s + 30].mean()
            candidates.append((traj_name, s, avg_rg))

    candidates.sort(key=lambda x: -x[2])
    return candidates[:top_k]


def build_batch(data_dir: str, traj_name: str, start_frame: int, n_beads: int):
    """Build a model input batch from a specific trajectory and starting frame."""
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

    return {
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


def run_folding_test(protein_name: str, model_path: str, n_rollouts: int = 10,
                     rollout_length: int = 2000, max_vel: float = 3.0,
                     output_dir: str = None):
    """Run the full folding evaluation for one protein."""
    cfg = PROTEIN_CONFIGS[protein_name]
    thresholds = FOLDING_THRESHOLDS[protein_name]
    thresh_fold = thresholds['folded']
    thresh_border = thresholds['borderline']

    logging.info(f'Evaluating {protein_name} (folded ≤ {thresh_fold}Å, borderline ≤ {thresh_border}Å)')

    # native structure
    native_cg, native_dm, n_beads = get_native_cg(cfg['native_pdb'])
    n_native_contacts = sum(1 for i in range(n_beads) for j in range(i+4, n_beads)
                           if native_dm[i, j] < 8.0)
    logging.info(f'  Native: {n_beads} beads, {n_native_contacts} contacts')

    # find unfolded starting point
    starts = find_unfolded_start(cfg['data_dir'], cfg['test_trajs'], n_beads)
    traj_name, start_frame, start_rg = starts[0]
    logging.info(f'  Starting from {traj_name}@{start_frame}, Rg≈{start_rg:.1f}Å')

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # build batch
    batch = build_batch(cfg['data_dir'], traj_name, start_frame, n_beads)
    batch = {k: v.to(device) for k, v in batch.items()}

    # run rollouts
    results = []
    for r in range(n_rollouts):
        seed_everything(r * 42 + 1)
        b = {k: v.clone() for k, v in batch.items()}
        with torch.no_grad():
            out = model.simulate(b, rollout_length, save_positions=True, deter=False,
                                save_frequency=1, disable_bar=True, max_vel=max_vel)
        pred = out['rollout_u_pos'][:n_beads].cpu()

        rmsds, rg_vals, q_vals = [], [], []
        for t in range(pred.shape[1]):
            rmsds.append(compute_rmsd(pred[:, t], native_cg))
            rg_vals.append(compute_weighted_rgs(
                pred[:, t], torch.ones(n_beads), torch.tensor([n_beads])).item())
            if t % 10 == 0:  # Q is expensive, sample every 10 frames
                q_vals.append((t, compute_native_contacts(pred[:, t], native_dm)))

        min_rmsd = min(rmsds)
        min_frame = np.argmin(rmsds)
        q_at_best = compute_native_contacts(pred[:, min_frame], native_dm)
        q_final = compute_native_contacts(pred[:, -1], native_dm)

        folded = min_rmsd <= thresh_fold and q_at_best >= 0.7
        borderline = not folded and min_rmsd <= thresh_border
        status = 'FOLDED' if folded else ('BORDERLINE' if borderline else 'not folded')

        results.append({
            'rollout': r, 'min_rmsd': min_rmsd, 'min_frame': min_frame,
            'final_rmsd': rmsds[-1], 'q_at_best': q_at_best, 'q_final': q_final,
            'min_rg': min(rg_vals), 'final_rg': rg_vals[-1], 'status': status,
            'rmsds': rmsds, 'rg_vals': rg_vals, 'q_vals': q_vals,
        })
        logging.info(f'  R{r}: RMSD min={min_rmsd:.2f}Å @{min_frame}, Q={q_at_best:.2f}, '
                     f'final RMSD={rmsds[-1]:.2f}, Q={q_final:.2f} → {status}')

    # summary
    n_folded = sum(1 for r in results if r['status'] == 'FOLDED')
    n_border = sum(1 for r in results if r['status'] == 'BORDERLINE')
    min_rmsds = [r['min_rmsd'] for r in results]

    logging.info(f'\n  === {protein_name.upper()} SUMMARY ===')
    logging.info(f'  Folded: {n_folded}/{n_rollouts}, Borderline: {n_border}/{n_rollouts}')
    logging.info(f'  Min RMSD: best={min(min_rmsds):.2f}, mean={np.mean(min_rmsds):.2f}Å')
    logging.info(f'  Mean Q at best frame: {np.mean([r["q_at_best"] for r in results]):.2f}')

    # plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        from utils.plot_style import (apply_thesis_style, smooth, distinct_colors,
                                      BAND_COLOR, MEDIAN_COLOR, HIGHLIGHT_COLOR)
        apply_thesis_style()

        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
        # x-axis in microseconds (200 ps per frame)
        x_us = np.arange(rollout_length) * 200e-6

        # Use lighter smoothing so transient folding dips below the threshold are still visible.
        SMOOTH_FRAC = 0.012  # ~120 frames for a 10000-step rollout

        # Ten overlaid traces are unreadable, so panels (a) to (c) show the whole
        # ensemble as a min-max band with its median, and pick out one
        # representative rollout: the one whose min RMSD is closest to the median.
        med_min_rmsd = float(np.median([r['min_rmsd'] for r in results]))
        rep = int(np.argmin([abs(r['min_rmsd'] - med_min_rmsd) for r in results]))
        rep_label = f'rollout {rep+1} (representative)'

        def band_panel(ax, series_list, x, ylabel, title, rep_idx=rep,
                       band_label='all rollouts (range)'):
            """Draw min-max band, median, and one highlighted trace."""
            sm = np.vstack([smooth(s, frac=SMOOTH_FRAC) for s in series_list])
            ax.fill_between(x, sm.min(axis=0), sm.max(axis=0),
                            color=BAND_COLOR, alpha=0.30, lw=0, zorder=1,
                            label=band_label)
            ax.plot(x, np.median(sm, axis=0), color=MEDIAN_COLOR, lw=1.8,
                    zorder=3, label='median rollout')
            ax.plot(x, series_list[rep_idx], color=HIGHLIGHT_COLOR, alpha=0.16,
                    lw=0.7, zorder=4)
            ax.plot(x, sm[rep_idx], color=HIGHLIGHT_COLOR, lw=2.0, zorder=5,
                    label=rep_label)
            ax.set_xlabel('Time (μs)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            return sm

        ax = axes[0, 0]
        band_panel(ax, [r['rmsds'] for r in results], x_us,
                   'RMSD to native (Å)',
                   f'(a) RMSD trajectory  ({n_folded} folded, {n_border} borderline)')
        # mark the best (lowest-RMSD) frame of the representative rollout
        best_idx = int(np.argmin(results[rep]['rmsds']))
        ax.scatter(x_us[best_idx], results[rep]['rmsds'][best_idx],
                   s=60, color=HIGHLIGHT_COLOR, edgecolor='black', linewidth=0.9,
                   zorder=10, label='best frame of that rollout')
        ax.axhline(thresh_fold, color='#2a9d2a', ls='--', lw=1.6,
                   label=f'folded ($\\leq${thresh_fold} Å)')
        ax.axhline(thresh_border, color='#c4a000', ls=':', lw=1.6,
                   label=f'borderline ($\\leq${thresh_border} Å)')
        ax.legend(loc='upper right', ncol=2, fontsize=8)

        band_panel(axes[0, 1], [r['rg_vals'] for r in results], x_us,
                   '$R_g$ (Å)', '(b) Radius of gyration')
        axes[0, 1].legend(loc='upper right', fontsize=8)

        ax = axes[1, 0]
        q_frames = np.array([f for f, _ in results[0]['q_vals']])
        band_panel(ax, [np.array([q for _, q in r['q_vals']]) for r in results],
                   q_frames * 200e-6, '$Q$ (native contacts)',
                   '(c) Fraction of native contacts')
        ax.axhline(0.7, color='#2a9d2a', ls='--', lw=1.6,
                   label='$Q\\geq 0.7$')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(-0.02, 1.05)

        # Panel (d) has one point per rollout, so distinct colours are readable
        # here. distinct_colors never reuses a colour, unlike indexing the
        # 8-entry palette modulo its length.
        ax = axes[1, 1]
        colors = distinct_colors(len(results))
        for i, r in enumerate(results):
            ax.scatter([r['min_rmsd']], [r['q_at_best']],
                       s=140, color=colors[i],
                       edgecolor='black', linewidth=1.0, zorder=5,
                       label=f'rollout {i+1}')
        # Shade the folded region, meaning RMSD <= threshold and Q >= 0.7 together.
        # fill_between takes data coordinates on both axes. axvspan would not: its
        # ymin/ymax are fractions of the axes height, so passing Q values there
        # placed the band at the wrong height.
        ax.fill_between([0, thresh_fold], 0.7, 1.05,
                        color='#2a9d2a', alpha=0.10, lw=0, zorder=0)
        ax.axvline(thresh_fold, color='#2a9d2a', ls='--', lw=1.6,
                   label=f'folded RMSD ($\\leq${thresh_fold} Å)')
        ax.axhline(0.7, color='#2a9d2a', ls=':', lw=1.6,
                   label='$Q\\geq 0.7$')
        ax.set_xlabel('Minimum RMSD (Å)')
        ax.set_ylabel('$Q$ at best frame')
        # plain "vs." here: this is a matplotlib title, not LaTeX, so the
        # thin-space escape used in the thesis text would be printed literally
        ax.set_title('(d) RMSD vs. $Q$ at best frame')
        ax.legend(loc='lower left', ncol=1, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=max(0, min(r['min_rmsd'] for r in results) - 0.3),
                    right=max(r['min_rmsd'] for r in results) + 0.6)
        ax.set_ylim(min(0.6, min(r['q_at_best'] for r in results) - 0.05), 1.05)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'folding_{protein_name}.png'),
                    bbox_inches='tight')
        plt.close(fig)
        logging.info(f'  Plot saved to {output_dir}/folding_{protein_name}.png')

    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Folding evaluation with RMSD + Q')
    parser.add_argument('--protein', type=str, required=True,
                        choices=list(PROTEIN_CONFIGS.keys()) + ['all'])
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model_best file')
    parser.add_argument('--model_dir', type=str, default='/home/ubuntu/models/PROTEIN-FOLDING-MODELS',
                        help='Base model directory (used with --protein all)')
    parser.add_argument('--n_rollouts', type=int, default=10)
    parser.add_argument('--rollout_length', type=int, default=2000)
    parser.add_argument('--max_vel', type=float, default=3.0)
    parser.add_argument('--output_dir', type=str,
                        default='/home/ubuntu/code/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations/docs/plots')
    args = parser.parse_args()

    if args.protein == 'all':
        model_map = {
            'chignolin': f'{args.model_dir}/Chignolin/chignolin_gru_only/model_best',
            'trp_cage': f'{args.model_dir}/Trp-cage/trp_cage_gru_only/model_best',
            'bba': f'{args.model_dir}/BBA/bba_gru_only/model_best',
            'villin': f'{args.model_dir}/Villin/villin_gru_only/model_best',
        }
        for pname, mpath in model_map.items():
            if os.path.exists(mpath):
                run_folding_test(pname, mpath, args.n_rollouts, args.rollout_length,
                                args.max_vel, args.output_dir)
            else:
                logging.warning(f'Model not found for {pname}: {mpath}')
    else:
        model_path = args.model_path
        if model_path is None:
            name_map = {'chignolin': 'Chignolin/chignolin_gru_only',
                       'trp_cage': 'Trp-cage/trp_cage_gru_only',
                       'bba': 'BBA/bba_gru_only', 'villin': 'Villin/villin_gru_only'}
            model_path = f'{args.model_dir}/{name_map[args.protein]}/model_best'
        run_folding_test(args.protein, model_path, args.n_rollouts, args.rollout_length,
                        args.max_vel, args.output_dir)
