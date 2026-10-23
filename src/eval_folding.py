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
import logging
import numpy as np
import h5py
import torch

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

# per-rollout series are saved here as .npz
# per-rollout series, read by summarize_folding_rows.py
CACHE_DIR = os.path.join(HERE, 'figures', 'folding_out')

DATA = '/home/ubuntu/data/PROTEIN-FOLDING-DATA'

# native_pdb is the experimental structure, topology_pdb only reproduces old scores
# native_map: (first simulated bead, first reference residue, number of residues)
# center: NTL9, WW and Protein B positions are centered at load time
PROTEIN_CONFIGS = {
    'chignolin': {
        'data_dir': f'{DATA}/Chignolin/protein_train',
        'native_pdb': f'{DATA}/Chignolin/folded/2rvd.pdb',
        'topology_pdb': f'{DATA}/Chignolin/folded/2rvd.pdb',
        'native_map': (0, 0, 10),
        'test_trajs': [f'traj_{i}' for i in range(38, 48)],
    },
    'trp_cage': {
        'data_dir': f'{DATA}/Trp-cage/protein_train',
        'native_pdb': f'{DATA}/Trp-cage/reference_2JOF.pdb',
        'topology_pdb': f'{DATA}/Trp-cage/topology.pdb',
        'native_map': (0, 0, 20),
        'test_trajs': [f'traj_{i}' for i in range(83, 98)],
    },
    'bba': {
        'data_dir': f'{DATA}/BBA/protein_train',
        'native_pdb': f'{DATA}/BBA/reference_1FME.pdb',
        'topology_pdb': f'{DATA}/BBA/topology.pdb',
        'native_map': (0, 0, 28),
        'test_trajs': [f'traj_{i}' for i in range(89, 105)],
    },
    'villin': {
        'data_dir': f'{DATA}/Villin/protein_train',
        'native_pdb': f'{DATA}/Villin/reference_2F4K.pdb',
        'topology_pdb': f'{DATA}/Villin/topology.pdb',
        'native_map': (0, 0, 35),
        'test_trajs': [f'traj_{i}' for i in range(44, 58)],
    },
    # unseen proteins - use all trajectories as "test" since we never train on them
    'ntl9': {
        'data_dir': f'{DATA}/NTL9/protein_train',
        'native_pdb': f'{DATA}/NTL9/native.pdb',
        'topology_pdb': f'{DATA}/NTL9/topology.pdb',
        'native_map': (0, 0, 39),
        'test_trajs': None,  # auto-discover
        'center': True,
    },
    'ww': {
        'data_dir': f'{DATA}/WW/protein_train',
        'native_pdb': f'{DATA}/WW/native.pdb',
        'topology_pdb': f'{DATA}/WW/topology.pdb',
        'native_map': (2, 5, 33),
        'test_trajs': None,
        'center': True,
    },
    'protein_b': {
        'data_dir': f'{DATA}/Protein_B/protein_train',
        'native_pdb': f'{DATA}/Protein_B/native.pdb',
        'topology_pdb': f'{DATA}/Protein_B/topology.pdb',
        'native_map': (0, 6, 47),
        'test_trajs': None,
        'center': True,
    },
}

# solvent and ions left in some trajectories, WW has six waters that are removed
NON_PROTEIN = {'TIP', 'TIP3', 'HOH', 'WAT', 'SOL', 'SOD', 'CLA', 'POT', 'NA', 'CL'}


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


def reference_residues(pdb_path: str):
    """Residues with a CA in the first model and chain, without waters and ions."""
    # not 'protein', 2F4K has its norleucines as HETATM
    s = prody.parsePDB(pdb_path, model=1).select('not water and not ion')
    s = s.select(f'chain {s.getChids()[0]}')
    return [r for r in s.getHierView().iterResidues() if r.select('name CA') is not None]


def get_native_cg(pdb_path: str, first: int = 0, n_res: int = None):
    """Load PDB, coarse-grain to residue level, return CG positions and distance matrix."""
    res = reference_residues(pdb_path)
    res = res[first:] if n_res is None else res[first:first + n_res]
    if n_res is not None and len(res) != n_res:
        raise ValueError(f'{pdb_path}: {len(res)} residues from {first}, expected {n_res}')
    coords = np.concatenate([r.getCoords() for r in res])
    cluster = torch.cat([torch.full((r.numAtoms(),), i, dtype=torch.long)
                         for i, r in enumerate(res)])
    native_cg, _ = compute_com(torch.tensor(coords, dtype=torch.float32).unsqueeze(1),
                               torch.ones(coords.shape[0]), cluster)
    native_cg = native_cg[:, 0, :]
    native_dm = compute_distance_matrix(native_cg)
    return native_cg, native_dm, native_cg.shape[0]


def residue_sequence(residues):
    return ''.join(prody.atomic.AAMAP.get(r.getResname(), 'X') for r in residues)


def native_reference(protein_name: str):
    """Native CG positions, distance matrix, n_beads and the scored bead indices."""
    cfg = PROTEIN_CONFIGS[protein_name]
    sim_first, ref_first, n_res = cfg['native_map']
    native_cg, native_dm, _ = get_native_cg(cfg['native_pdb'], ref_first, n_res)
    sim_seq = simulated_sequence(cfg['data_dir'])
    n_beads = len(sim_seq)
    ref_seq = residue_sequence(reference_residues(cfg['native_pdb'])[ref_first:ref_first + n_res])
    pairs = list(zip(sim_seq[sim_first:sim_first + n_res], ref_seq))
    identity = sum(a == b for a, b in pairs) / n_res
    if len(pairs) != n_res or identity < 0.8:
        raise ValueError(f'{protein_name}: reference mapping {cfg["native_map"]} gives '
                         f'{identity:.0%} identity\n  sim {sim_seq}\n  ref {ref_seq}')
    bead_idx = torch.arange(sim_first, sim_first + n_res)
    return native_cg, native_dm, n_beads, bead_idx


def topology_reference(protein_name: str):
    """CG positions and distance matrix of topology_pdb, for the old scores."""
    cfg = PROTEIN_CONFIGS[protein_name]
    sim_first, _ref_first, n_res = cfg['native_map']
    return get_native_cg(cfg['topology_pdb'], sim_first, n_res)[:2]


def list_trajs(data_dir: str, test_trajs: list = None):
    if test_trajs is not None:
        return test_trajs
    return sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('traj_')],
                  key=lambda d: int(d.split('_')[-1]))


def load_traj(tp: str, frames=None, center: bool = False):
    """Load one segment without non-protein atoms, optionally centered per frame."""
    with h5py.File(f'{tp}/position.h5', 'r') as h:
        position = h['position'][...] if frames is None else h['position'][frames]
    position = np.asarray(position, dtype=np.float32)
    ptype = load_data(['particle_type'], f'{tp}/ptype.h5')[0]
    bonds = load_data(['bond_indices'], f'{tp}/bond.h5')[0].astype(np.int64)
    rgs = load_data(['rgs'], f'{tp}/rgs.h5')[0]
    rgs = rgs if frames is None else rgs[frames]
    residue_num = load_data(['residue_num'], f'{tp}/residue_num.h5')[0]
    resname = load_data(['residue'], f'{tp}/residue.h5')[0]
    keep = np.array([r.decode().strip() not in NON_PROTEIN for r in resname])
    if not keep.all():
        new_idx = np.cumsum(keep) - 1
        bonds = bonds[keep[bonds].all(1)]
        bonds = new_idx[bonds]
        position, ptype, residue_num = position[:, keep], ptype[keep], residue_num[keep]
        mass = load_data(['mass'], f'{tp}/mass.h5')[0][keep]
        com = (position * mass[None, :, None]).sum(1, keepdims=True) / mass.sum()
        rgs = np.sqrt((((position - com) ** 2).sum(-1) * mass).sum(-1) / mass.sum())
    if center:
        position = position - position.mean(1, keepdims=True)
    longest = np.linalg.norm(position[:, bonds[:, 0]] - position[:, bonds[:, 1]], axis=-1).max()
    if longest > 3.0:
        raise ValueError(f'{tp}: a bond is {longest:.1f} A long, the molecule is split '
                         'across the periodic box and has to be made whole first')
    return {'position': position, 'ptype': ptype, 'bonds': bonds, 'rgs': rgs,
            'residue_num': residue_num}


def simulated_sequence(data_dir: str):
    """One letter per simulated protein residue, read from the first segment."""
    tp = os.path.join(data_dir, list_trajs(data_dir)[0])
    resname = load_data(['residue'], f'{tp}/residue.h5')[0]
    residue_num = load_data(['residue_num'], f'{tp}/residue_num.h5')[0]
    seq = {}
    for n, r in zip(residue_num, resname):
        r = r.decode().strip()
        if r not in NON_PROTEIN:
            seq.setdefault(n, prody.atomic.AAMAP.get(r, 'X'))
    return ''.join(seq[n] for n in sorted(seq))


def cg_traj(data: dict):
    """Unit-weight residue centroids of a loaded segment, [n_beads, n_frames, 3]."""
    cluster = torch.tensor(data['residue_num'] - min(data['residue_num']), dtype=torch.long)
    cg_pos, _ = compute_com(torch.tensor(data['position']).permute(1, 0, 2),
                            torch.ones(data['position'].shape[1]), cluster)
    return cg_pos


def kabsch_rmsd_frames(pos: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """compute_rmsd for every frame at once. pos [T, N, 3], ref [N, 3]."""
    p = pos.float() - pos.float().mean(1, keepdim=True)
    q = (ref.float() - ref.float().mean(0, keepdim=True)).expand_as(p)
    U, S, Vt = torch.linalg.svd(p.transpose(1, 2) @ q)
    d = torch.sign(torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    return torch.sqrt(((p @ R.transpose(1, 2) - q) ** 2).sum(-1).mean(-1))


def check_reference(protein_name: str, limit: float = None, stride: int = 10,
                    max_trajs: int = 60):
    """Fail unless some held-out MD frame comes within limit A of the native structure."""
    cfg = PROTEIN_CONFIGS[protein_name]
    native_cg, _, n_beads, bead_idx = native_reference(protein_name)
    limit = FOLDING_THRESHOLDS[protein_name]['folded'] if limit is None else limit
    trajs = list_trajs(cfg['data_dir'], cfg['test_trajs'])
    trajs = trajs[::max(1, len(trajs) // max_trajs)]
    best = (np.inf, None, None)
    for traj_name in trajs:
        tp = os.path.join(cfg['data_dir'], traj_name)
        if not os.path.exists(f'{tp}/position.h5'):
            continue
        cg = cg_traj(load_traj(tp, np.s_[::stride]))[bead_idx].permute(1, 0, 2)
        rmsd = kabsch_rmsd_frames(cg, native_cg)
        i = int(rmsd.argmin())
        if rmsd[i] < best[0]:
            best = (float(rmsd[i]), traj_name, i * stride)
        if best[0] <= limit:
            break
    if best[0] > limit:
        raise RuntimeError(f'{protein_name}: no held-out MD frame comes within {limit} A '
                           f'of {cfg["native_pdb"]} (closest {best[0]:.2f} A at '
                           f'{best[1]}@{best[2]}); the reference or its residue mapping '
                           'is wrong')
    logging.info(f'  Reference check: MD reaches {best[0]:.2f} A of the native '
                 f'structure at {best[1]}@{best[2]} (limit {limit} A)')
    return best


def find_unfolded_start(data_dir: str, test_trajs: list, n_beads: int, top_k: int = 5,
                        center: bool = False):
    """Find the most unfolded 30-frame windows in the test trajectories."""
    candidates = []
    for traj_name in list_trajs(data_dir, test_trajs):
        tp = os.path.join(data_dir, traj_name)
        if not os.path.exists(f'{tp}/position.h5'):
            continue
        # first 1000 frames only
        cg_pos = cg_traj(load_traj(tp, np.s_[:1000], center))
        rgs = np.array([
            compute_weighted_rgs(cg_pos[:, t], torch.ones(n_beads), torch.tensor([n_beads])).item()
            for t in range(cg_pos.shape[1])
        ])
        # find 30-frame window with highest avg Rg
        for s in range(0, len(rgs) - 30):
            avg_rg = rgs[s:s + 30].mean()
            candidates.append((traj_name, s, avg_rg))

    candidates.sort(key=lambda x: -x[2])
    return candidates[:top_k]


def build_batch(data_dir: str, traj_name: str, start_frame: int, n_beads: int,
                center: bool = False):
    """Build a model input batch from a specific trajectory and starting frame."""
    tp = os.path.join(data_dir, traj_name)
    sel = np.arange(start_frame, start_frame + 31)
    data = load_traj(tp, sel, center)
    position, ptype, bonds = data['position'], data['ptype'], data['bonds']
    residue_num = data['residue_num']
    cluster = torch.tensor(residue_num - min(residue_num), dtype=torch.long)
    if cluster.max().item() + 1 != n_beads:
        raise ValueError(f'{tp}: {cluster.max().item() + 1} residues, expected {n_beads}')

    rev_bonds = np.concatenate([bonds[:, 1:], bonds[:, :1]], axis=1)
    ab = torch.tensor(np.concatenate([bonds, rev_bonds], axis=0), dtype=torch.long)
    row, col = ab[:, 0], ab[:, 1]
    boundary = cluster[row] != cluster[col]
    cg_bonds = torch.unique(
        torch.cat([cluster[row[boundary]][:, None], cluster[col[boundary]][:, None]], dim=1), dim=0)

    return {
        'position': torch.tensor(position[:-1], dtype=torch.float32).permute(1, 0, 2),
        'target': torch.tensor(position[-1], dtype=torch.float32),
        'particle_types': torch.tensor(ptype, dtype=torch.long),
        'n_particle': torch.tensor([ptype.shape[0]], dtype=torch.long),
        'bonds': ab, 'n_bond': torch.tensor([ab.shape[0]], dtype=torch.long),
        'rgs': torch.tensor(data['rgs'][:-1], dtype=torch.float32).unsqueeze(0),
        'residue_num': torch.tensor(residue_num, dtype=torch.long),
        'n_keypoint': torch.tensor([n_beads], dtype=torch.long),
        'cluster': cluster, 'cg_bonds': cg_bonds,
        'n_cg_bond': torch.tensor([len(cg_bonds)], dtype=torch.long),
    }


def save_results(path: str, results: list, thresh_fold: float, thresh_border: float):
    """Cache the per-rollout series to an .npz."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        rmsds=np.array([r['rmsds'] for r in results], dtype=np.float32),
        rg_vals=np.array([r['rg_vals'] for r in results], dtype=np.float32),
        q_frames=np.array([f for f, _ in results[0]['q_vals']], dtype=np.int32),
        q_series=np.array([[q for _, q in r['q_vals']] for r in results],
                          dtype=np.float32),
        min_rmsd=np.array([r['min_rmsd'] for r in results], dtype=np.float32),
        min_frame=np.array([r['min_frame'] for r in results], dtype=np.int32),
        final_rmsd=np.array([r['final_rmsd'] for r in results], dtype=np.float32),
        q_at_best=np.array([r['q_at_best'] for r in results], dtype=np.float32),
        q_final=np.array([r['q_final'] for r in results], dtype=np.float32),
        min_rg=np.array([r['min_rg'] for r in results], dtype=np.float32),
        final_rg=np.array([r['final_rg'] for r in results], dtype=np.float32),
        status=np.array([r['status'] for r in results]),
        thresholds=np.array([thresh_fold, thresh_border], dtype=np.float32),
    )
    logging.info(f'  Cached rollout series to {path}')


def run_folding_test(protein_name: str, model_path: str, n_rollouts: int = 10,
                     rollout_length: int = 2000, max_vel: float = 3.0,
                     cache_dir: str = CACHE_DIR, tag: str = '', start: str = None,
                     rollouts: list = None,
                     search_trajs: list = None, progress: bool = False):
    """Run the full folding evaluation for one protein.

    start ('traj_56@198') skips the start search, rollout r is seeded with r * 42 + 1.
    """
    cache_path = os.path.join(cache_dir, f'folding_{protein_name}{tag}.npz')
    cfg = PROTEIN_CONFIGS[protein_name]
    center = cfg.get('center', False)
    thresholds = FOLDING_THRESHOLDS[protein_name]
    thresh_fold = thresholds['folded']
    thresh_border = thresholds['borderline']

    logging.info(f'Evaluating {protein_name} (folded ≤ {thresh_fold}Å, borderline ≤ {thresh_border}Å)')

    # native structure
    native_cg, native_dm, n_beads, bead_idx = native_reference(protein_name)
    n_scored = len(bead_idx)
    n_native_contacts = sum(1 for i in range(n_scored) for j in range(i+4, n_scored)
                           if native_dm[i, j] < 8.0)
    native_rg = compute_weighted_rgs(native_cg, torch.ones(n_scored),
                                     torch.tensor([n_scored])).item()
    logging.info(f'  Native: {cfg["native_pdb"]}, {n_beads} beads, {n_scored} scored, '
                 f'{n_native_contacts} contacts, Rg {native_rg:.2f}Å')
    check_reference(protein_name)
    topo_cg, topo_dm = topology_reference(protein_name)

    # find unfolded starting point
    if start is None:
        trajs = search_trajs if search_trajs is not None else cfg['test_trajs']
        traj_name, start_frame, start_rg = find_unfolded_start(
            cfg['data_dir'], trajs, n_beads, center=center)[0]
    else:
        traj_name, start_frame = start.split('@')
        start_frame = int(start_frame)
        cg = cg_traj(load_traj(os.path.join(cfg['data_dir'], traj_name),
                               np.arange(start_frame, start_frame + 30), center))
        start_rg = np.mean([compute_weighted_rgs(cg[:, t], torch.ones(n_beads),
                                                 torch.tensor([n_beads])).item()
                            for t in range(30)])
    logging.info(f'  Starting from {traj_name}@{start_frame}, Rg≈{start_rg:.1f}Å'
                 f'{" (centered)" if center else ""}')

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # build batch
    batch = build_batch(cfg['data_dir'], traj_name, start_frame, n_beads, center)
    batch = {k: v.to(device) for k, v in batch.items()}

    # run rollouts
    results = []
    for r in (range(n_rollouts) if rollouts is None else rollouts):
        seed_everything(r * 42 + 1)
        b = {k: v.clone() for k, v in batch.items()}
        with torch.no_grad():
            out = model.simulate(b, rollout_length, save_positions=True, deter=False,
                                save_frequency=1, disable_bar=not progress, max_vel=max_vel)
        pred_all = out['rollout_u_pos'][:n_beads].cpu()
        pred = pred_all[bead_idx]

        rmsds, rg_vals, q_vals = [], [], []
        for t in range(pred.shape[1]):
            rmsds.append(compute_rmsd(pred[:, t], native_cg))
            rg_vals.append(compute_weighted_rgs(
                pred[:, t], torch.ones(n_scored), torch.tensor([n_scored])).item())
            if t % 10 == 0:  # Q is expensive, sample every 10 frames
                q_vals.append((t, compute_native_contacts(pred[:, t], native_dm)))

        min_rmsd = min(rmsds)
        min_frame = np.argmin(rmsds)
        q_at_best = compute_native_contacts(pred[:, min_frame], native_dm)
        q_final = compute_native_contacts(pred[:, -1], native_dm)

        folded = min_rmsd <= thresh_fold and q_at_best >= 0.7
        borderline = not folded and min_rmsd <= thresh_border
        status = 'FOLDED' if folded else ('BORDERLINE' if borderline else 'not folded')

        # scores against topology_pdb, for comparison with old runs
        topo_rmsd = kabsch_rmsd_frames(pred.permute(1, 0, 2), topo_cg)
        topo_frame = int(topo_rmsd.argmin())
        topo_q = compute_native_contacts(pred[:, topo_frame], topo_dm)
        step = (pred_all[:, 1:] - pred_all[:, :-1]).norm(dim=-1)
        clamped = (step > max_vel - 1e-3).float().mean().item()

        results.append({
            'rollout': r, 'min_rmsd': min_rmsd, 'min_frame': min_frame,
            'final_rmsd': rmsds[-1], 'q_at_best': q_at_best, 'q_final': q_final,
            'min_rg': min(rg_vals), 'final_rg': rg_vals[-1], 'status': status,
            'rmsds': rmsds, 'rg_vals': rg_vals, 'q_vals': q_vals,
        })
        logging.info(f'  R{r}: RMSD min={min_rmsd:.2f}Å @{min_frame}, Q={q_at_best:.2f}, '
                     f'final RMSD={rmsds[-1]:.2f}, Q={q_final:.2f} → {status}')
        logging.info(f'  R{r}: Rg min={min(rg_vals):.2f} final={rg_vals[-1]:.2f}Å '
                     f'(native {native_rg:.2f}, final/native {rg_vals[-1] / native_rg:.2f}), '
                     f'clamp active {clamped:.1%}, topology.pdb RMSD min='
                     f'{topo_rmsd[topo_frame]:.2f}Å @{topo_frame}, Q={topo_q:.2f}')

    # summary
    n_run = len(results)
    n_folded = sum(1 for r in results if r['status'] == 'FOLDED')
    n_border = sum(1 for r in results if r['status'] == 'BORDERLINE')
    min_rmsds = [r['min_rmsd'] for r in results]

    logging.info(f'\n  === {protein_name.upper()} SUMMARY ===')
    logging.info(f'  Folded: {n_folded}/{n_run}, Borderline: {n_border}/{n_run}')
    logging.info(f'  Min RMSD: best={min(min_rmsds):.2f}, mean={np.mean(min_rmsds):.2f}Å')
    logging.info(f'  Mean Q at best frame: {np.mean([r["q_at_best"] for r in results]):.2f}')

    save_results(cache_path, results, thresh_fold, thresh_border)

    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Folding evaluation with RMSD + Q')
    parser.add_argument('--protein', type=str, required=True,
                        choices=list(PROTEIN_CONFIGS.keys()) + ['all'])
    parser.add_argument('--cache_dir', type=str, default=CACHE_DIR)
    parser.add_argument('--tag', type=str, default='',
                        help='suffix for the cache name, e.g. _k1')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model_best file')
    parser.add_argument('--model_dir', type=str, default='/home/ubuntu/models/PROTEIN-FOLDING-MODELS',
                        help='Base model directory (used with --protein all)')
    parser.add_argument('--n_rollouts', type=int, default=10)
    parser.add_argument('--rollout_length', type=int, default=2000)
    parser.add_argument('--max_vel', type=float, default=3.0)
    parser.add_argument('--start', type=str, default=None,
                        help='start window as traj_N@frame instead of searching for one')
    parser.add_argument('--search_trajs', type=str, nargs='+', default=None,
                        help='segments to search for the start window instead of the '
                             'configured test segments')
    parser.add_argument('--rollouts', type=int, nargs='+', default=None,
                        help='run only these rollout indices (seeds r * 42 + 1)')
    parser.add_argument('--progress', action='store_true')
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
                                args.max_vel, args.cache_dir)
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
                        args.max_vel, args.cache_dir, args.tag, args.start, args.rollouts,
                        args.search_trajs, args.progress)
