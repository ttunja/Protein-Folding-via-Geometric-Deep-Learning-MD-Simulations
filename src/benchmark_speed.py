"""Measure rollout throughput in frames per second, per protein.

A warm-up rollout runs first, and CUDA is synchronized around the timed part.

Usage:
    python benchmark_speed.py                       # all four training proteins
    python benchmark_speed.py --protein chignolin
    python benchmark_speed.py --steps 2000 --repeats 3
"""

import argparse
import logging
import os
import time

import torch

from eval_folding import PROTEIN_CONFIGS, build_batch, find_unfolded_start, native_reference

# spacing of the recorded DESRES frames, which is what one model step advances by
PS_PER_FRAME = 200.0

DEFAULT_MODELS = {
    'chignolin': 'Chignolin/chignolin_gru_multistep/model_best',
    'trp_cage': 'Trp-cage/trp_cage_gru_multistep/model_best',
    'bba': 'BBA/bba_gru_multistep/model_best',
    'villin': 'Villin/villin_gru_multistep/model_best',
}

MODEL_ROOT = '/home/ubuntu/models/PROTEIN-FOLDING-MODELS'


def time_rollout(model, batch, steps, max_vel=3.0):
    """Wall-clock seconds for a rollout of <steps> steps, GPU work included."""
    b = {k: v.clone() for k, v in batch.items()}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        model.simulate(b, steps, save_positions=False, deter=False,
                       save_frequency=1, disable_bar=True, max_vel=max_vel)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time() - start


def benchmark(protein, model_path, steps, repeats, warmup_steps=100):
    cfg = PROTEIN_CONFIGS[protein]
    center = cfg.get('center', False)
    _, _, n_beads, _ = native_reference(protein)

    traj_name, start_frame, _ = find_unfolded_start(
        cfg['data_dir'], cfg['test_trajs'], n_beads, center=center)[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    batch = build_batch(cfg['data_dir'], traj_name, start_frame, n_beads, center)
    batch = {k: v.to(device) for k, v in batch.items()}

    time_rollout(model, batch, warmup_steps)          # not timed
    times = [time_rollout(model, batch, steps) for _ in range(repeats)]

    best = min(times)                                  # least contended run
    return n_beads, steps / best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein', default='all',
                        choices=list(DEFAULT_MODELS) + ['all'])
    parser.add_argument('--model_root', default=MODEL_ROOT)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if torch.cuda.is_available():
        logging.info(f'device: {torch.cuda.get_device_name(0)}')
    else:
        logging.info('device: cpu')
    logging.info(f'{args.steps} steps, best of {args.repeats}\n')
    logging.info(f'{"protein":12s} {"beads":>6s} {"frames/s":>10s} {"ns/s":>8s}')

    proteins = list(DEFAULT_MODELS) if args.protein == 'all' else [args.protein]
    for name in proteins:
        path = os.path.join(args.model_root, DEFAULT_MODELS[name])
        if not os.path.exists(path):
            logging.info(f'{name:12s} {"":>6s} {"missing":>10s}')
            continue
        beads, fps = benchmark(name, path, args.steps, args.repeats)
        ns_per_s = fps * PS_PER_FRAME / 1000.0
        logging.info(f'{name:12s} {beads:>6d} {fps:>10.1f} {ns_per_s:>8.1f}')


if __name__ == '__main__':
    main()
