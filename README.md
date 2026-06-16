# Protein Folding via Temporal Graph Neural Networks

This project learns coarse-grained protein dynamics from molecular dynamics (MD) simulation trajectories using a GRU-based temporal graph neural network. The model operates at residue-level resolution, predicting time-integrated accelerations that drive proteins through conformational space. Instead of simulating atomistic forces at femtosecond timesteps, the network takes 30-frame windows of coarse-grained positions as input and predicts the next frame's acceleration, enabling autoregressive rollouts that cover microsecond-scale folding events in seconds of wall-clock time.

## Project Structure

```
src/
  models/           TINGnn.py (GNN architecture), simulator.py, simulator_diff.py
  preprocessing/    data.py (dataloaders), preprocess.py (DCD->HDF5 conversion)
  utils/            losses, visualizations, metrics
  conf/             Hydra YAML configs per protein and model variant
  train.py          training entry point
  eval.py           single-step evaluation
  eval_folding.py   folding test (RMSD + native contacts)
  eval_dynamics.py  long-timescale dynamics analysis
  eval_trajectory_quality.py  structural quality (Ramachandran, bonds, collisions)
data/               train/val/test split text files
```

## Setup

```bash
python -m venv ~/envs/protein_env
source ~/envs/protein_env/bin/activate
pip install -r requirements.txt
```

PyTorch Geometric wheels depend on your CUDA version. The requirements.txt pins `cu121` builds. If you have a different CUDA, grab the right wheels from https://data.pyg.org/whl/.

Raw trajectory data (DCD format) comes from D.E. Shaw Research's fast-folding protein dataset. You need to download it separately and place it under a data directory (e.g. `/home/ubuntu/data/PROTEIN-FOLDING-DATA/`).

## Preprocessing

Convert DCD trajectories to HDF5 coarse-grained format:

```bash
python src/preprocessing/preprocess.py \
    --dir <DCD_DIR> \
    --top_path <PDB_TOPOLOGY> \
    --save_dir <OUTPUT_DIR>
```

This extracts positions, bond topology, residue assignments, and radius of gyration per frame.

## Training

Training uses Hydra for configuration. Example for Chignolin with GRU-only model and multi-step rollout:

```bash
python src/train.py --config-name=chignolin \
    model=gru_only \
    model.rollout_steps=5 \
    model.name=gru_multistep \
    train.pl_trainer.max_epochs=10
```

Other proteins: replace `--config-name` with `trp_cage`, `bba`, `villin`, or `combined` (joint training on all four).

## Evaluation

**Folding test** -- runs stochastic rollouts from unfolded conformations and checks if the protein reaches the native state (RMSD + fraction of native contacts):

```bash
python src/eval_folding.py --protein chignolin \
    --model_path /path/to/model_best \
    --rollout_length 10000 --n_rollouts 5
```

**Dynamics analysis** -- extended rollouts tracking RMSD, Rg, and native contact fraction over time:

```bash
python src/eval_dynamics.py --protein chignolin \
    --model_path /path/to/model_best \
    --rollout_length 10000
```

**Trajectory quality** -- Ramachandran angles, bond length distributions, bead collision rates:

```bash
python src/eval_trajectory_quality.py --protein chignolin \
    --model_path /path/to/model_best
```

## Proteins

| Protein    | Residues | Folding time | Status  |
|------------|----------|-------------|---------|
| Chignolin  | 10       | 0.6 us      | trained |
| Trp-cage   | 20       | 14 us       | trained |
| BBA        | 28       | 18 us       | trained |
| Villin     | 35       | 2.8 us      | trained |
| NTL9       | 39       | 1.5 us      | unseen  |
| WW domain  | 35       | 21 us       | unseen  |
| Protein B  | 47       | 3.5 us      | unseen  |

Folding times from Lindorff-Larsen et al. (2011), Science 334, 517.

## Key Results

- Chignolin folds to within 2 A RMSD of the native state in 2 out of 5 rollouts from unfolded conformations.
- Multi-step training (K=5 rollout steps during training) is critical for stable autoregressive rollout; single-step models diverge within hundreds of frames.
- The GRU temporal encoder is the main driver of rollout stability -- it reduces mean trajectory error from 7.1 A (vanilla RNN) to 0.46 A.
- A combined model trained jointly on all four proteins generalizes zero-shot to unseen proteins (NTL9, WW domain, Protein B), producing physically plausible trajectories with correct bond length distributions and low collision rates.

## Citation

Master's Thesis, TU Wien, 2025. Tin M. Tunjic, supervised by Thomas Gartner.
