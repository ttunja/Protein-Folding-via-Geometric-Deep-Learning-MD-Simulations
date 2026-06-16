#!/bin/bash
# Sequential experiment runner for the Chignolin ablation study and the
# per-protein extensions.
#
# Usage: bash run_experiments.sh [start_from] [epochs] [stop_at]
#   start_from: experiment number to start from (1-7), default 1
#   epochs:     max_epochs for each training, default 4 (the value used to
#               produce the ablation numbers currently in the thesis)
#   stop_at:    last experiment to run (inclusive), default 7
#
# Examples:
#   bash run_experiments.sh           # full pipeline, 4 epochs each
#   bash run_experiments.sh 1 4 4     # only the four ablation variants
#   bash run_experiments.sh 5 10 7    # only the per-protein full models, 10 ep

set -e
source ~/envs/protein_env/bin/activate
cd /home/ubuntu/code/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations/src

export PROJECT_ROOT="/home/ubuntu/code/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations"
export MODEL_DIR="/home/ubuntu/models/PROTEIN-FOLDING-MODELS"
export PROTEIN_DATASET_DIR="/home/ubuntu/data/PROTEIN-FOLDING-DATA/Chignolin/protein_train"

TRACKER="python utils/experiment_tracker.py"
START_FROM=${1:-1}
EPOCHS=${2:-4}
STOP_AT=${3:-7}

guard() {
    local n=$1
    [ "$START_FROM" -le "$n" ] && [ "$n" -le "$STOP_AT" ]
}

echo "============================================"
echo "Experiment queue: steps $START_FROM..$STOP_AT, ${EPOCHS} epochs each"
echo "============================================"

# ------------------------------------------------------------------
# Experiment 0: No temporal encoding on Chignolin (Markov-style ablation)
#   The 30-frame velocity/edge window is flattened and projected with a
#   single Linear layer. No GRU, no RNN. Tests whether the temporal
#   encoder is needed at all.
# ------------------------------------------------------------------
if guard 0; then
    EXP_NAME="no_temporal_chignolin"
    EXP_DIR="$MODEL_DIR/Chignolin/chignolin_no_temporal"
    echo ""
    echo "[Exp 0/7] $EXP_NAME"
    echo "  No temporal encoding on Chignolin"

    python train.py --config-name=chignolin \
        model=no_temporal \
        train.pl_trainer.max_epochs=$EPOCHS \
        2>&1 | tee /tmp/${EXP_NAME}.log

    $TRACKER register \
        --name "$EXP_NAME" \
        --log "$EXP_DIR/train.log" \
        --config "$EXP_DIR/train_config.yaml" \
        --desc "Ablation: no temporal encoding on Chignolin ($EPOCHS epochs)" || true
fi

# ------------------------------------------------------------------
# Experiment 1: Baseline (RNN + non-equivariant) on Chignolin
# NOTE: This is already running separately. Skip if already done.
# ------------------------------------------------------------------
if guard 1; then
    EXP_NAME="baseline_chignolin"
    EXP_DIR="$MODEL_DIR/Chignolin/chignolin_baseline"
    echo ""
    echo "[Exp 1/7] $EXP_NAME"
    echo "  RNN + non-equivariant on Chignolin"

    python train.py --config-name=chignolin \
        model=baseline \
        train.pl_trainer.max_epochs=$EPOCHS \
        2>&1 | tee /tmp/${EXP_NAME}.log

    $TRACKER register \
        --name "$EXP_NAME" \
        --log "$EXP_DIR/train.log" \
        --config "$EXP_DIR/train_config.yaml" \
        --desc "Ablation: RNN + non-equivariant on Chignolin ($EPOCHS epochs)"
fi

# ------------------------------------------------------------------
# Experiment 2: GRU-only (GRU + non-equivariant) on Chignolin
# ------------------------------------------------------------------
if guard 2; then
    EXP_NAME="gru_only_chignolin"
    EXP_DIR="$MODEL_DIR/Chignolin/chignolin_gru_only"
    echo ""
    echo "[Exp 2/7] $EXP_NAME"
    echo "  GRU + non-equivariant on Chignolin"

    python train.py --config-name=chignolin \
        model=gru_only \
        train.pl_trainer.max_epochs=$EPOCHS \
        2>&1 | tee /tmp/${EXP_NAME}.log

    $TRACKER register \
        --name "$EXP_NAME" \
        --log "$EXP_DIR/train.log" \
        --config "$EXP_DIR/train_config.yaml" \
        --desc "Ablation: GRU + non-equivariant on Chignolin ($EPOCHS epochs)"
fi

# ------------------------------------------------------------------
# Experiment 3: EGNN-only (RNN + equivariant) on Chignolin
# ------------------------------------------------------------------
if guard 3; then
    EXP_NAME="egnn_only_chignolin"
    EXP_DIR="$MODEL_DIR/Chignolin/chignolin_egnn_only"
    echo ""
    echo "[Exp 3/7] $EXP_NAME"
    echo "  RNN + EGNN on Chignolin"

    python train.py --config-name=chignolin \
        model=egnn_only \
        train.pl_trainer.max_epochs=$EPOCHS \
        2>&1 | tee /tmp/${EXP_NAME}.log

    $TRACKER register \
        --name "$EXP_NAME" \
        --log "$EXP_DIR/train.log" \
        --config "$EXP_DIR/train_config.yaml" \
        --desc "Ablation: RNN + EGNN on Chignolin ($EPOCHS epochs)"
fi

# ------------------------------------------------------------------
# Experiment 4: Full model (GRU + EGNN) on Chignolin
# ------------------------------------------------------------------
if guard 4; then
    EXP_NAME="full_model_chignolin"
    EXP_DIR="$MODEL_DIR/Chignolin/chignolin_TINGnn"
    echo ""
    echo "[Exp 4/7] $EXP_NAME"
    echo "  GRU + EGNN (full model) on Chignolin"

    python train.py --config-name=chignolin \
        train.pl_trainer.max_epochs=$EPOCHS \
        2>&1 | tee /tmp/${EXP_NAME}.log

    $TRACKER register \
        --name "$EXP_NAME" \
        --log "$EXP_DIR/train.log" \
        --config "$EXP_DIR/train_config.yaml" \
        --desc "Full model: GRU + EGNN on Chignolin ($EPOCHS epochs)"
fi

# ------------------------------------------------------------------
# Experiment 5: Full model on Trp-cage
# ------------------------------------------------------------------
if guard 5; then
    EXP_NAME="full_model_trp_cage"
    EXP_DIR="$MODEL_DIR/Trp-cage/trp_cage_TINGnn"
    echo ""
    echo "[Exp 5/7] $EXP_NAME"
    echo "  GRU + EGNN on Trp-cage"

    python train.py --config-name=trp_cage \
        model=chignolin \
        train.pl_trainer.max_epochs=$EPOCHS \
        2>&1 | tee /tmp/${EXP_NAME}.log

    $TRACKER register \
        --name "$EXP_NAME" \
        --log "$EXP_DIR/train.log" \
        --config "$EXP_DIR/train_config.yaml" \
        --desc "Full model: GRU + EGNN on Trp-cage ($EPOCHS epochs)"
fi

# ------------------------------------------------------------------
# Experiment 6: Full model on BBA
# ------------------------------------------------------------------
if guard 6; then
    EXP_NAME="full_model_bba"
    EXP_DIR="$MODEL_DIR/BBA/bba_TINGnn"
    echo ""
    echo "[Exp 6/7] $EXP_NAME"
    echo "  GRU + EGNN on BBA"

    python train.py --config-name=bba \
        model=chignolin \
        train.pl_trainer.max_epochs=$EPOCHS \
        2>&1 | tee /tmp/${EXP_NAME}.log

    $TRACKER register \
        --name "$EXP_NAME" \
        --log "$EXP_DIR/train.log" \
        --config "$EXP_DIR/train_config.yaml" \
        --desc "Full model: GRU + EGNN on BBA ($EPOCHS epochs)"
fi

# ------------------------------------------------------------------
# Experiment 7: Full model on Villin
# ------------------------------------------------------------------
if guard 7; then
    EXP_NAME="full_model_villin"
    EXP_DIR="$MODEL_DIR/Villin/villin_TINGnn"
    echo ""
    echo "[Exp 7/7] $EXP_NAME"
    echo "  GRU + EGNN on Villin"

    python train.py --config-name=villin \
        model=chignolin \
        train.pl_trainer.max_epochs=$EPOCHS \
        2>&1 | tee /tmp/${EXP_NAME}.log

    $TRACKER register \
        --name "$EXP_NAME" \
        --log "$EXP_DIR/train.log" \
        --config "$EXP_DIR/train_config.yaml" \
        --desc "Full model: GRU + EGNN on Villin ($EPOCHS epochs)"
fi

# ------------------------------------------------------------------
# Generate comparison plots
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "All experiments complete. Generating plots..."
echo "============================================"
$TRACKER plot

echo ""
echo "Done! Check docs/plots/ for comparison figures."
