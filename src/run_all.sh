#!/bin/bash
# Master script: waits for running evals, then runs full pipeline.
# Launched once, handles everything.
#
# Started: $(date)

set -e
source ~/envs/protein_env/bin/activate
cd /home/ubuntu/code/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations/src

export PROJECT_ROOT="/home/ubuntu/code/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations"
export MODEL_DIR="/home/ubuntu/models/PROTEIN-FOLDING-MODELS"
DATA_ROOT="/home/ubuntu/data/PROTEIN-FOLDING-DATA"
PLOTS="$PROJECT_ROOT/docs/plots"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"; }

# ============================================================
# Wait for any running eval_folding processes to finish
# ============================================================
log "Waiting for running eval_folding processes to finish..."
while pgrep -f eval_folding > /dev/null 2>&1; do
    sleep 30
done
log "All evals done."

# ============================================================
# PHASE 1: Preprocessing unseen proteins
# ============================================================
log "=== PHASE 1: Preprocessing unseen proteins ==="

preprocess_protein() {
    local NAME=$1
    local DIR=$2
    local SAVE_DIR="$DATA_ROOT/$DIR/protein_train"

    if [ -d "$SAVE_DIR" ] && [ "$(ls "$SAVE_DIR" 2>/dev/null | wc -l)" -gt 0 ]; then
        log "  $NAME: already preprocessed ($(ls "$SAVE_DIR" | wc -l) trajs)"
        return
    fi

    log "  $NAME: preprocessing..."
    mkdir -p "$SAVE_DIR"
    cd "$DATA_ROOT/$DIR"

    for tar in *-protein.tar; do
        [ -f "$tar" ] || continue
        dirname="${tar%.tar}"
        [ -d "$dirname" ] || tar xf "$tar"
    done

    for dcd_dir in DESRES-Trajectory_*-protein; do
        [ -d "$dcd_dir" ] || continue
        TOP=$(find "$dcd_dir" -name "*.pdb" | head -1)
        [ -z "$TOP" ] && TOP=$(find "$DATA_ROOT/$DIR" -name "*.pdb" | head -1)
        [ -z "$TOP" ] && { log "  WARNING: No PDB for $NAME"; continue; }
        [ -f "$DATA_ROOT/$DIR/topology.pdb" ] || cp "$TOP" "$DATA_ROOT/$DIR/topology.pdb"

        cd /home/ubuntu/code/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations/src
        log "  $NAME: processing $dcd_dir..."
        python preprocessing/preprocess.py \
            --dir "$DATA_ROOT/$DIR/$dcd_dir" \
            --top_path "$TOP" \
            --save_dir "$SAVE_DIR" || log "  WARNING: Failed $dcd_dir"
    done
    cd /home/ubuntu/code/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations/src
    log "  $NAME: $(ls "$SAVE_DIR" 2>/dev/null | wc -l) trajectories"
}

preprocess_protein "NTL9" "NTL9"
preprocess_protein "WW domain" "WW"
preprocess_protein "Protein B" "Protein_B"

# ============================================================
# PHASE 2: Training
# ============================================================
log "=== PHASE 2: Training ==="

# Trp-cage K=5
TRP_MODEL="$MODEL_DIR/Trp-cage/trp_cage_gru_multistep/model_best"
if [ ! -f "$TRP_MODEL" ]; then
    log "  Training Trp-cage K=5..."
    python train.py --config-name=trp_cage \
        model=gru_only model.rollout_steps=5 model.name=gru_multistep \
        train.pl_trainer.max_epochs=10
    log "  Trp-cage K=5 done"
else
    log "  Trp-cage K=5 exists, skipping"
fi

# Combined 4-protein K=5
COMB_MODEL="$MODEL_DIR/Combined/combined_gru_multistep/model_best"
if [ ! -f "$COMB_MODEL" ]; then
    log "  Training Combined 4-protein K=5..."
    python train.py --config-name=combined \
        model=gru_only model.rollout_steps=5 model.name=gru_multistep \
        train.pl_trainer.max_epochs=10
    log "  Combined K=5 done"
else
    log "  Combined K=5 exists, skipping"
fi

# ============================================================
# PHASE 3: Folding evaluations
# ============================================================
log "=== PHASE 3: Folding evaluations ==="

run_eval() {
    local LABEL=$1 PROTEIN=$2 MPATH=$3 STEPS=$4 NROLL=$5 OUTDIR=$6 LOGF=$7
    if [ -f "$LOGF" ] && grep -q "SUMMARY" "$LOGF" 2>/dev/null; then
        log "  [$LABEL] done, skipping"; return
    fi
    [ ! -f "$MPATH" ] && { log "  [$LABEL] model not found: $MPATH"; return; }
    log "  [$LABEL] $PROTEIN, ${STEPS} steps, ${NROLL} rollouts..."
    python eval_folding.py --protein "$PROTEIN" \
        --model_path "$MPATH" --rollout_length "$STEPS" --n_rollouts "$NROLL" \
        --output_dir "$OUTDIR" 2>&1 | tee "$LOGF"
}

# Trp-cage multistep own model
run_eval "Trp-cage K=5 own" trp_cage \
    "$MODEL_DIR/Trp-cage/trp_cage_gru_multistep/model_best" \
    70000 3 "$PLOTS/multistep" \
    "$MODEL_DIR/Trp-cage/trp_cage_gru_multistep/eval_folding_70k.log"

# Combined → all 4 known proteins
run_eval "Combined→Chig" chignolin "$COMB_MODEL" 10000 5 "$PLOTS/combined" \
    "$MODEL_DIR/Combined/eval_chignolin_10k.log"
run_eval "Combined→Trp" trp_cage "$COMB_MODEL" 70000 3 "$PLOTS/combined" \
    "$MODEL_DIR/Combined/eval_trp_cage_70k.log"
run_eval "Combined→BBA" bba "$COMB_MODEL" 90000 3 "$PLOTS/combined" \
    "$MODEL_DIR/Combined/eval_bba_90k.log"
run_eval "Combined→Villin" villin "$COMB_MODEL" 14000 3 "$PLOTS/combined" \
    "$MODEL_DIR/Combined/eval_villin_14k.log"

# Cross-protein: Chig multistep → remaining (skip if already done from earlier run)
run_eval "Chig→BBA" bba \
    "$MODEL_DIR/Chignolin/chignolin_gru_multistep/model_best" \
    90000 3 "$PLOTS/cross_protein" \
    "$MODEL_DIR/BBA/eval_cross_chig_multistep_90k.log"
run_eval "Chig→Villin" villin \
    "$MODEL_DIR/Chignolin/chignolin_gru_multistep/model_best" \
    14000 3 "$PLOTS/cross_protein" \
    "$MODEL_DIR/Villin/eval_cross_chig_multistep_14k.log"

# ============================================================
# PHASE 4: Unseen protein zero-shot
# ============================================================
log "=== PHASE 4: Unseen protein zero-shot ==="

# NTL9 (39 res, folding time 1.5μs = 7500 frames)
run_eval "Combined→NTL9" ntl9 "$COMB_MODEL" 7500 3 "$PLOTS/unseen" \
    "$MODEL_DIR/Combined/eval_ntl9_7500.log"

# WW domain (35 res, folding time 21μs = 105000 frames)
run_eval "Combined→WW" ww "$COMB_MODEL" 105000 3 "$PLOTS/unseen" \
    "$MODEL_DIR/Combined/eval_ww_105k.log"

# Protein B (47 res, folding time 3.5μs = 17500 frames)
run_eval "Combined→ProteinB" protein_b "$COMB_MODEL" 17500 3 "$PLOTS/unseen" \
    "$MODEL_DIR/Combined/eval_protein_b_17500.log"

log "============================================"
log "ALL DONE."
log "============================================"
