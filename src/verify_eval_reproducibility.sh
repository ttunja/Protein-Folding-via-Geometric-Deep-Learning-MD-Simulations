#!/bin/bash
# Re-runs every cross-protein, joint-multi-protein, and zero-shot evaluation
# that produces a number cited in Chapter 6 of the thesis. Logs are written
# to /tmp/verify_eval/ rather than MODEL_DIR so the original logs (which
# generated the published numbers) stay intact.
#
# Usage: bash verify_eval_reproducibility.sh

set -e
source ~/envs/protein_env/bin/activate
cd /home/ubuntu/code/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations/src

export PROJECT_ROOT="/home/ubuntu/code/Protein-Folding-via-Geometric-Deep-Learning-MD-Simulations"
export MODEL_DIR="/home/ubuntu/models/PROTEIN-FOLDING-MODELS"

OUT=/tmp/verify_eval
mkdir -p "$OUT"
SUMMARY="$OUT/_summary.log"
: > "$SUMMARY"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$SUMMARY"; }

run_eval() {
    local LABEL=$1 PROTEIN=$2 MPATH=$3 STEPS=$4 NROLL=$5
    local LOGF="$OUT/${LABEL}.log"
    [ ! -f "$MPATH" ] && { log "  [$LABEL] model not found: $MPATH"; return; }
    if [ -f "$LOGF" ] && grep -q "SUMMARY" "$LOGF" 2>/dev/null; then
        log "  [$LABEL] already completed in this output dir, skipping"
        grep "Min RMSD\|Mean Q\|Folded:" "$LOGF" | tee -a "$SUMMARY"
        return
    fi
    log ">>> [$LABEL] $PROTEIN @ ${STEPS} steps x ${NROLL} rollouts"
    python eval_folding.py --protein "$PROTEIN" \
        --model_path "$MPATH" --rollout_length "$STEPS" --n_rollouts "$NROLL" \
        --output_dir "$OUT/${LABEL}" 2>&1 | tee "$LOGF" > /dev/null
    if grep -q "SUMMARY" "$LOGF"; then
        grep "Min RMSD\|Mean Q\|Folded:" "$LOGF" | tee -a "$SUMMARY"
        log "<<< [$LABEL] done"
    else
        log "!!! [$LABEL] FAILED - see $LOGF"
        tail -20 "$LOGF" | tee -a "$SUMMARY"
    fi
}

CHIG_K5="$MODEL_DIR/Chignolin/chignolin_gru_multistep/model_best"
COMB_K5="$MODEL_DIR/Combined/combined_gru_multistep/model_best"

# === Phase 1: decisive evidence (one short eval per category) ===========
# Cross-protein already covered by cross_trp_cage and cross_bba.
run_eval "cross_trp_cage"  trp_cage  "$CHIG_K5" 70000 3
run_eval "cross_bba"       bba       "$CHIG_K5" 90000 3
run_eval "joint_chignolin" chignolin "$COMB_K5" 10000 5
run_eval "zs_ntl9"         ntl9      "$COMB_K5"  7500 3

# === Phase 2: complete coverage of all three blocks ====================
run_eval "cross_villin"   villin    "$CHIG_K5"  14000 3
run_eval "joint_trp_cage" trp_cage  "$COMB_K5"  70000 3
run_eval "joint_bba"      bba       "$COMB_K5"  90000 3
run_eval "joint_villin"   villin    "$COMB_K5"  14000 3
run_eval "zs_ww"          ww        "$COMB_K5" 105000 3
run_eval "zs_protein_b"   protein_b "$COMB_K5"  17500 3

log "============================================"
log "All verification evals complete. Summary: $SUMMARY"
log "============================================"
