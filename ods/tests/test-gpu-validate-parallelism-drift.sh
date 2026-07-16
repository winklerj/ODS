#!/usr/bin/env bash
# ============================================================================
# ODS CLI — `ods gpu validate` parallelism-mode drift check
# ============================================================================
# Covers the Check 4 added to _gpu_validate() in ods-cli: it re-runs
# assign_gpus.py's select_parallelism() (via `--check-gpus`) against the
# GPUs currently assigned to llama-server and compares the result to
# LLAMA_ARG_SPLIT_MODE, so a stale/hand-edited .env is flagged even though
# Checks 1-3 (GPU_COUNT, UUID presence, split-mode-vs-GPU-count) all pass.
#
# Usage: ./tests/test-gpu-validate-parallelism-drift.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ODS_CLI="$ROOT_DIR/ods-cli"
FIXTURE_TOPO="$SCRIPT_DIR/fixtures/topology_json/nvidia_smi_topo_matrix_2gpus_phb_coloc.json"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

PASSED=0
FAILED=0
pass() { echo -e "  ${GREEN}✓ PASS${NC} $1"; PASSED=$((PASSED + 1)); }
fail() { echo -e "  ${RED}✗ FAIL${NC} $1"; FAILED=$((FAILED + 1)); }

echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║   GPU Validate — Parallelism Drift Check      ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

if ! env bash -c '(( BASH_VERSINFO[0] >= 4 ))' 2>/dev/null; then
    echo "[SKIP] ods-cli requires Bash 4+ on PATH."
    exit 0
fi

[[ -x "$ODS_CLI" ]] || { fail "ods-cli not executable at $ODS_CLI"; exit 1; }
command -v jq >/dev/null 2>&1 || { fail "jq is required"; exit 1; }
command -v python3 >/dev/null 2>&1 || { fail "python3 is required"; exit 1; }
[[ -f "$FIXTURE_TOPO" ]] || { fail "fixture not found: $FIXTURE_TOPO"; exit 1; }

FIXTURE=$(mktemp -d /tmp/test-gpu-validate-drift.XXXXXX)
FAKE_INSTALL="$FIXTURE/install"
mkdir -p "$FAKE_INSTALL/config"
trap 'rm -rf "$FIXTURE"' EXIT

: > "$FAKE_INSTALL/docker-compose.base.yml"
cp "$FIXTURE_TOPO" "$FAKE_INSTALL/config/gpu-topology.json"

# PHB rank=30, 2 GPUs → select_parallelism() recommends pipeline (split_mode=layer).
UUID_0="GPU-00000000-0000-0000-0000-000000000000"
UUID_1="GPU-11111111-1111-1111-1111-111111111111"

write_env() {
    local split_mode="$1"
    cat > "$FAKE_INSTALL/.env" <<EOF
GPU_COUNT=2
GPU_BACKEND=nvidia
LLAMA_SERVER_GPU_UUIDS=${UUID_0},${UUID_1}
LLAMA_ARG_SPLIT_MODE=${split_mode}
GPU_ASSIGNMENT_JSON_B64=e30=
EOF
}

run_ods_cli() {
    set +e
    OUT=$(ODS_HOME="$FAKE_INSTALL" "$ODS_CLI" "$@" 2>&1)
    RC=$?
    set -e
}

# ----------------------------------------------------------------------------
# Case 1: configured split_mode (row/tensor) disagrees with the topology
# recommendation (pipeline/layer for a same-NUMA PCIe PHB pair) → warn.
# ----------------------------------------------------------------------------
echo "── 1. mismatched split_mode is flagged ──"
write_env "row"
run_ods_cli gpu validate

if [[ $RC -eq 0 ]]; then
    pass "ods gpu validate exits 0"
else
    fail "ods gpu validate exited $RC; output: $OUT"
fi

if echo "$OUT" | grep -q "differs from the topology-recommended mode (pipeline -> split_mode=layer)"; then
    pass "flags LLAMA_ARG_SPLIT_MODE=row as inconsistent with the pipeline recommendation"
else
    fail "expected drift warning not found; output: $OUT"
fi

if echo "$OUT" | grep -q "ods gpu reassign --auto"; then
    pass "fix hint points at 'ods gpu reassign --auto'"
else
    fail "fix hint missing; output: $OUT"
fi

# ----------------------------------------------------------------------------
# Case 2: configured split_mode matches the recommendation → success, no warn.
# ----------------------------------------------------------------------------
echo "── 2. matching split_mode passes cleanly ──"
write_env "layer"
run_ods_cli gpu validate

if echo "$OUT" | grep -q "matches the topology-recommended mode (pipeline)"; then
    pass "confirms LLAMA_ARG_SPLIT_MODE=layer matches the pipeline recommendation"
else
    fail "expected success line not found; output: $OUT"
fi

if echo "$OUT" | grep -q "differs from the topology-recommended mode"; then
    fail "unexpected drift warning on a matching config; output: $OUT"
else
    pass "no drift warning when config already matches"
fi

# ----------------------------------------------------------------------------
# Case 3: no cached topology → skip cleanly instead of a false positive/negative.
# ----------------------------------------------------------------------------
echo "── 3. missing topology cache is skipped, not misreported ──"
rm -f "$FAKE_INSTALL/config/gpu-topology.json"
run_ods_cli gpu validate

if echo "$OUT" | grep -q "Skipping parallelism-mode check"; then
    pass "skips the check when no topology cache is present"
else
    fail "expected skip message not found; output: $OUT"
fi

if echo "$OUT" | grep -qE "matches the topology-recommended mode|differs from the topology-recommended mode"; then
    fail "should not report a verdict without a topology cache; output: $OUT"
else
    pass "no verdict reported without a topology cache"
fi

# ----------------------------------------------------------------------------
# Case 4: single-GPU llama-server assignment → mode is trivially "none",
# Check 4 must stay silent (nothing to compare).
# ----------------------------------------------------------------------------
echo "── 4. single llama GPU produces no drift check output ──"
cp "$FIXTURE_TOPO" "$FAKE_INSTALL/config/gpu-topology.json"
cat > "$FAKE_INSTALL/.env" <<EOF
GPU_COUNT=2
GPU_BACKEND=nvidia
LLAMA_SERVER_GPU_UUIDS=${UUID_0}
LLAMA_ARG_SPLIT_MODE=none
GPU_ASSIGNMENT_JSON_B64=e30=
EOF
run_ods_cli gpu validate

if echo "$OUT" | grep -qE "topology-recommended mode|Skipping parallelism-mode check"; then
    fail "drift check should not run for a single llama GPU; output: $OUT"
else
    pass "no drift-check output for a single llama GPU"
fi

echo ""
echo "Result: $PASSED passed, $FAILED failed"
[[ $FAILED -eq 0 ]]
