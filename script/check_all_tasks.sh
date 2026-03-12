#!/usr/bin/env bash
# Run multiview collect for all tasks: 1 episode, 1 view. Report which tasks fail.
# Usage: from repo root, ./script/check_all_tasks.sh [observer_configs.json]

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OBSERVER_CONFIGS="${1:-observer_configs.json}"
if [[ ! -f "$OBSERVER_CONFIGS" ]]; then
  echo "Error: observer configs not found: $OBSERVER_CONFIGS"
  echo "Usage: $0 [observer_configs.json]"
  exit 1
fi

# Task modules: envs/*.py excluding internal/base modules
FAILED=()
TOTAL=0

for f in envs/*.py; do
  name="$(basename "$f" .py)"
  case "$name" in
    __init__|_base_task|_multiview_task|_GLOBAL_CONFIGS) continue ;;
  esac
  TOTAL=$((TOTAL + 1))
  echo "----------------------------------------"
  echo "[$TOTAL] Checking task: $name"
  if python script/collect_data_multiview.py "$name" demo_multiview \
      --num_views 1 \
      --episodes_per_view 1 \
      --observer_configs "$OBSERVER_CONFIGS" \
      > "script/check_log_${name}.txt" 2>&1; then
    echo "  OK: $name"
  else
    echo "  FAIL: $name (see script/check_log_${name}.txt)"
    FAILED+=("$name")
  fi
done

echo ""
echo "========================================"
echo "Done: $((TOTAL - ${#FAILED[@]})) passed, ${#FAILED[@]} failed (total $TOTAL)"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed tasks:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
exit 0
