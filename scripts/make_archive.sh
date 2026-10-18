#!/usr/bin/env bash
# Build the submission archive.
#
# Uses `git archive` rather than `zip -r .` on purpose: it ships exactly the
# tracked files, so the 3.7 GB frame cache, the virtualenv, __pycache__ and
# anything holding a key cannot be swept in by accident. The checks afterwards
# verify that rather than assume it.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${1:-../hack-opencv1-submission.zip}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "WARNING: uncommitted changes exist; git archive ships HEAD, not the working tree."
  git status --short
fi

rm -f "$OUT"
git archive --format=zip -9 --prefix=smoke-sentinel/ -o "$OUT" HEAD

echo
echo "--- contents check ---"
LISTING=$(unzip -Z1 "$OUT")
COUNT=$(printf '%s\n' "$LISTING" | wc -l)
echo "files: $COUNT"
echo "size:  $(du -h "$OUT" | cut -f1)"

fail=0
check_absent() {
  local pattern="$1" label="$2"
  local hits
  hits=$(printf '%s\n' "$LISTING" | grep -E "$pattern" || true)
  if [ -n "$hits" ]; then
    echo "FAIL: $label present in the archive:"
    printf '%s\n' "$hits" | head -5
    fail=1
  else
    echo "ok:   no $label"
  fi
}

check_absent '(^|/)\.venv/'        "virtualenv"
check_absent '(^|/)data/'          "frame cache / data directory"
check_absent '__pycache__|\.pyc$'  "python caches"
check_absent '(^|/)\.env$|(^|/)\.env\.'  "env files"
check_absent '\.(mp4|webm|mp3)$'   "media files"
check_absent '(^|/)out[0-9]*/'     "recording output directories"

check_present() {
  if printf '%s\n' "$LISTING" | grep -qE "$1"; then
    echo "ok:   $2 present"
  else
    echo "FAIL: $2 MISSING"
    fail=1
  fi
}

check_present 'smoke-sentinel/README\.md$'            "README"
check_present 'smoke-sentinel/docs/report\.md$'       "technical report"
check_present 'smoke-sentinel/docs/architecture\.png$' "architecture diagram"
check_present 'smoke-sentinel/docs/agent-workflow\.png$' "agent workflow diagram"
check_present 'smoke-sentinel/uv\.lock$'              "pinned dependencies"
check_present 'smoke-sentinel/src/sentinel/detector\.py$' "detector source"
check_present 'smoke-sentinel/evaluation/results\.json$'  "evaluation results"
check_present 'smoke-sentinel/web/pack/index\.json$'  "demo pack"
check_present 'smoke-sentinel/infra/template\.yaml$'  "IaC"

# A key in a tracked file would be a much worse failure than a stray cache.
echo
if git grep -nIE '(sk-[A-Za-z0-9]{16,}|xi-api-key:[[:space:]]*[A-Za-z0-9]|AKIA[0-9A-Z]{16})' -- . >/dev/null 2>&1; then
  echo "FAIL: something that looks like a credential is tracked:"
  git grep -nIE '(sk-[A-Za-z0-9]{16,}|xi-api-key:[[:space:]]*[A-Za-z0-9]|AKIA[0-9A-Z]{16})' -- . | head -5
  fail=1
else
  echo "ok:   no credential-shaped strings in tracked files"
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "ARCHIVE REJECTED: fix the failures above."
  exit 1
fi
echo "ARCHIVE OK -> $(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"
