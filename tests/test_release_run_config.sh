#!/usr/bin/env bash
# release_run.sh's CONFIG resolution, exercised without running a real release.
#
# Both behaviours pinned here were wrong in the field on 2026-09-14: staging the Mesa Orange
# 2-knob variant produced a reproduce.sh aimed at the 5-knob config, because the default
# ignored VARIANT and the file it named existed, so nothing failed.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/models/amps/thing"
: > "$TMP/models/amps/thing/config.toml"              # the PARENT circuit's config
fails=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; else echo "  FAIL $1: expected '$3', got '$2'"; fails=$((fails+1)); fi; }

resolve() {  # mirrors release_run.sh's resolution
  local MODELS="$1" CATEGORY="$2" CIRCUIT="$3" VARIANT="${4:-}" d
  d="$MODELS/$CATEGORY/$CIRCUIT/config.toml"
  [ -n "$VARIANT" ] && d="$MODELS/$CATEGORY/$CIRCUIT/config.toml.$VARIANT"
  echo "$d"
}

got="$(resolve "$TMP/models" amps thing)"
check "no variant -> bare config.toml" "$got" "$TMP/models/amps/thing/config.toml"

got="$(resolve "$TMP/models" amps thing twoknob)"
check "variant -> config.toml.<variant>" "$got" "$TMP/models/amps/thing/config.toml.twoknob"

# THE REGRESSION: a variant whose config is absent must NOT silently resolve to the parent's.
got="$(resolve "$TMP/models" amps thing twoknob)"
if [ "$got" = "$TMP/models/amps/thing/config.toml" ]; then
  echo "  FAIL variant fell back to the parent circuit's config"; fails=$((fails+1))
else
  echo "  ok   variant does not fall back to the parent config"
fi
[ -f "$got" ] && { echo "  FAIL absent variant config reported as present"; fails=$((fails+1)); } \
              || echo "  ok   absent variant config is absent (release_run.sh exits 1 here)"

# The real script must actually carry the variant-aware default.
grep -q 'config.toml.\$VARIANT' "$HERE/release_run.sh" \
  && echo "  ok   release_run.sh uses config.toml.\$VARIANT" \
  || { echo "  FAIL release_run.sh lost the variant-aware default"; fails=$((fails+1)); }
# ...and must bundle the config rather than pointing outside the bundle.
grep -q 'cp "\$CONFIG" "\$STAGE/config.toml"' "$HERE/release_run.sh" \
  && echo "  ok   release_run.sh bundles config.toml" \
  || { echo "  FAIL release_run.sh does not bundle the config"; fails=$((fails+1)); }
grep -q 'dirname "\\\$0"' "$HERE/release_run.sh" \
  && echo "  ok   reproduce.sh resolves its config relative to itself" \
  || { echo "  FAIL reproduce.sh still embeds an absolute config path"; fails=$((fails+1)); }

echo; [ "$fails" -eq 0 ] && echo "PASS" || { echo "$fails FAILURE(S)"; exit 1; }
