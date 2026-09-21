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


# THE OTHER REGRESSION (2026-09-21): a deck-sourced run (ngspice-deck/ltspice-deck, no .schx,
# but NOT a real-hardware capture either) must not be classified as capture-sourced just
# because schx is null -- it would print "N real captured .nam files (not a circuit
# simulation)" (N=0, since deck runs have no source_files) into a published MANIFEST.md for a
# run that IS a circuit simulation. Confirmed in the field on the Boss OD-3 ngspice-deck release.
grep -q 'is_capture = cfg.get("backend") == "capture"' "$HERE/release_run.sh" \
  && echo "  ok   is_capture keys off backend==capture, not schx is None" \
  || { echo "  FAIL release_run.sh reverted to schx-null-means-capture"; fails=$((fails+1)); }
grep -q 'is_deck = (not is_capture) and cfg.get("schx") is None' "$HERE/release_run.sh" \
  && echo "  ok   is_deck exists as its own category" \
  || { echo "  FAIL release_run.sh lost the deck-sourced category"; fails=$((fails+1)); }
grep -q 'elif \[ "\$IS_DECK" -eq 1 \]' "$HERE/release_run.sh" \
  && echo "  ok   PROVENANCE_LINE has a real IS_DECK branch, not just IS_CAPTURE" \
  || { echo "  FAIL PROVENANCE_LINE has no IS_DECK branch (deck runs fall into capture wording)"; fails=$((fails+1)); }
# The old, plain "IS_CAPTURE -eq 0" gate must be gone from every schx-bundling site -- if it
# comes back, a deck run (IS_CAPTURE=0, IS_DECK=1) would try to `cp "$SCHX"` a file that does
# not exist for this run.
if grep -qE '\[ "\$IS_CAPTURE" -eq 0 \]' "$HERE/release_run.sh"; then
  echo "  FAIL a bare \"\$IS_CAPTURE -eq 0\" gate is back -- deck runs would hit schx-only code"
  fails=$((fails+1))
else
  echo "  ok   schx-bundling sites gate on HAS_SCHX, not the old bare IS_CAPTURE check"
fi

echo; [ "$fails" -eq 0 ] && echo "PASS" || { echo "$fails FAILURE(S)"; exit 1; }
