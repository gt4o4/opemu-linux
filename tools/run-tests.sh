#!/usr/bin/env bash
# Build and run the tests — the ONE list of what "the tests pass" means,
# used both by the Nix build (common/pkgs/sse42-emu/default.nix) and by a
# developer loop on a machine with the silicon (lix-ory).
#
#   ZYDIS=<zydis prefix> ZYCORE=<zycore prefix> tools/run-tests.sh build|run|all [outdir]
#
# Zydis is linked statically from $ZYDIS/lib/libZydis.a + $ZYCORE/lib/libZycore.a.
# test-emu, test-vec and test-gpr need real SSE4.2/AES-NI/PCLMULQDQ and exit 2
# without them; test-ref runs on any x86-64.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
cmd=${1:-all}
out=${2:-$here/build}
: "${CC:=gcc}"
cflags=(-O2 -Wall -Wextra -Werror -std=gnu11)

build() {
  : "${ZYDIS:?set ZYDIS to the zydis prefix}" "${ZYCORE:?set ZYCORE to the zycore prefix}"
  local inc=(-I"$ZYDIS/include" -I"$ZYCORE/include")
  local libs=("$ZYDIS/lib/libZydis.a" "$ZYCORE/lib/libZycore.a")
  mkdir -p "$out"
  cd "$here"
  # the preload library
  $CC "${cflags[@]}" -fPIC -fvisibility=hidden -shared "${inc[@]}" -o "$out/libsse42emu.so" \
      preload/sse42emu.c core/emu-core.c "${libs[@]}" -ldl
  # the tests that drive the preload's decoder (Zydis) and the core
  for t in test-emu test-vec test-gpr; do
    $CC "${cflags[@]}" "${inc[@]}" -o "$out/$t" tests/$t.c preload/sse42emu.c core/emu-core.c "${libs[@]}" -ldl
  done
  # the core against the scalar reference: no Zydis, no silicon
  $CC "${cflags[@]}" -o "$out/test-ref" tests/test-ref.c tests/pcmpstr-ref.c core/emu-core.c
}

run() {
  local status=0 t
  for t in test-emu test-vec test-gpr test-ref; do
    echo "== $t"
    if ! "$out/$t"; then echo "== $t FAILED"; status=1; fi
  done
  return $status
}

case $cmd in
  build) build ;;
  run)   run ;;
  all)   build && run ;;
  *) echo "usage: $0 build|run|all [outdir]" >&2; exit 2 ;;
esac
