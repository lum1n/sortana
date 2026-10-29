#!/usr/bin/env bash
# Generates sortana-ffi's Kotlin bindings (package io.github.lum1n.sortana,
# loaded through JNA) into OUT_DIR from a host build. Runs anywhere cargo
# does; scripts/build-android.sh uses it for the Android library in kotlin/.
#
#   scripts/kotlin-bindings.sh OUT_DIR
#
# Honors CARGO_TARGET_DIR. Prints the path of the built host library, which
# JVM programs can load with -Djna.library.path=<its directory>.
set -euo pipefail
OUT="${1:?usage: kotlin-bindings.sh OUT_DIR}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
cd "$ROOT"
cargo build -q -p sortana-ffi --release
LIB="$TARGET/release/libsortana_ffi.so"
[[ -f "$TARGET/release/libsortana_ffi.dylib" ]] && LIB="$TARGET/release/libsortana_ffi.dylib"
mkdir -p "$OUT"
cargo run -q -p sortana-ffi --features bindgen --bin uniffi-bindgen -- \
  generate --library "$LIB" --language kotlin --no-format --out-dir "$OUT" >&2
echo "$LIB"
