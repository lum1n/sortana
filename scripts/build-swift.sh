#!/usr/bin/env bash
# Builds the Sortana Swift package in swift/: sortana-ffi for iOS (device +
# simulator) and macOS as SortanaFFI.xcframework, plus the generated
# bindings in swift/Sources/Sortana. Needs macOS and Xcode.
#
# Apps that already ship a Rust static library should not use the
# xcframework (two Rust static libraries in one binary duplicate std).
# Depend on crates/sortana-ffi from that library and generate the bindings
# with scripts/swift-bindings.sh instead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/swift"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
[[ "$(uname)" == "Darwin" ]] || { echo "build-swift.sh needs macOS and Xcode" >&2; exit 1; }

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
"$ROOT/scripts/swift-bindings.sh" "$OUT" >/dev/null
cp "$OUT/sortana_ffi.swift" "$PKG/Sources/Sortana/sortana_ffi.swift"

cd "$ROOT"
TARGETS=(aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios aarch64-apple-darwin x86_64-apple-darwin)
rustup target add "${TARGETS[@]}" 2>/dev/null || true
for t in "${TARGETS[@]}"; do
  IPHONEOS_DEPLOYMENT_TARGET=18.0 MACOSX_DEPLOYMENT_TARGET=14.0 \
    cargo build -q -p sortana-ffi --release --target "$t"
done

STAGE="$OUT/stage"
mkdir -p "$STAGE/headers/sortana_ffiFFI" "$STAGE/sim" "$STAGE/mac"
cp "$OUT/sortana_ffiFFI.h" "$STAGE/headers/sortana_ffiFFI/"
cp "$OUT/sortana_ffiFFI.modulemap" "$STAGE/headers/sortana_ffiFFI/module.modulemap"
lib() { echo "$CARGO_TARGET_DIR/$1/release/libsortana_ffi.a"; }
lipo -create "$(lib aarch64-apple-ios-sim)" "$(lib x86_64-apple-ios)" -output "$STAGE/sim/libsortana_ffi.a"
lipo -create "$(lib aarch64-apple-darwin)" "$(lib x86_64-apple-darwin)" -output "$STAGE/mac/libsortana_ffi.a"

rm -rf "$PKG/SortanaFFI.xcframework"
xcodebuild -create-xcframework \
  -library "$(lib aarch64-apple-ios)" -headers "$STAGE/headers" \
  -library "$STAGE/sim/libsortana_ffi.a" -headers "$STAGE/headers" \
  -library "$STAGE/mac/libsortana_ffi.a" -headers "$STAGE/headers" \
  -output "$PKG/SortanaFFI.xcframework"
echo "Swift package → swift/ (SortanaFFI.xcframework + Sources/Sortana/sortana_ffi.swift)"
