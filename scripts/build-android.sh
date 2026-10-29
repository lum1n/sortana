#!/usr/bin/env bash
# Builds the Sortana Android library in kotlin/: sortana-ffi for arm64-v8a,
# armeabi-v7a and x86_64 into kotlin/src/main/jniLibs (with cargo-ndk), the
# generated Kotlin bindings into kotlin/src/main/kotlin, then the AAR with
# Gradle unless --no-gradle is given.
#
#   scripts/build-android.sh [--no-gradle]
#
# Needs cargo-ndk (`cargo install cargo-ndk`) and an NDK: ANDROID_NDK_HOME, or
# the newest one under $ANDROID_HOME/ndk (default ~/Android/Sdk).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/kotlin"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Android/Sdk}"
if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
  ANDROID_NDK_HOME="$(ls -d "$ANDROID_HOME"/ndk/* 2>/dev/null | sort -V | tail -1)"
  [[ -n "$ANDROID_NDK_HOME" ]] || { echo "no NDK found; set ANDROID_NDK_HOME" >&2; exit 1; }
  export ANDROID_NDK_HOME
fi
command -v cargo-ndk >/dev/null || { echo "needs cargo-ndk: cargo install cargo-ndk" >&2; exit 1; }

SRC="$PKG/src/main/kotlin"
rm -rf "$SRC/io"
"$ROOT/scripts/kotlin-bindings.sh" "$SRC" >/dev/null

cd "$ROOT"
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android 2>/dev/null || true
rm -rf "$PKG/src/main/jniLibs"
cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 --platform 26 -o "$PKG/src/main/jniLibs" \
  build -q -p sortana-ffi --release

if [[ "${1:-}" != "--no-gradle" ]]; then
  echo "sdk.dir=$ANDROID_HOME" > "$PKG/local.properties"
  (cd "$PKG" && gradle -q assembleRelease)
  echo "Android library → kotlin/build/outputs/aar/sortana-release.aar"
else
  echo "Android sources → kotlin/src/main (jniLibs + Kotlin bindings)"
fi
