#!/usr/bin/env bash
# Build NIST NBIS 5.0.0 (mindtct, bozorth3, nfiq) into tools/nbis/bin.
# Optional: without NBIS the project reuses the NBIS outputs saved in results/.
#
# Needs a C compiler, make and cmake:
#   macOS: xcode-select --install && brew install cmake
#   Linux: sudo apt install build-essential cmake
# Usage (from the project root): bash scripts/build_nbis.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/tools/nbis_src"
DEST="$ROOT/tools/nbis"

rm -rf "$SRC" && mkdir -p "$SRC" "$DEST"
unzip -q "$ROOT/nbis_v5_0_0.zip" -d "$SRC"
cd "$SRC/Rel_5.0.0"

ARCH_OPT=""
[ "$(uname)" = "Linux" ] && ARCH_OPT="--64"
./setup.sh "$DEST" --without-X11 $ARCH_OPT

# NBIS is 2015-era C. Modern compilers (GCC 10+, Apple clang 15+) reject some of
# its patterns by default; keep them as warnings, as the original toolchain did.
EXTRA="-fcommon -Wno-error=implicit-function-declaration -Wno-error=implicit-int"
if cc --version 2>/dev/null | grep -qi clang; then
  EXTRA="$EXTRA -Wno-error=int-conversion -Wno-error=incompatible-function-pointer-types"
fi
sed -i.bak "s|^\(CFLAGS[[:space:]]*:=.*\)$|\1 $EXTRA|" rules.mak   # BSD + GNU sed
grep -q -- "-fcommon" rules.mak || { echo "Could not patch CFLAGS in rules.mak"; exit 1; }

# CMake 4 refuses the bundled OpenJPEG's old cmake_minimum_required; allow it.
export CMAKE_POLICY_VERSION_MINIMUM=3.5

make config
make it
make install LIBNBIS=no

for b in mindtct bozorth3 nfiq; do
  [ -x "$DEST/bin/$b" ] || { echo "Build failed: $DEST/bin/$b missing"; exit 1; }
done
rm -rf "$SRC"
echo "NBIS built: $DEST/bin (mindtct, bozorth3, nfiq). The scripts find it automatically."
