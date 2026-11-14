#!/bin/bash
# Downloads HAM10000 (Harvard Dataverse, doi:10.7910/DVN/DBW86T, CC BY-NC 4.0) and unpacks it into
# data/ham10000/{images,masks}/ as expected by the scripts. ~2.8 GB.
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/ham10000
mkdir -p "$D/raw" "$D/images" "$D/masks"
base="https://dataverse.harvard.edu/api/access/datafile"
for spec in "3172585:HAM10000_images_part_1.zip" "3172584:HAM10000_images_part_2.zip" \
            "3838943:HAM10000_segmentations_lesion_tschandl.zip" "4338392:HAM10000_metadata.tab"; do
  id="${spec%%:*}"; name="${spec#*:}"
  echo "==> $name"
  curl -L --retry 3 --retry-delay 5 -C - -# -o "$D/raw/$name" "$base/$id"
done
unzip -q -o "$D/raw/HAM10000_images_part_1.zip" -d "$D/images"
unzip -q -o "$D/raw/HAM10000_images_part_2.zip" -d "$D/images"
unzip -q -o "$D/raw/HAM10000_segmentations_lesion_tschandl.zip" -d "$D/masks"
mv "$D"/masks/HAM10000_segmentations_lesion_tschandl/*.png "$D/masks/"
rm -rf "$D/masks/HAM10000_segmentations_lesion_tschandl" "$D/masks/__MACOSX"
echo "images: $(ls "$D/images" | wc -l)   masks: $(ls "$D/masks" | wc -l)   (expected 10015 / 10015)"
