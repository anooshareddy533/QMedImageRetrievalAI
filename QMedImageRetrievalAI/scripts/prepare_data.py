#!/usr/bin/env python3
"""Build the authoritative patient-split CSV manifest from official dataset folders."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import pandas as pd
from qmed.data.manifest import (prepare_deeplesion, prepare_brats, prepare_chestxray,
                                 patient_level_split, validate_manifest)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deeplesion-root", help="Contains DL_info.csv and extracted PNGs")
    p.add_argument("--deeplesion-csv")
    p.add_argument("--brats-root", help="BraTS20_Training_* folders with four NIfTI sequences and segmentation")
    p.add_argument("--brats-max-slices", type=int, default=50)
    p.add_argument("--chest-root", help="Data_Entry_2017*.csv and images_N directories")
    p.add_argument("--chest-csv")
    p.add_argument("--input-manifest", help="Already normalized manifest CSV; just validate and split")
    p.add_argument("--output", default="data/manifest.csv")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    pieces = []
    if args.input_manifest:
        pieces.append(pd.read_csv(args.input_manifest))
    if args.deeplesion_root:
        pieces.append(prepare_deeplesion(args.deeplesion_root, args.deeplesion_csv))
    if args.brats_root:
        pieces.append(prepare_brats(args.brats_root, args.brats_max_slices, args.seed))
    if args.chest_root:
        pieces.append(prepare_chestxray(args.chest_root, args.chest_csv))
    if not pieces:
        p.error("Provide at least one --*-root or --input-manifest")
    combined = validate_manifest(pd.concat(pieces, ignore_index=True), validate_paths=True)
    combined = patient_level_split(combined, seed=args.seed)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(dest, index=False)
    print(f"Saved {len(combined):,} records -> {dest.resolve()}")
    print(pd.crosstab([combined.modality, combined.uccs], combined.split).to_string())
    print("All patient groups are assigned to exactly one split (70/15/15 target proportions)")

if __name__ == "__main__":
    main()
