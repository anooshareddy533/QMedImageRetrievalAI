# Dataset layouts

Do **not** place patient-identifying data in a public repository. Official datasets are not included in this ZIP.

## DeepLesion

Expected: a root containing `DL_info.csv` plus extracted `*.png` images (paths may be nested).
The adapter reads RECIST long-axis measurements (mm if available, otherwise computes them from endpoint coordinates and pixel spacing). Only records with valid diameter measurements receive UCCS C1/C2 labels. Annotations are not used as query-time ROIs unless annotation-assisted mode is explicitly enabled.

## BraTS 2020

Expected: folders such as `BraTS20_Training_001/` with:

- `BraTS20_Training_001_t1.nii.gz`
- `BraTS20_Training_001_t1ce.nii.gz`
- `BraTS20_Training_001_t2.nii.gz`
- `BraTS20_Training_001_flair.nii.gz`
- `BraTS20_Training_001_seg.nii.gz`

Subject groups are split before sampling 2D slices. The segmentation is used to calculate UCCS categories for training/evaluation. It is not given to the image-only query encoder.

## NIH ChestX-ray14

Expected: `Data_Entry_2017*.csv` and extracted `images_*/images/*.png` (or equivalent nested images).
The adapter reads `Image Index`, `Finding Labels` and `Patient ID` and maps the multi-label findings with the priority specified in manuscript Table 3A.

## Unified manifest

Required columns: `image_id,patient_id,modality,uccs,image_path,split`.
The `split` column is created by `scripts/prepare_data.py` using patient group splitting.
`uccs`: **0=C1**, **1=C2**, **2=C3**, **3=C4**, **4=C5**.
`modality`: one of `CT`, `MRI`, `XRAY`.
Optional MRI columns: `seq_t1,seq_t1ce,seq_t2,seq_flair,slice_idx`.
Optional ROI column: `roi_mask_path` (only applied when `roi_policy=supplied`).
All paths should be absolute or relative to the location of the CSV manifest.

To use an existing manifest:

```bash
python scripts/prepare_data.py --input-manifest raw_manifest.csv --output data/manifest.csv
```

If the input already contains a split column, `prepare_data.py` deliberately recomputes it to enforce group disjointness. Keep a copy of your original split in your experimental records.
