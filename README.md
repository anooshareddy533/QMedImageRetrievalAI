[![DOI](https://zenodo.org/badge/1467415386.svg)](https://doi.org/10.5281/zenodo.24438503)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/anooshareddy533/QMedImageRetrievalAI)

# QMedImageRetrievalAI

## A Quantum-Enhanced Deep Learning Framework for Content-Based Retrieval of Multi-Modal Medical Images

QMedImageRetrievalAI is a hybrid quantum-classical deep learning framework for content-based retrieval of clinically analogous CT, MRI, and chest X-ray images. It integrates multi-scale CNN features, Transformer-based feature extraction, an eight-qubit Variational Quantum Circuit (VQC), category-aware cross-modal alignment, attention-based fusion, and FAISS similarity indexing.

The framework supports image preprocessing, model training, multimodal feature learning, Top-K image retrieval, classical baseline comparisons, component ablation studies, statistical evaluation, and visual interpretation.

## Key Features

- Independent dataset adapters for DeepLesion, BraTS 2020, and NIH ChestX-ray14.
- Patient-level 70:15:15 training, validation, and testing partitioning.
- Unified Clinical Category Space (UCCS) with five categories (C1–C5).
- Grayscale image preprocessing and min-max normalization.
- Optional annotation-assisted region-of-interest enhancement.
- Four-sequence MRI processing with attention-based feature aggregation.
- ResNet50-based multi-scale CNN and eight-head Transformer architecture.
- Differentiable eight-qubit, four-layer Variational Quantum Circuit.
- Quantum data re-uploading using RY and RZ rotations.
- Learnable RX, RY, and RZ quantum rotations with ring CNOT entanglement.
- 96 trainable quantum circuit parameters.
- 36 quantum expectation-value readouts (8 Z and 28 ZZ).
- 64-dimensional classical projection and residual representation.
- 128-dimensional normalized retrieval embedding.
- PyTorch complex statevector simulation with optional PennyLane backend.
- Separate CT, MRI, and X-ray alignment heads.
- Alignment, contrastive, batch-hard triplet, and centroid-regularization losses.
- FAISS IVF-HNSW/Flat indexing with exact NumPy cosine-search fallback.
- Image-based Top-K retrieval with optional target-modality filtering.
- Classical ResNet50, DenseNet121, and ViT baseline comparisons.
- Optional CLIP, BiomedCLIP, and DINOv2 comparisons.
- Component and quantum circuit ablation experiments.
- Grad-CAM visualization and retrieval similarity analysis.
- Five-seed experimental evaluation and statistical analysis.

## System Requirements

| Component | Requirement |
|---|---|
| Python | 3.10–3.12 recommended |
| Deep Learning | PyTorch |
| Operating System | Windows or Linux |
| GPU | CUDA-compatible GPU recommended |
| Quantum Simulation | PyTorch statevector simulator |
| Image Indexing | FAISS |
| Data Processing | NumPy, Pandas, Pillow |
| Visualization | Matplotlib |
| Testing | Pytest |

Python 3.13 can also be used for the included CPU verification, subject to dependency compatibility.

## Project Architecture

The framework consists of the following major components:

1. **Dataset Management:** Independent CT, MRI, and X-ray dataset loading and patient-level partitioning.

2. **Image Preprocessing:** Grayscale conversion, normalization, optional ROI enhancement, and image quality assessment.

3. **Multi-Scale Feature Extraction:** ResNet50 CNN and Transformer-based spatial feature extraction.

4. **Quantum Feature Learning:** Eight-qubit Variational Quantum Circuit with trainable rotations and entanglement operations.

5. **Cross-Modal Alignment:** Separate modality-specific alignment heads for CT, MRI, and X-ray representations.

6. **Attention-Based Fusion:** Integration of classical and quantum representations with four-sequence MRI attention.

7. **Similarity Indexing:** FAISS-based vector indexing and cosine similarity search.

8. **Image Retrieval:** Ranked Top-K retrieval results across supported imaging modalities.

9. **Performance Evaluation:** Retrieval metrics, baseline comparisons, ablation studies, and statistical analysis.

## Dataset Sources

The framework supports three publicly available medical imaging datasets.

| Dataset | Modality | Dataset Source | UCCS Categories |
|---|---|---|---|
| DeepLesion | CT | [Download DeepLesion](https://nihcc.app.box.com/v/DeepLesion) | C1, C2 |
| BraTS 2020 | MRI | [Download BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) | C1, C3, C5 |
| NIH ChestX-ray14 | Chest X-ray | [Download ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) | C1, C2, C3, C4, C5 |

### UCCS Category Mapping

| Category | Description |
|---|---|
| C1 | Primary lesion or finding categories |
| C2 | Smaller lesions and nodules |
| C3 | Edema-related findings |
| C4 | Other chest X-ray findings |
| C5 | No-finding or tumour-free categories, according to modality |

The exact category mapping is modality-dependent and is defined in the dataset adapter implementation.

**Dataset preparation notes:**

- Download the datasets from their official sources.
- BraTS 2020 access requires acceptance of its data-use terms.
- DeepLesion annotations represent lesions and should not be treated as unique CT image counts.
- The BraTS adapter selects up to 50 anatomical slices per subject using deterministic class-balanced sampling.
- Dataset manifests contain image paths, modality information, category labels, and patient identifiers.

Refer to `data/README.md` for dataset organization and manifest requirements.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anooshareddy533/QMedImageRetrievalAI.git
cd QMedImageRetrievalAI
```

### 2. Create a Python Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux:**

```bash
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -e .
```

### 5. Check the Environment

```bash
python scripts/system_check.py
```

### 6. Run Tests

```bash
python -m pytest -q
```

## Optional Dependencies

For additional dataset processing, quantum backends, and indexing functionality:

```bash
pip install -e ".[medical,quantum,index]"
```

For optional quantum noise experiments and foundation-model comparisons:

```bash
pip install "qiskit>=1.0" "qiskit-aer>=0.14" "open_clip_torch>=2.26" "transformers>=4.42"
```

For GPU execution, install a PyTorch version compatible with the installed CUDA runtime before installing the remaining dependencies.

## Dataset Preparation

Prepare the normalized dataset manifest using:

```bash
python scripts/prepare_data.py \
    --deeplesion-root /data/DeepLesion \
    --brats-root /data/BraTS2020 \
    --chest-root /data/ChestXray14 \
    --output data/manifest.csv \
    --seed 0
```

If a normalized CSV manifest is already available:

```bash
python scripts/prepare_data.py --input-manifest data/manifest.csv
```

The preparation process validates image paths, assigns patient-level partitions, and reports modality, category, and split distributions.

The target partition ratio is:

- Training: 70%
- Validation: 15%
- Testing: 15%

Actual proportions may vary slightly because partitioning is performed at the patient level.

## Model Training

Start model training with the default configuration:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --manifest data/manifest.csv \
    --out outputs/full/seed_0
```

The training pipeline includes:

- Multi-scale CNN feature extraction.
- Transformer-based spatial representation learning.
- Variational quantum circuit processing.
- Cross-modal feature alignment.
- Attention-based fusion.
- Combined optimization objectives.
- Validation monitoring.
- Model checkpoint saving and resuming.

The model uses a differentiable eight-qubit statevector simulator for quantum circuit operations.

To enable pretrained ResNet50 initialization, configure:

```yaml
model:
  pretrained: true
```

A compatible pretrained model weight download or local cache is required.

### Five-Seed Experiment

Run the complete experiment using five random seeds:

```bash
python scripts/run_full_experiment.py \
    --config configs/default.yaml \
    --manifest data/manifest.csv \
    --seeds 0,1,2,3,4 \
    --out outputs/full
```

The same patient-level data partition is maintained across seeds for paired comparisons.

## Build the Retrieval Index

Generate a retrieval index from a trained checkpoint:

```bash
python scripts/build_index.py \
    --checkpoint outputs/full/seed_0/best.pt \
    --manifest data/manifest.csv \
    --split test \
    --out outputs/full/index
```

The indexing module supports:

- FAISS IVF-HNSW/Flat indexing.
- Cosine similarity using normalized embeddings.
- Automatic index configuration for smaller galleries.
- Exact NumPy cosine-search fallback when FAISS is unavailable.
- Patient-level exclusion during evaluation.

The default FAISS configuration uses IVF1024_HNSW32,Flat with an nprobe value of 16, with smaller configurations selected when appropriate.

## Medical Image Retrieval

Retrieve similar images using a query image:

```bash
python scripts/retrieve.py \
    --checkpoint outputs/full/seed_0/best.pt \
    --index-dir outputs/full/index \
    --image /path/to/new_ct.png \
    --modality CT \
    --target XRAY \
    --k 10 \
    --out outputs/query_results.json
```

Supported query modalities:

- CT
- MRI
- X-ray

The retrieval system returns ranked image results with similarity scores and image paths.

### MRI Retrieval

The framework supports four-sequence MRI inputs:

- T1
- T1ce
- T2
- FLAIR

Example:

```bash
python scripts/retrieve.py \
    --checkpoint outputs/full/seed_0/best.pt \
    --index-dir outputs/full/index \
    --modality MRI \
    --t1 /path/T1.nii.gz \
    --t1ce /path/T1ce.nii.gz \
    --t2 /path/T2.nii.gz \
    --flair /path/FLAIR.nii.gz \
    --slice 80 \
    --k 10
```

### Optional Retrieval Interface

Install Gradio:

```bash
pip install gradio
```

Launch the local retrieval interface:

```bash
python demo/app.py \
    --checkpoint outputs/full/seed_0/best.pt \
    --index-dir outputs/full/index
```

## Model Evaluation

Evaluate retrieval performance using:

```bash
python scripts/evaluate.py \
    --index-dir outputs/full/index \
    --out outputs/full/evaluation \
    --k 5 10 20 \
    --bootstrap 10000
```

### Retrieval Metrics

The framework calculates:

| Metric | Description |
|---|---|
| mAP | Mean Average Precision |
| MRR | Mean Reciprocal Rank |
| Precision@K | Relevant images among the top K results |
| Recall@K | Retrieved relevant images relative to all eligible relevant gallery images |
| F1@K | Harmonic mean of Precision@K and Recall@K |
| NDCG@K | Normalized Discounted Cumulative Gain |

Evaluation is performed at K = 5, 10, and 20.

The evaluation covers all nine within-modality and cross-modality retrieval directions.

### Evaluation Outputs

The evaluation process produces:

- `retrieval_summary.csv`
- `per_query.csv`
- `overall.json`
- `overall_per_query.csv`
- Patient-cluster bootstrap confidence intervals, when enabled.
- ANN versus exact-index performance measurements.

The evaluation implementation excludes queries without eligible relevant gallery items and reports the number of evaluated queries.

## Index Performance Benchmarking

Benchmark indexing latency and approximate nearest-neighbour performance:

```bash
python scripts/benchmark_index.py \
    --index-dir outputs/full/index \
    --out outputs/full/index_benchmark.json
```

The benchmark measures index performance against exact nearest-neighbour search.

## Ablation Studies

Run component-level ablation experiments:

```bash
python scripts/run_ablations.py \
    --config configs/default.yaml \
    --manifest data/manifest.csv \
    --variants full,no_quantum,classical_control,no_alignment,no_mri_attention,no_contrastive,no_triplet \
    --seeds 0,1,2,3,4 \
    --out outputs/ablations
```

The available ablation configurations include:

- Full hybrid model.
- Without quantum processing.
- Classical control model.
- Without cross-modal alignment.
- Without MRI attention.
- Without contrastive loss.
- Without triplet loss.

### Quantum Circuit Configuration Experiments

To evaluate different quantum circuit configurations:

```bash
python scripts/run_ablations.py \
    --config configs/default.yaml \
    --manifest data/manifest.csv \
    --quantum-grid \
    --seeds 0,1,2,3,4 \
    --out outputs/quantum_grid
```

The quantum grid supports:

**Qubit configurations:** 4, 6, 8, and 10.

**Circuit depths:** 2, 4, 6, and 8.

## Classical Baseline Comparisons

Train classical deep learning baselines:

```bash
python scripts/train_baselines.py \
    --config configs/default.yaml \
    --manifest data/manifest.csv \
    --models resnet50,densenet121,vit_b_16 \
    --pretrained \
    --out outputs/baselines
```

Supported baseline architectures:

- ResNet50
- DenseNet121
- Vision Transformer (ViT-B/16)

Optional foundation-model comparisons include:

- CLIP
- BiomedCLIP
- DINOv2

These optional comparisons require their respective packages and model weights.

The classical control configuration uses 96 trainable head parameters for comparison with the eight-qubit, four-layer VQC, excluding shared projection and readout components.

## Quantum Noise Experiments

Optional quantum readout-noise experiments can be performed using Qiskit Aer.

Install the required packages:

```bash
pip install "qiskit>=1.0" "qiskit-aer>=0.14"
```

Run the experiment:

```bash
python scripts/noise_experiment.py \
    --checkpoint outputs/full/seed_0/best.pt \
    --samples 5 \
    --shots 4096 \
    --p1 0.001 \
    --p2 0.01 \
    --out outputs/noise.json
```

This experiment evaluates changes in quantum readout measurements under configured noise conditions.

## Visualization and Explainability

Generate evaluation visualizations:

```bash
python scripts/visualize_results.py \
    --index-dir outputs/full/index \
    --ablation-summary outputs/ablations/ablation_summary.csv \
    --retrieval-summary outputs/full/evaluation/retrieval_summary.csv
```

Generate a retrieval similarity saliency visualization using Grad-CAM:

```bash
python scripts/explain_retrieval.py \
    --checkpoint outputs/full/seed_0/best.pt \
    --query /path/query.png \
    --query-modality CT \
    --retrieved /path/result.png \
    --retrieved-modality CT \
    --out outputs/gradcam.png
```

The visualization module supports:

- Retrieval similarity analysis.
- Grad-CAM-based feature visualization.
- Ablation result comparisons.
- Evaluation metric plots.
- Retrieval performance visualization.

## Project Structure

```text
QMedImageRetrievalAI/
│
├── configs/
│   ├── default.yaml
│   ├── experiments.yaml
│   └── smoke.yaml
│
├── data/
│   └── README.md
│
├── demo/
│   └── app.py
│
│
├── scripts/
│   ├── benchmark_index.py
│   ├── build_index.py
│   ├── evaluate.py
│   ├── explain_retrieval.py
│   ├── noise_experiment.py
│   ├── prepare_data.py
│   ├── quality_screen.py
│   ├── retrieve.py
│   ├── run_ablations.py
│   ├── run_full_experiment.py
│   ├── smoke_test.py
│   ├── system_check.py
│   ├── train.py
│   ├── train_baselines.py
│   └── visualize_results.py
│
├── src/
│   └── qmed/
│       ├── data/
│       ├── evaluation/
│       ├── models/
│       ├── retrieval/
│       └── training/
│
├── tests/
│
├── outputs/
│
└── requirements.txt
```

## Reproducibility and Experimental Configuration

The framework provides configurable and repeatable experimental workflows.

Important experimental settings include:

- Fixed random seeds.
- Patient-level dataset partitioning.
- Configurable image preprocessing.
- Configurable CNN and Transformer parameters.
- Quantum circuit width and depth.
- Modality-specific feature alignment.
- Configurable retrieval metrics.
- Model checkpoint management.
- Five-seed evaluation.
- Patient-level bootstrap confidence intervals.
- Holm-adjusted paired statistical comparisons.

For consistent comparisons, use the same dataset partitions and evaluation configuration across model variants.



## References and Dataset Acknowledgements

This implementation uses the following publicly available dataset resources:

- [DeepLesion Dataset – NIH](https://nihcc.app.box.com/v/DeepLesion)
- [BraTS 2020 Dataset](https://www.med.upenn.edu/cbica/brats2020/data.html)
- [NIH ChestX-ray14 Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)

Please follow the respective dataset access conditions and citation requirements when using these resources.


## Repository

**GitHub:** https://github.com/anooshareddy533/QMedImageRetrievalAI

