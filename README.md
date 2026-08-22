# AIPRE — AI-based Physical-layer REcognition of G.975.1 FEC Codes

Blind recognition of Forward Error Correction (FEC) codes in coherent optical communication systems using deep learning. Classifies 9 G.975.1 FEC code families from received signal features under varying OSNR and nonlinear interference (NLI) conditions.

## Repository Structure

```
.
├── scripts/                  # Main training and classification scripts
│   ├── AIRP.py              #   v1 — Baseline temporal algebraic features
│   ├── AIRP_v2.py           #   v2 — Enhanced feature engineering
│   ├── AIRP_v3.py           #   v3 — Hybrid features (HOS + syndrome + 2D CNN)
│   └── AIRP_v4.py           #   v4 — Spectral PSD → 1D-CNN (recommended)
│
├── dataset_scripts/          # Dataset generation and fiber simulation
│   ├── dataset_g975_fast.py  #   Fast SSFM dataset generator
│   ├── dataset_g975_1.py     #   Full G.975.1 simulation pipeline
│   ├── dataset_g975_1_fixed.py  # Corrected dataset generator
│   ├── dataset_g975.1.py     #   Alternative generator
│   └── dataset_g975_bitexact_sovereign_final.h5  # Pre-generated dataset (HDF5)
│
├── features/                 # Cached feature arrays (NumPy .npy)
│   ├── features_v3_cache_phys.npy
│   ├── features_v3_cache_other.npy
│   ├── features_v4_cache_alg.npy
│   ├── acf_v4_cache_acf.npy
│   └── psd_v4_cache_1024_32dB.npy
│
├── models/                   # Pre-trained model weights
│   ├── best_airp_v4.h5       #   v4 PSD-based model (recommended)
│   ├── best_airp_v4_spectral.h5
│   ├── best_airp_v3.h5
│   └── best_airp_v2.keras
│
├── evaluation/               # Visualization and diagnostic tools
│   ├── tsne_diagnostic.py    #   t-SNE embedding visualization
│   ├── eye_diagram_only.py   #   Eye diagram plotting
│   ├── forensic_rescue.py    #   Sovereign RRC + syndrome feature rescue
│   └── lowpower_test.py      #   Low launch-power stress test
│
├── requirements.txt
└── README.md
```

## Class Definitions (9 G.975.1 FEC Codes)

| Class | Code | Type |
|-------|------|------|
| 0 | RS(255,239) | Reed-Solomon |
| 1 | BCH(1023,971) | BCH |
| 2 | BCH(1023,951) | BCH |
| 3 | Staircase(36636,32768) | Staircase |
| 4 | LDPC(32640,30592) | LDPC |
| 5 | LDPC(32640,29120) | LDPC |
| 6 | Concatenated RS+BCH | Concatenated |
| 7 | Concatenated BCH+LDPC | Concatenated |
| 8 | Concatenated RS+LDPC | Concatenated |

## Dataset Generation

The dataset simulates coherent optical transmission with Split-Step Fourier Method (SSFM) fiber propagation, covering:
- **OSNR levels**: 20, 24, 28, 32 dB
- **Launch power**: 3 dBm (1.0 mW) default
- **Fiber**: Standard single-mode fiber with chromatic dispersion, PMD, and Kerr nonlinearity
- **DSP**: Chromatic Dispersion Compensation (CDC), matched filtering

To regenerate the dataset from scratch:

```bash
cd dataset_scripts
python dataset_g975_1_fixed.py
```

This produces an HDF5 file with signal segments and class labels. The pre-generated dataset `dataset_g975_bitexact_sovereign_final.h5` is included for direct use.

## Training (AIRP v4 — Recommended)

The v4 model uses Power Spectral Density (PSD) features fed into a 1D-CNN:

```bash
cd scripts
python AIRP_v4.py
```

**Key parameters** (in `AIRP_v4.py`):
- `PSD_DIM = 1024` — PSD input dimension
- `BATCH_SIZE = 64`
- `N_CLASSES = 9`
- `OSNR_LIST = [20, 24, 28, 32]` — dB

The script will:
1. Load the dataset from `dataset_scripts/dataset_g975_bitexact_sovereign_final.h5`
2. Compute PSD features (cached in `features/`)
3. Train a 1D-CNN classifier
4. Save best weights to `models/best_airp_v4.h5`
5. Print classification report and confusion matrix

## Evaluation

```bash
cd evaluation
python tsne_diagnostic.py       # t-SNE feature space visualization
python eye_diagram_only.py      # Eye diagram for pulse-shaping audit
python lowpower_test.py         # Stress test at -3 dBm launch power
python forensic_rescue.py       # RRC + hybrid feature rescue pipeline
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+, CUDA 11.8+ (for GPU acceleration with TensorFlow).

## License

Research use only.
