import os, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

DATA_PATH = "dataset_g975_bitexact_sovereign_final.h5"
N_CLASSES = 9
N_PER_CLASS = 1000
OSNR_LIST = [20, 24, 28, 32]
N_SAMPLES_PER_OSNR = 3000

rng = np.random.default_rng(42)

print("Opening dataset...")
f = h5py.File(DATA_PATH, 'r')
X = f['X']
y = f['y']
N_TOTAL = X.shape[0]

def get_indices_for_class(c):
    """Return indices for all samples of class c across all OSNR levels."""
    idx = []
    for osnr_idx in range(len(OSNR_LIST)):
        start = osnr_idx * N_CLASSES * N_SAMPLES_PER_OSNR + c * N_SAMPLES_PER_OSNR
        end = start + N_SAMPLES_PER_OSNR
        idx.extend(range(start, end))
    return np.array(idx)

def get_indices_for_class_osnr(c, osnr_idx):
    """Return indices for class c at a specific OSNR level."""
    start = osnr_idx * N_CLASSES * N_SAMPLES_PER_OSNR + c * N_SAMPLES_PER_OSNR
    return np.arange(start, start + N_SAMPLES_PER_OSNR)

print("Sampling 1000 per class for t-SNE...")
sample_indices = []
sample_labels = []
for c in range(N_CLASSES):
    all_idx = get_indices_for_class(c)
    chosen = rng.choice(all_idx, N_PER_CLASS, replace=False)
    sample_indices.extend(chosen)
    sample_labels.extend([c] * N_PER_CLASS)

sample_indices = np.array(sample_indices)
sample_labels = np.array(sample_labels)

sort_order = np.argsort(sample_indices)
sample_indices = sample_indices[sort_order]
sample_labels = sample_labels[sort_order]

print(f"Total samples for t-SNE: {len(sample_indices)}")

BATCH = 500
all_feats = []
print("Computing Mag + DiffPhase invariants...")
for start in range(0, len(sample_indices), BATCH):
    end = min(start + BATCH, len(sample_indices))
    batch_idx = sample_indices[start:end]
    batch_iq = X[batch_idx]

    I = batch_iq[:, :, 0]
    Q = batch_iq[:, :, 1]

    mag = np.sqrt(I.astype(np.float64)**2 + Q.astype(np.float64)**2)
    phase = np.arctan2(Q.astype(np.float64), I.astype(np.float64))

    diff_phase = np.diff(np.unwrap(phase, axis=1), axis=1)

    feats = np.concatenate([mag, diff_phase], axis=1)
    all_feats.append(feats)
    print(f"  Batch {start}-{end} done")

feats = np.concatenate(all_feats, axis=0).astype(np.float32)
print(f"Feature matrix shape: {feats.shape}")

print("Running PCA (50 components)...")
pca = PCA(n_components=50, random_state=42)
feats_pca = pca.fit_transform(feats)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}")

print("Running t-SNE (this may take a while)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=2000, verbose=1)
feats_tsne = tsne.fit_transform(feats_pca)
print("t-SNE done!")

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
colors = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))
for c in range(N_CLASSES):
    mask = sample_labels == c
    ax.scatter(feats_tsne[mask, 0], feats_tsne[mask, 1],
               c=[colors[c]], label=f'Class {c}', alpha=0.6, s=5)

ax.set_title('t-SNE of IQ → [Mag, ΔPhase] Invariants (1000/class)', fontsize=14)
ax.legend(markerscale=5, fontsize=10)
plt.tight_layout()
plt.savefig('tsne_mag_dphase_all_classes.png', dpi=150)
print("Saved: tsne_mag_dphase_all_classes.png")

print("\n--- Label Audit: Class 1 vs Class 8 at 32dB OSNR ---")
osnr_idx = 3
idx_c1 = get_indices_for_class_osnr(1, osnr_idx)
idx_c8 = get_indices_for_class_osnr(8, osnr_idx)

audit_indices = np.concatenate([idx_c1, idx_c8])
audit_labels = np.concatenate([np.full(3000, 1), np.full(3000, 8)])
sort_order_audit = np.argsort(audit_indices)
audit_indices = audit_indices[sort_order_audit]
audit_labels = audit_labels[sort_order_audit]

audit_all_feats = []
BATCH_A = 500
for start in range(0, len(audit_indices), BATCH_A):
    end = min(start + BATCH_A, len(audit_indices))
    batch_idx = audit_indices[start:end]
    batch_iq = X[batch_idx]

    I = batch_iq[:, :, 0]
    Q = batch_iq[:, :, 1]

    mag = np.sqrt(I.astype(np.float64)**2 + Q.astype(np.float64)**2)
    phase = np.arctan2(Q.astype(np.float64), I.astype(np.float64))
    diff_phase = np.diff(np.unwrap(phase, axis=1), axis=1)

    feats_b = np.concatenate([mag, diff_phase], axis=1)
    audit_all_feats.append(feats_b)

audit_feats = np.concatenate(audit_all_feats, axis=0).astype(np.float32)

audit_pca = PCA(n_components=50, random_state=42)
audit_feats_pca = audit_pca.fit_transform(audit_feats)

audit_tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=2000, verbose=1)
audit_feats_tsne = audit_tsne.fit_transform(audit_feats_pca)

fig2, ax2 = plt.subplots(1, 1, figsize=(12, 9))
mask1 = audit_labels == 1
mask8 = audit_labels == 8
ax2.scatter(audit_feats_tsne[mask1, 0], audit_feats_tsne[mask1, 1],
            c='blue', label='Class 1', alpha=0.4, s=3)
ax2.scatter(audit_feats_tsne[mask8, 0], audit_feats_tsne[mask8, 1],
            c='red', label='Class 8', alpha=0.4, s=3)
ax2.set_title('Label Audit: Class 1 vs Class 8 at 32dB OSNR', fontsize=14)
ax2.legend(markerscale=10, fontsize=12)
plt.tight_layout()
plt.savefig('tsne_label_audit_c1_vs_c8_32dB.png', dpi=150)
print("Saved: tsne_label_audit_c1_vs_c8_32dB.png")

f.close()
print("\nDone! Both plots saved.")
