#!/usr/bin/env python3
"""Eye Diagram for Pulse Shaping Audit (Class 0, 32dB OSNR)"""
import os, sys, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import dataset_g975_1_fixed as dataset_mod
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SPS = dataset_mod.SPS
SAMPLING_RATE = dataset_mod.SAMPLING_RATE
AI_WINDOW = dataset_mod.AI_WINDOW

np.random.seed(42)

# Generate Class 0 RS(255,239) signal
import galois
GF256 = galois.GF(2**8, irreducible_poly=0x11d)
rs = galois.ReedSolomon(255, 239, field=GF256, c=0)
raw_bits = np.random.randint(0, 2, 1000000, dtype=np.uint8)
syms_array = dataset_mod.bits_to_uints_gf_exact(raw_bits[:239*8], 8)
bits = np.array([[(s >> j) & 1 for j in range(7, -1, -1)]
                 for s in rs.encode(syms_array)]).flatten()
segment = np.resize(bits, AI_WINDOW)
complex_sig = (segment.astype(np.complex128)*2 - 1) * np.sqrt(dataset_mod.P_LAUNCH_W)

print(f"  complex_sig: {complex_sig.shape}, dtype={complex_sig.dtype}")

# TX pulse shaping
oversampled = dataset_mod._gaussian_filter_fft(complex_sig, SPS)
n_symbols = len(complex_sig)
n = len(oversampled)
print(f"  after pulse shape: {n} samples, {n_symbols} symbols")

dt = 1.0 / SAMPLING_RATE
frequencies = np.fft.fftfreq(n, d=dt)
beta2 = -2.1e-26
alpha = 4.6e-5
distance_m = 80000.0
num_steps = 40
h_step = distance_m / num_steps
h_eff = (1.0 - np.exp(-alpha * h_step)) / alpha
channel_spacing_hz = 100e9
walkoff_delay_sec = np.abs(beta2 * 2.0 * np.pi * channel_spacing_hz * h_step)
walkoff_shift_per_step = walkoff_delay_sec * SAMPLING_RATE
dispersion_half_step = np.exp(-1j * 0.5 * beta2 * (2.0 * np.pi * frequencies)**2 * (h_step / 2.0))

# XPM channels — same length as signal
xpm_sum = np.zeros(n, dtype=np.float64)
for _ in range(4):
    bits_ch = np.random.randint(0, 4, n_symbols, dtype=np.uint8)
    sig_ch = (np.cos(bits_ch * np.pi / 2) + 1j * np.sin(bits_ch * np.pi / 2)) * np.sqrt(dataset_mod.P_LAUNCH_W)
    interf = dataset_mod._gaussian_filter_fft(sig_ch, SPS)
    xpm_sum += np.abs(interf)**2

print(f"  xpm_sum: {xpm_sum.shape}")

# SSFM
u1 = oversampled.copy()
for step in range(num_steps):
    u1 = np.fft.ifft(np.fft.fft(u1) * dispersion_half_step)
    power_spm = np.abs(u1)**2
    shift_samples = int(step * walkoff_shift_per_step)
    power_xpm = dataset_mod.roll_array_1d(xpm_sum, shift_samples) * np.exp(-alpha * step * h_step)
    phi_nl = dataset_mod.GAMMA * h_eff * (power_spm + 2.0 * power_xpm)
    u1 = u1 * (np.cos(phi_nl) + 1j * np.sin(phi_nl))
    u1 = np.fft.ifft(np.fft.fft(u1) * dispersion_half_step)
    u1 *= np.exp(-0.5 * alpha * h_step)

# ASE at 32dB
ref_bw = 12.5e9
N0 = dataset_mod.P_LAUNCH_W / (2.0 * 10**(32 / 10.0) * ref_bw)
noise_power = N0 * SAMPLING_RATE
noise_std = np.sqrt(noise_power / 2.0)
noise = (np.random.normal(0, noise_std, n) + 1j * np.random.normal(0, noise_std, n))
received = u1 + noise

# Matched filter full-rate
mf_full = dataset_mod._gaussian_filter_fft(received, SPS)
I_full = np.real(mf_full)
samples_per_symbol = SPS
total_symbols = len(I_full) // samples_per_symbol
print(f"  MF output: {len(I_full)} samples, {total_symbols} symbols")

# Eye diagram: overlay 3-symbol segments
fig, ax = plt.subplots(figsize=(12, 6))
symbols_to_show = 3
n_segments = 500
segment_offset = 0
for _ in range(n_segments):
    sym_start = np.random.randint(0, total_symbols - symbols_to_show)
    idx_start = sym_start * samples_per_symbol + segment_offset
    idx_end = idx_start + symbols_to_show * samples_per_symbol
    seg = I_full[idx_start:idx_end]
    t_rel = np.arange(len(seg)) - segment_offset
    ax.plot(t_rel, seg, 'b-', alpha=0.12, linewidth=0.5)

ax.set_xlabel('Sample offset', fontsize=12)
ax.set_ylabel('Amplitude (I)', fontsize=12)
ax.set_title(f'Eye Diagram — Class 0 (RS 239/255), 32dB OSNR, 3dBm\n'
             f'Gaussian σ={SPS/2:.0f}, SPS={SPS}, {n_segments} traces',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
for s in range(-samples_per_symbol, symbols_to_show * samples_per_symbol + 1, samples_per_symbol):
    ax.axvline(s, color='red', linestyle='--', alpha=0.2)
ax.axvline(SPS//2, color='green', linestyle='-', alpha=0.8, label=f'Current sampling @ idx {SPS//2}')
ax.legend()
plt.tight_layout()
plt.savefig("eye_diagram_class0_32dB.png", dpi=300, bbox_inches='tight')
print("\neye_diagram_class0_32dB.png saved")

# Eye statistics
n_avg = min(5000, total_symbols)
symbol_wfs = np.zeros((n_avg, samples_per_symbol))
for i in range(n_avg):
    symbol_wfs[i] = I_full[i * samples_per_symbol: (i+1) * samples_per_symbol]
avg_wf = np.mean(symbol_wfs, axis=0)
std_wf = np.std(symbol_wfs, axis=0)

print(f"\n  Symbol-spaced statistics (SPS={SPS}):")
for i in range(samples_per_symbol):
    lbl = " <-- CURRENT" if i == SPS//2 else ""
    print(f"    Sample {i}: mean={avg_wf[i]:.6f}, std={std_wf[i]:.6f}{lbl}")
best_idx = np.argmin(std_wf)
print(f"\n  Current: SPS//2={SPS//2} (std={std_wf[SPS//2]:.6f})")
print(f"  Best:    idx {best_idx} (std={std_wf[best_idx]:.6f})")
print(f"  Eye opening ratio: {np.mean(avg_wf[best_idx+1::2]) / np.mean(avg_wf[best_idx::2]):.3f}")
