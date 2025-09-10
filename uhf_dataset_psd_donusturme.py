#!/usr/bin/env python3
"""
Convert UHF dataset shards (produced by uhf_sinyal_uretme_11_2.py) into
1D Welch PSD vectors with log scaling, saved as new shards.

- Reads *.pkl shards from SOURCE_DIR.
- For each sample, computes Welch PSD from complex IQ.
- Centers frequency axis (fftshift) and interpolates to TARGET_BINS.
- Applies log10 scaling in dB; optional normalization.
- Writes new shards to DEST_DIR as psd_shard_XXXX.pkl
- Creates dataset_stats.json describing the new set.

F5-ready: just set SOURCE_DIR and DEST_DIR below or use CLI args.

Example (CLI):
  python convert_to_welch_psd.py \
      --source "C:/Users/Osman/Desktop/BITES/sinyal_uhf/uhf_dataset_real_11" \
      --dest   "C:/Users/Osman/Desktop/BITES/sinyal_uhf/uhf_dataset_real_11_psd" \
      --nperseg 1024 --noverlap 512 --target-bins 1024 --norm zscore
"""
import os
import re
import json
import math
import pickle
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal as scipy_signal

# ---------------------------
# Defaults (override by CLI)
# ---------------------------
SOURCE_DIR = r"C:\\Users\\Osman\\Desktop\\BITES\\sinyal_uhf\\uhf_dataset_real_11"
DEST_DIR   = r"C:\\Users\\Osman\\Desktop\\BITES\\sinyal_uhf\\uhf_dataset_real_11_psd"
NPERSEG    = 1024
NOVERLAP   = 512
WINDOW     = "hann"          # any scipy window name or tuple
SCALING    = "density"       # 'density' or 'spectrum'
ONESIDED   = False            # keep both sides, then fftshift to center 0 Hz
TARGET_BINS= 1024             # resample PSD to this many frequency bins
LOG_EPS    = 1e-12            # epsilon before log10
NORM       = "none"           # 'none' | 'minmax' | 'zscore' (per-sample)
SAVE_IQ    = False            # if True, also keep original iq (bigger files)
SHARD_PREFIX_IN  = r"^shard_(\d{4})\.pkl$"
SHARD_PREFIX_OUT = "psd_shard_{:04d}.pkl"

# ---------------------------
# Helpers
# ---------------------------

def find_shards(src_dir: str) -> List[Tuple[int, str]]:
    pat = re.compile(SHARD_PREFIX_IN)
    out = []
    for name in os.listdir(src_dir):
        m = pat.match(name)
        if m:
            out.append((int(m.group(1)), os.path.join(src_dir, name)))
    out.sort(key=lambda x: x[0])
    return out


def compute_welch_psd(iq: np.ndarray, fs: float,
                      nperseg:int, noverlap:int, window:str, scaling:str,
                      onesided: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Return (freqs_centered_Hz, psd_linear) with length depending on Welch.
    Then we'll interpolate to TARGET_BINS outside this function.
    """
    # scipy welch expects real or complex; returns one-sided for real by default
    f, Pxx = scipy_signal.welch(iq, fs=fs, window=window, nperseg=nperseg,
                                noverlap=noverlap, return_onesided=onesided,
                                scaling=scaling, detrend=False)
    # If two-sided, center 0 Hz using fftshift for a baseband-friendly layout
    if not onesided:
        f = np.fft.fftshift(f)
        Pxx = np.fft.fftshift(Pxx)
    return f.astype(np.float32), Pxx.astype(np.float32)


def interp_to_uniform_grid(f_in: np.ndarray, p_in: np.ndarray, target_bins:int, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate PSD onto a uniform grid in [-fs/2, fs/2) with target_bins samples."""
    f_min, f_max = -fs/2.0, fs/2.0
    f_uniform = np.linspace(f_min, f_max, target_bins, endpoint=False, dtype=np.float32)
    # Ensure monotonicity for interp; f_in should be sorted but guard anyway
    sort_idx = np.argsort(f_in)
    f_sorted = f_in[sort_idx]
    p_sorted = p_in[sort_idx]
    # Extrapolate by edge values
    p_uniform = np.interp(f_uniform, f_sorted, p_sorted, left=p_sorted[0], right=p_sorted[-1]).astype(np.float32)
    return f_uniform, p_uniform


def log_and_norm(psd_linear: np.ndarray, eps: float, mode: str) -> np.ndarray:
    psd_db = 10.0 * np.log10(np.maximum(psd_linear, eps)).astype(np.float32)
    if mode == "none":
        return psd_db
    elif mode == "minmax":
        pmin, pmax = float(np.min(psd_db)), float(np.max(psd_db))
        if pmax - pmin < 1e-9:
            return np.zeros_like(psd_db, dtype=np.float32)
        return ((psd_db - pmin) / (pmax - pmin)).astype(np.float32)
    elif mode == "zscore":
        mu = float(np.mean(psd_db))
        sigma = float(np.std(psd_db) + 1e-9)
        return ((psd_db - mu) / sigma).astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def convert_shard(shard_path: str, out_path: str, target_bins:int,
                  nperseg:int, noverlap:int, window:str, scaling:str,
                  onesided: bool, log_eps: float, norm: str, save_iq: bool) -> int:
    with open(shard_path, 'rb') as f:
        samples = pickle.load(f)

    converted = []
    for sample in samples:
        iq: np.ndarray = sample.get('iq')
        fs: float = float(sample.get('fs'))
        if iq is None:
            # Fall back: reconstruct rough IQ from spectrogram magnitude if necessary
            # but in our dataset IQ is present; skip otherwise.
            continue

        # Welch PSD (linear scale)
        f_in, p_in = compute_welch_psd(iq, fs, nperseg, noverlap, window, scaling, onesided)
        # Interpolate to uniform centered grid of TARGET_BINS
        f_uni, p_uni = interp_to_uniform_grid(f_in, p_in, target_bins, fs)
        # Log scaling and optional normalization
        psd_proc = log_and_norm(p_uni, log_eps, norm)

        # Build minimal sample for SigdetNet-style input
        new_sample = {
            'psd': psd_proc.astype(np.float32),      # shape: [TARGET_BINS]
            'freqs': f_uni.astype(np.float32),       # Hz
            'fs': fs,
            'sample_id': sample.get('sample_id'),
            'sample_type': sample.get('sample_type'),
            'uhf_carrier_hz': sample.get('uhf_carrier_hz'),
            'n_signals': sample.get('n_signals'),
            'signals': sample.get('signals'),        # keep per-signal metadata
            'timestamp': sample.get('timestamp'),
            'welch_params': {
                'nperseg': nperseg,
                'noverlap': noverlap,
                'window': window,
                'scaling': scaling,
                'onesided': onesided,
                'target_bins': target_bins,
                'log_eps': log_eps,
                'norm': norm,
            },
        }
        if save_iq:
            new_sample['iq'] = iq  # Warning: increases file size

        converted.append(new_sample)

    # Save converted shard
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(converted, f, protocol=pickle.HIGHEST_PROTOCOL)

    return len(converted)


def write_stats(src_dir: str, dst_dir: str, total_converted: int,
                target_bins:int, nperseg:int, noverlap:int, window:str,
                scaling:str, onesided: bool, log_eps: float, norm: str):
    # Try to inherit original stats if present
    src_stats_path = os.path.join(src_dir, 'dataset_stats.json')
    base = {}
    if os.path.isfile(src_stats_path):
        try:
            with open(src_stats_path, 'r') as f:
                base = json.load(f)
        except Exception:
            base = {}

    stats = {
        'source_dir': src_dir,
        'created_at': datetime.now().isoformat(),
        'total_samples': int(total_converted),
        'target_bins': int(target_bins),
        'welch': {
            'nperseg': int(nperseg),
            'noverlap': int(noverlap),
            'window': window,
            'scaling': scaling,
            'onesided': bool(onesided),
            'log_eps': float(log_eps),
            'norm': norm,
        },
        'original': {
            k: base.get(k) for k in [
                'fs', 'duration', 'n_fft', 'n_overlap', 'uhf_range', 'proportions',
                'modulation_counts', 'scenario_counts', 'seed'
            ] if k in base
        }
    }

    with open(os.path.join(dst_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Convert UHF dataset shards to Welch PSD format")
    ap.add_argument('--source', default=SOURCE_DIR, help='Source dataset directory')
    ap.add_argument('--dest',   default=DEST_DIR,   help='Destination directory for PSD shards')
    ap.add_argument('--nperseg', type=int, default=NPERSEG)
    ap.add_argument('--noverlap', type=int, default=NOVERLAP)
    ap.add_argument('--window', default=WINDOW)
    ap.add_argument('--scaling', default=SCALING, choices=['density','spectrum'])
    ap.add_argument('--onesided', action='store_true', help='Use onesided PSD (not typical for complex BB)')
    ap.add_argument('--target-bins', type=int, default=TARGET_BINS)
    ap.add_argument('--log-eps', type=float, default=LOG_EPS)
    ap.add_argument('--norm', default=NORM, choices=['none','minmax','zscore'])
    ap.add_argument('--keep-iq', action='store_true', help='Also keep original IQ in output shards')
    return ap.parse_args()


def main():
    args = parse_args()

    src_dir = os.path.abspath(args.source)
    dst_dir = os.path.abspath(args.dest)
    os.makedirs(dst_dir, exist_ok=True)

    shards = find_shards(src_dir)
    if not shards:
        print(f"No shards found in {src_dir}")
        return

    print(f"Converting {len(shards)} shards from:\n  {src_dir}\n→ to:\n  {dst_dir}")
    print(f"Welch: nperseg={args.nperseg}, noverlap={args.noverlap}, window={args.window}, scaling={args.scaling}, onesided={args.onesided}")
    print(f"Resample to TARGET_BINS={args.target_bins}, norm={args.norm}")

    total = 0
    for sid, shard_path in shards:
        out_name = SHARD_PREFIX_OUT.format(sid)
        out_path = os.path.join(dst_dir, out_name)
        n = convert_shard(
            shard_path, out_path, args.target_bins,
            args.nperseg, args.noverlap, args.window, args.scaling,
            args.onesided, args.log_eps, args.norm, args.keep_iq
        )
        total += n
        print(f"  shard {sid:04d}: {n:6d} samples → {out_name}")

    write_stats(
        src_dir, dst_dir, total,
        args.target_bins, args.nperseg, args.noverlap, args.window,
        args.scaling, args.onesided, args.log_eps, args.norm
    )

    print(f"\nDone. Converted samples: {total}")
    print(f"New stats: {os.path.join(dst_dir, 'dataset_stats.json')}")


if __name__ == '__main__':
    main()
