

import os, json, pickle, random, math, warnings
warnings.filterwarnings('ignore')
from scipy import signal as scipy_signal
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# =========================
# USER SETTINGS
# =========================
OUT_DIR = r"C:\Users\Osman\Desktop\BITES\sinyal_uhf\uhf_dataset_HYPER_REALISTIC_V10" 
NUM_SAMPLES = 50_000
PROPORTIONS = {"noise":0.07, "single":0.15, "mixed_close":0.53, "mixed_far":0.25}
FS = 2_000_000           # Hz
DURATION = 1e-3          # s
N_FFT = 256              # STFT nperseg
N_OVERLAP = 128          # STFT noverlap
SHARD_SIZE = 2_000
SEED = 20250814
MAX_SIGNAL = 3
# UHF metadata
UHF_MIN = 300e6
UHF_MAX = 3e9

# ---- Modülasyonlar ----
MODS = ['FM', 'OFDM', 'GFSK', 'QPSK']

# Mixed offsets
CLOSE_OFFSET_HZ = (5e3, 80e3)
FAR_OFFSET_FRAC = (0.20, 0.45)

# =========================
# TEMEL YARDIMCILAR
# =========================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_cpu_np(x: torch.Tensor):
    return x.detach().to('cpu').numpy()

def db2lin(db): return 10.0 ** (db/10.0)

def rc_impulse_response(beta: float, sps: int, span: int, device):
    L = span * sps
    t = torch.arange(-L/2, L/2 + 1, device=device, dtype=torch.float32) / sps
    pi = math.pi
    sinc = torch.where(t==0, torch.ones_like(t), torch.sin(pi*t)/(pi*t))
    num = torch.cos(pi*beta*t)
    den = 1 - (2*beta*t)**2
    h = sinc * num / torch.where(den.abs()<1e-8, torch.full_like(den, 1e-8), den)
    if beta > 0:
        idx0 = (t==0)
        if idx0.any(): h[idx0] = 1 - beta + 4*beta/pi
        idxs = (den.abs()<1e-6) & (~idx0)
        if idxs.any():
            val = (beta/2.0) * ( math.sin(pi/(2*beta)) + math.cos(pi/(2*beta))*(4/pi - 1.0) )
            h[idxs] = val
    h = h / torch.sqrt(torch.sum(h**2) + 1e-12)
    return h

def gaussian_pulse(BT: float, sps: int, span: int, device):
    L = span*sps + 1
    t = torch.linspace(-span/2, span/2, L, device=device, dtype=torch.float32)
    sigma_sym = 0.32/ max(BT, 1e-3)
    g = torch.exp(-0.5*(t/sigma_sym)**2)
    g = g / (g.sum() + 1e-12)
    return g

def resample_linear(x: torch.Tensor, factor: float):
    N = x.shape[0]
    idx_f = torch.arange(N, device=x.device, dtype=torch.float32) * factor
    idx_f = torch.clamp(idx_f, 0.0, N-1.001)
    idx0 = idx_f.floor().to(torch.long)
    idx1 = torch.clamp(idx0 + 1, max=N-1)
    frac = (idx_f - idx0.to(torch.float32)).unsqueeze(0)
    xr0 = x.real[idx0]; xr1 = x.real[idx1]
    xi0 = x.imag[idx0]; xi1 = x.imag[idx1]
    yr = (1-frac)*xr0 + frac*xr1
    yi = (1-frac)*xi0 + frac*xi1
    return (yr.squeeze(0) + 1j*yi.squeeze(0)).to(torch.complex64)

# ---------- SPEKTRAL ÖLÇÜM YARDIMCILARI ----------
def _next_pow2(n: int):
    p = 1
    while p < n: p <<= 1
    return p

def measure_spectral_metrics(iq: torch.Tensor, fs: float, p_occ: float = 0.99):
    device = iq.device
    N = iq.numel()
    win = torch.hann_window(N, device=device, dtype=torch.float32)
    x = iq * win.to(iq.dtype)
    nfft = _next_pow2(int(2*N))
    X = torch.fft.fft(x, n=nfft)
    X = torch.fft.fftshift(X)
    P = (X.real**2 + X.imag**2).to(torch.float32) + 1e-30
    freqs = torch.linspace(-fs/2, fs/2, steps=nfft, device=device, dtype=torch.float32)
    Psum = torch.sum(P)
    f_center = torch.sum(freqs * P) / Psum
    Pcum = torch.cumsum(P, dim=0) / Psum
    lo_q = (1.0 - p_occ) / 2.0; hi_q = 1.0 - lo_q
    il = torch.searchsorted(Pcum, torch.tensor(lo_q, device=device))
    ih = torch.searchsorted(Pcum, torch.tensor(hi_q, device=device))
    il = int(torch.clamp(il, 0, nfft-1)); ih = int(torch.clamp(ih, 0, nfft-1))
    bw_occ = float((freqs[ih] - freqs[il]).abs())
    var = torch.sum(P * (freqs - f_center)**2) / Psum
    bw_rms = float(2.0 * torch.sqrt(torch.clamp(var, min=0.0)))
    peak = torch.max(P)
    thr = peak / 2.0
    above = torch.nonzero(P >= thr, as_tuple=False).squeeze(-1)
    if above.numel() > 1:
        f_lo = float(freqs[above[0]]); f_hi = float(freqs[above[-1]])
        bw_3db = abs(f_hi - f_lo)
    else:
        bw_3db = 0.0
    return float(f_center.item()), float(bw_occ), float(bw_rms), float(bw_3db)

# ==========================================================
# NIHAI IQ ÜRETİCİ VE TÜM BOZULMA KÜTÜPHANESİ
# ==========================================================
class UltimateTorchIQGenerator:
    def __init__(self, fs=FS, duration=DURATION, device=None):
        self.fs = fs
        self.duration = duration
        self.N = int(round(fs*duration))
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.t = torch.arange(self.N, device=self.device, dtype=torch.float32) / fs
        self.pi = math.pi

    # ---- BÖLÜM 1: VERİCİ (TX) VE DONANIM BOZULMALARI ----

    def apply_pa_nonlinearity(self, iq: torch.Tensor, backoff_db=3.0) -> torch.Tensor:
        """Rapp modeli (AM/AM) ve basit AM/PM ile PA simülasyonu"""
        # Sinyali normalize et (peak power = 1.0) ve backoff uygula
        peak_power = torch.max(torch.abs(iq)**2) + 1e-12
        norm_factor = torch.sqrt(peak_power)
        iq_norm = iq / norm_factor
        
        # Backoff uygula (sinyali doyma noktasının altına çek)
        scale_lin = 10**(-backoff_db/20.0)
        iq_backed_off = iq_norm * scale_lin

        p_sat_linear = 1.0  # Normalized saturation power
        smoothness = 2.0  # Rapp p parametresi
        
        envelope = torch.abs(iq_backed_off) + 1e-12
        phase = torch.angle(iq_backed_off)
        
        # Rapp AM-AM modeli
        num = envelope
        den = torch.pow(1 + torch.pow(envelope**smoothness / p_sat_linear, 2), 1/(2*smoothness))
        envelope_out = num / den
        
        # AM-PM dönüşümü (Saleh modeli benzeri basit yaklaşım)
        # Güç arttıkça faz kayar
        am_pm_coeff = self.pi / 3.0  # Doyma noktasında 60 derece kayma (tipik)
        phase_shift = am_pm_coeff * (envelope**2) / (1 + envelope**2)
        
        # Orijinal ölçeğe geri dön
        return (envelope_out * torch.exp(1j * (phase + phase_shift)) * (norm_factor / scale_lin)).to(torch.complex64)

    def apply_memory_effects(self, iq: torch.Tensor, memory_depth=3) -> torch.Tensor:
        """Basit Volterra/Hafıza Polinomu modeli"""
        # Gerçekçi olmayan ama temsili katsayılar
        a = torch.tensor([
            [0.95 + 0.01j, -0.05 + 0.02j],  # k=1 (linear term + memory)
            [0.02 - 0.01j, 0.005 - 0.003j], # k=3 (nonlinear term + memory)
        ], device=self.device, dtype=torch.complex64)
        
        output = torch.zeros_like(iq)
        iq_abs_sq = torch.abs(iq)**2

        # k=1 (lineer terimler)
        output += a[0,0] * iq
        for m in range(1, memory_depth):
            delayed = torch.roll(iq, shifts=m)
            delayed[:m] = 0 # Sıfır doldurma
            output += a[0,1] * delayed # Basitleştirilmiş katsayı

        # k=3 (non-lineer terimler)
        output += a[1,0] * iq * iq_abs_sq
        for m in range(1, memory_depth):
            delayed = torch.roll(iq, shifts=m)
            delayed[:m] = 0
            output += a[1,1] * delayed * iq_abs_sq # Basitleştirilmiş
        
        return output.to(torch.complex64)

    def apply_lo_phase_noise(self, iq: torch.Tensor, pn_profile='1/f3') -> torch.Tensor:
        """Spektral şekilli gerçekçi LO Faz Gürültüsü"""
        N = len(iq)
        noise = torch.randn(N, device=self.device, dtype=torch.float32)
        freqs = torch.fft.fftfreq(N, 1/self.fs, device=self.device)
        freqs[0] = 1e-12 # DC'de sıfıra bölmeyi engelle

        if pn_profile == '1/f3':
            H = 1.0 / (torch.abs(freqs)**1.5) # Güç 1/f^3 -> genlik 1/f^1.5
            pn_power_dbchz = -120  # 1kHz'de dBc/Hz (temsili)
        else: # '1/f2' (Wiener/random walk benzeri)
            H = 1.0 / torch.abs(freqs) # Güç 1/f^2 -> genlik 1/f
            pn_power_dbchz = -110

        H = H * math.sqrt(10**(pn_power_dbchz/10))
        H[0] = 0 # DC bileşeni yok
        
        noise_fft = torch.fft.fft(noise)
        filtered_fft = noise_fft * H.to(torch.complex64)
        phase_noise_deriv = torch.fft.ifft(filtered_fft).real
        
        phase = torch.cumsum(phase_noise_deriv, dim=0) * 2 * math.pi / self.fs
        return (iq * torch.exp(1j * phase)).to(torch.complex64)

    def apply_iq_imbalance_complete(self, iq: torch.Tensor):
        I = iq.real; Q = iq.imag
        gI = float(torch.normal(1.0, 0.07, size=(1,), device=self.device))
        gQ = float(torch.normal(1.0, 0.07, size=(1,), device=self.device))
        eps_rad = float(torch.normal(0.0, math.radians(4.0), size=(1,), device=self.device))
        dc_i = float(torch.normal(0.0, 2e-3, size=(1,), device=self.device))
        dc_q = float(torch.normal(0.0, 2e-3, size=(1,), device=self.device))
        
        I_out = gI * I + dc_i
        Q_out = gQ * (Q*math.cos(eps_rad) + I*math.sin(eps_rad)) + dc_q
        
        out = (I_out + 1j*Q_out).to(torch.complex64)
        meta = dict(I_gain=gI, Q_gain=gQ, iq_phase_err_deg=math.degrees(eps_rad), dc=(dc_i, dc_q))
        return out, meta
        
    def apply_cfo_and_slow_doppler(self, iq: torch.Tensor):
        cfo0 = float(torch.empty(1, device=self.device).uniform_(-8e3, 8e3))
        cfo_slope = float(torch.empty(1, device=self.device).uniform_(-80.0, 80.0))  # Hz/s
        t = self.t
        cfo_t = cfo0 + cfo_slope * t
        phase = 2*self.pi * torch.cumsum(cfo_t / self.fs, dim=0)
        return (iq * torch.exp(1j*phase)).to(torch.complex64), dict(cfo0_hz=cfo0, cfo_slope_hz_s=cfo_slope)

    def apply_clock_drift(self, iq: torch.Tensor):
        drift_ppm = float(torch.empty(1, device=self.device).uniform_(-80.0, 80.0))
        factor = 1.0 + drift_ppm * 1e-6
        iq2 = resample_linear(iq, factor=factor)
        return iq2, dict(clock_drift_ppm=drift_ppm)

    # ---- BÖLÜM 2: KANAL MODELİ ----

    def apply_multipath_realistic(self, iq: torch.Tensor, 
                                 environment: str, velocity_ms: float) -> torch.Tensor:
        """Geometrik, fizik-tabanlı çok yollu kanal (Ricean/Rayleigh + Jakes Doppler)"""
        
        if environment == 'urban':
            delays_ns = [0, 100, 200, 500, 1000, 2000]
            powers_db = [0, -1, -3, -8, -12, -18]
            angles_deg = [0, 15, -30, 45, -60, 80]
            k_factor_db = 3  # Ricean K-faktörü (LOS zayıf)
        elif environment == 'suburban':
            delays_ns = [0, 50, 150, 800, 1500]
            powers_db = [0, -2, -6, -12, -20]
            angles_deg = [0, 20, -25, 40, -50]
            k_factor_db = 6
        else:  # 'rural'
            delays_ns = [0, 30, 400]
            powers_db = [0, -4, -15]
            angles_deg = [0, 10, -15]
            k_factor_db = 10 # Ricean K-faktörü (LOS güçlü)
        
        delays_samples = [int(d * 1e-9 * self.fs) for d in delays_ns]
        powers_linear = [10**(p/10.0) for p in powers_db] # Güç için /10
        powers_linear_norm = torch.tensor(powers_linear, device=self.device) / sum(powers_linear)
        
        fc_carrier = 900e6  # Doppler hesabı için temsili UHF taşıyıcı (900 MHz)
        c = 3e8
        
        output = torch.zeros_like(iq)
        meta_paths = []
        
        for i, (delay, power_norm, angle) in enumerate(zip(delays_samples, powers_linear_norm, angles_deg)):
            # Maksimum Doppler kayması (hızdan bağımsız)
            max_fd = velocity_ms * fc_carrier / c
            # Bu yol için Açıya bağlı Doppler
            fd = max_fd * math.cos(math.radians(angle))
            
            if i == 0:
                # ANA YOL (LOS): Ricean Fading
                k_linear = 10**(k_factor_db/10)
                los_power = power_norm * k_linear / (1 + k_linear)
                nlos_power = power_norm / (1 + k_linear)
                
                # LOS bileşeni (sabit Doppler kayması)
                doppler_phase = 2 * math.pi * fd * self.t
                los_component = torch.sqrt(los_power) * torch.exp(1j * doppler_phase)
                
                # NLOS (dağınık) bileşen (Jakes spektrumlu Rayleigh)
                rayleigh_fading = self._generate_rayleigh_fading(max_fd, self.N) # Jakes, maks doppler'i kullanır
                nlos_component = torch.sqrt(nlos_power) * rayleigh_fading
                
                channel_tap = los_component + nlos_component
            else:
                # YANSIMA YOLLARI (NLOS): Sadece Rayleigh Fading
                rayleigh_fading = self._generate_rayleigh_fading(max_fd, self.N)
                channel_tap = torch.sqrt(power_norm) * rayleigh_fading

            path_signal = iq * channel_tap

            # Gecikmeyi uygula
            if delay > 0 and delay < self.N:
                path_signal = torch.roll(path_signal, shifts=delay)
                path_signal[:delay] = 0.0 # Başı sıfırla
            
            output += path_signal
            meta_paths.append(dict(delay=delay, power_db=10*math.log10(float(power_norm)), fd_hz=fd, is_los=(i==0)))

        return output.to(torch.complex64), meta_paths

    def _generate_rayleigh_fading(self, max_fd, length):
        """Jakes Doppler Spektrumu ile Rayleigh sönümlemesi üretir"""
        if max_fd < 0.1: # Neredeyse statik kanal
             g = (torch.randn(1, device=self.device) + 1j*torch.randn(1, device=self.device)) / math.sqrt(2.0)
             return torch.full((length,), g, device=self.device, dtype=torch.complex64)
             
        # Jakes modeli için filtre tasarımı
        N_filt = 256
        t_filt = torch.arange(-N_filt//2, N_filt//2 + 1, device=self.device, dtype=torch.float32) / self.fs
        
        # Jakes spektrumu otokorelasyonu (Bessel J0)
        autocorr = torch.special.bessel_j0(2 * math.pi * max_fd * torch.abs(t_filt))
        
        # Filtre katsayıları (FIR filtresi)
        fir_taps = autocorr * torch.hann_window(N_filt+1, device=self.device)
        fir_taps = fir_taps / torch.norm(fir_taps) # Normalizasyon
        fir_taps = fir_taps.to(torch.complex64)
        
        # Beyaz gürültü üret ve filtrele
        white_noise = (torch.randn(length + N_filt, device=self.device) + 
                       1j*torch.randn(length + N_filt, device=self.device)) / math.sqrt(2.0)
        
        # Konvolüsyon
        fading = F.conv1d(white_noise.view(1, 1, -1), fir_taps.view(1, 1, -1), padding=N_filt//2).squeeze()
        return fading[:length]

    # ---- BÖLÜM 3: ALICI (RX) VE DIŞ BOZULMALAR ----

    def add_awgn_total(self, iq_mix: torch.Tensor, snr_db: float):
        """Alıcının girişindeki termal gürültü (AWGN)"""
        sp = torch.mean(torch.abs(iq_mix)**2) + 1e-20
        npow = sp / db2lin(snr_db)
        n = torch.sqrt(npow/2) * (torch.randn(self.N, device=self.device) + 1j*torch.randn(self.N, device=self.device))
        return (iq_mix + n.to(torch.complex64)).to(torch.complex64)

    def apply_thermal_drift(self, iq: torch.Tensor, temp_range_c=10) -> torch.Tensor:
        """Alıcı donanımının sıcaklığa bağlı kayması"""
        temp_freq_hz = 0.1  # Yavaş sıcaklık değişimi
        temp_variation = temp_range_c * torch.sin(2 * math.pi * temp_freq_hz * self.t)
        
        gain_temp_coeff = -0.01  # -0.01 dB/°C
        phase_temp_coeff = 0.5   # 0.5°/°C
        freq_temp_coeff = -2.5   # -2.5 ppm/°C (RX LO kayması)
        
        gain_drift_db = gain_temp_coeff * temp_variation
        phase_drift_deg = phase_temp_coeff * temp_variation
        freq_drift_ppm = freq_temp_coeff * temp_variation
        
        gain_drift_lin = 10**(gain_drift_db / 20.0)
        phase_drift_rad = phase_drift_deg * math.pi / 180.0
        freq_drift_hz = freq_drift_ppm * 1e-6 * 900e6 # 900Mhz LO varsayımı
        
        freq_phase_drift = 2 * math.pi * torch.cumsum(freq_drift_hz, dim=0) / self.fs
        total_phase_drift = phase_drift_rad + freq_phase_drift
        
        return (iq * gain_drift_lin * torch.exp(1j * total_phase_drift)).to(torch.complex64)

    def apply_adc_realistic(self, iq: torch.Tensor, bits=12, 
                          full_scale_dbfs=-1.0, add_harmonics=True) -> torch.Tensor:
        """ADC Kuantalama, Kırpma (Clipping) ve Harmonik Distorsiyon"""
        full_scale_lin = 10**(full_scale_dbfs / 20.0)
        
        # Kırpma (Clipping) - ADC'nin dinamik aralığını aşan sinyaller
        max_abs = torch.max(torch.abs(iq))
        if max_abs > full_scale_lin:
            # Sinyal çok güçlüyse kırp
            iq = torch.clamp(iq.real, -full_scale_lin, full_scale_lin) + \
               1j*torch.clamp(iq.imag, -full_scale_lin, full_scale_lin)
        
        levels = 2**bits
        step = 2 * full_scale_lin / levels
        
        if add_harmonics:
            hd2_dbc = -60; hd3_dbc = -65 # Tipik ADC harmonik seviyeleri
            hd2_amp = 10**(hd2_dbc/20.0); hd3_amp = 10**(hd3_dbc/20.0)
            
            # Harmonikleri ekle (normalleştirilmiş sinyal üzerinden)
            norm_iq = iq / full_scale_lin
            harmonic2 = hd2_amp * norm_iq**2 
            harmonic3 = hd3_amp * norm_iq**3
            iq = (iq + (harmonic2 + harmonic3) * full_scale_lin).to(torch.complex64)

        # Kuantalama
        I_q = torch.round(iq.real / step) * step
        Q_q = torch.round(iq.imag / step) * step
        
        return (I_q + 1j*Q_q).to(torch.complex64)

    def add_realistic_interference(self, iq: torch.Tensor) -> torch.Tensor:
        """Dış parazit kaynakları ekle: WiFi, Radar, vb."""
        interference = torch.zeros_like(iq)
        
        # WiFi paraziti (Geniş bant OFDM-benzeri gürültü)
        if random.random() < 0.3:
            wifi_freq_offset = random.uniform(-self.fs/2, self.fs/2)
            wifi_power_db = random.uniform(-60, -40) # dB (sinyal gücüne göre değil, mutlak)
            wifi_bw = 20e6
            
            wifi_signal = self._generate_ofdm_interference(wifi_bw, wifi_power_db)
            wifi_signal *= torch.exp(1j * 2 * math.pi * wifi_freq_offset * self.t)
            interference += wifi_signal
        
        # Radar palsleri (Chirp)
        if random.random() < 0.1:
            pulse_power_db = random.uniform(-50, -30)
            radar_signal = self._generate_radar_pulses(
                pulse_width=random.uniform(1e-6, 10e-6),
                pri=random.uniform(100e-6, 1e-3),
                power_db=pulse_power_db
            )
            radar_freq_offset = random.uniform(-self.fs/2, self.fs/2)
            radar_signal *= torch.exp(1j * 2 * math.pi * radar_freq_offset * self.t)
            interference += radar_signal
        
        return iq + interference.to(torch.complex64)

    def _generate_ofdm_interference(self, bandwidth, power_db):
        """WiFi benzeri geniş bant gürültü üretir"""
        power_lin = db2lin(power_db)
        noise = torch.sqrt(torch.tensor(power_lin/2, device=self.device)) * (
            torch.randn(self.N, device=self.device) + 1j*torch.randn(self.N, device=self.device)
        )
        # Basit bir alçak geçiren filtre (bant genişliğini simüle etmek için)
        cutoff_norm = (bandwidth / 2.0) / (self.fs / 2.0)
        if cutoff_norm < 1.0:
             # Scipy kullanarak basit bir FIR filtresi tasarlayalım (Torch'ta zor)
             try:
                 taps = scipy_signal.firwin(numtaps=65, cutoff=cutoff_norm, window='hann')
                 taps_t = torch.tensor(taps, device=self.device, dtype=torch.float32).view(1, 1, -1)
                 noise_r = F.conv1d(noise.real.view(1,1,-1), taps_t, padding='same').squeeze()
                 noise_i = F.conv1d(noise.imag.view(1,1,-1), taps_t, padding='same').squeeze()
                 return (noise_r + 1j*noise_i).to(torch.complex64)
             except Exception:
                 return noise.to(torch.complex64) # Filtre başarısız olursa ham gürültü
        return noise.to(torch.complex64)

    def _generate_radar_pulses(self, pulse_width, pri, power_db):
        power_lin = db2lin(power_db)
        pulse_samples = int(pulse_width * self.fs); pri_samples = int(pri * self.fs)
        if pri_samples <= 0: pri_samples = self.N + 1

        signal = torch.zeros(self.N, device=self.device, dtype=torch.complex64)
        t_pulse = torch.arange(pulse_samples, device=self.device, dtype=torch.float32) / self.fs
        chirp_bw = random.uniform(1e6, 5e6) # 1-5 MHz chirp BW
        chirp_rate = chirp_bw / pulse_width
        
        phase = math.pi * chirp_rate * t_pulse**2
        pulse = torch.sqrt(torch.tensor(power_lin, device=self.device)) * torch.exp(1j * phase)

        pulse_start = random.randint(0, pri_samples//2)
        while pulse_start + pulse_samples < self.N:
            signal[pulse_start : pulse_start + pulse_samples] = pulse
            pulse_start += pri_samples
        
        return signal

    # ---- BÖLÜM 4: ÖZELLİK ÇIKARMA (9_2'DEN ALINAN FİLTRE) ----

    def compute_welch_log_spectrogram(self, iq: torch.Tensor, n_fft=N_FFT, noverlap=N_OVERLAP):
        hop = n_fft - noverlap
        window = torch.hann_window(n_fft, device=self.device)
        Z = torch.stft(iq, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=window, return_complex=True, center=False, onesided=False)
        
        # KANAL 1: Welch-benzeri Güç Spektrogramı ve Logaritmik Geliştirme
        power_spec = torch.abs(Z)**2
        c = 1.0
        enhanced_spec = torch.log1p(c * power_spec) / math.log(c + 1.0)
        min_val = torch.min(enhanced_spec); max_val = torch.max(enhanced_spec)
        power_channel = (enhanced_spec - min_val) / (max_val - min_val + 1e-8)
        power_channel = power_channel.to(torch.float32)

        # KANAL 2: Faz (Normalleştirilmiş)
        phase = torch.angle(Z)
        phase = (phase + self.pi)/(2*self.pi)
        phase = phase.to(torch.float32)

        # KANAL 3: Anlık Frekans Spektrogramı
        prev = torch.roll(iq, shifts=1); prev[0] = 0.0
        dphi = torch.angle(iq * torch.conj(prev))
        inst_f = (self.fs/(2*self.pi))*dphi.to(torch.float32)
        Z_if = torch.stft(inst_f, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                          window=window, return_complex=True, center=False, onesided=False)
        if_spec = torch.abs(Z_if)
        q1 = torch.quantile(if_spec, 0.01); q9 = torch.quantile(if_spec, 0.99)
        if_spec = torch.clamp(if_spec, q1, q9)
        if_spec = ((if_spec - if_spec.min())/(if_spec.max()-if_spec.min()+1e-8)).to(torch.float32)

        # KANAL 4: Faz Türevi Spektrogramı
        phase_der = torch.diff(phase, dim=0, prepend=phase[0:1])
        q1 = torch.quantile(phase_der, 0.01); q9 = torch.quantile(phase_der, 0.99)
        phase_der = torch.clamp(phase_der, q1, q9)
        phase_der = ((phase_der - phase_der.min())/(phase_der.max()-phase_der.min()+1e-8)).to(torch.float32)
        
        return power_channel, phase, if_spec, phase_der

    # ---- BÖLÜM 5: MODÜLATÖRLER (9_2'DEN ALINAN) VE ENTEGRASYON ----

    def _filter_up(self, symbols: torch.Tensor, taps: torch.Tensor, sps: int):
        L = symbols.numel()*sps
        up = torch.zeros(L, device=self.device, dtype=torch.complex64)
        up[::sps] = symbols.to(torch.complex64)
        pad = taps.numel()//2
        I = F.conv1d(up.real.view(1,1,-1), taps.view(1,1,-1), padding=pad).view(-1)
        Q = F.conv1d(up.imag.view(1,1,-1), taps.view(1,1,-1), padding=pad).view(-1)
        y = (I + 1j*Q).to(torch.complex64)
        if y.numel() < self.N: y = F.pad(y, (0, self.N-y.numel()))
        return y[:self.N]

    # --- TX ZİNCİRİNİ UYGULAYAN YENİ FİNALİZE FONKSİYONU ---
    def _apply_full_tx_chain(self, iq_base: torch.Tensor, tx_scenario: Dict) -> Tuple[torch.Tensor, Dict]:
        """Tüm Verici (TX) ve Kanal bozulmalarını sırayla uygular"""
        iq = iq_base.clone()
        meta = {}
        
        # 1. IQ Dengesizliği (Baseband TX)
        iq, iq_meta = self.apply_iq_imbalance_complete(iq)
        meta.update(iq_meta)

        # 2. PA Non-lineerlik ve Hafıza (RF TX Amplifikatör)
        if tx_scenario['use_pa']:
             iq = self.apply_pa_nonlinearity(iq, backoff_db=tx_scenario['backoff_db'])
             iq = self.apply_memory_effects(iq)
             meta.update(pa_backoff_db=tx_scenario['backoff_db'])
        
        # 3. LO Faz Gürültüsü (TX LO)
        iq = self.apply_lo_phase_noise(iq, pn_profile=tx_scenario['pn_profile'])
        meta.update(pn_profile=tx_scenario['pn_profile'])

        # 4. Frekans/Saat Sapmaları (TX vs RX uyumsuzluğu)
        iq, cfo_meta = self.apply_cfo_and_slow_doppler(iq)
        meta.update(cfo_meta)
        if random.random() < 0.8:
            iq, clk_meta = self.apply_clock_drift(iq)
            meta.update(clk_meta)

        # 5. KANAL
        iq, path_meta = self.apply_multipath_realistic(iq, 
                                          environment=tx_scenario['environment'], 
                                          velocity_ms=tx_scenario['velocity_ms'])
        meta.update(rayleigh_paths=path_meta, environment=tx_scenario['environment'])

        # Normalize power (kanaldan sonra güç değişebilir)
        p_cur = torch.mean(torch.abs(iq)**2) + 1e-20
        p_tgt = db2lin(tx_scenario['power_db'])
        iq = iq * torch.sqrt(torch.tensor(p_tgt, device=self.device, dtype=torch.float32)/p_cur)
        
        # Teorik frekans merkezi hesabı
        D = self.N / self.fs
        f_center_theory = meta.get("cfo0_hz", 0.0) + 0.5 * meta.get("cfo_slope_hz_s", 0.0) * D
        meta.update(f_center_theory_hz_approx=float(f_center_theory))

        return iq.to(torch.complex64), meta

    # --- MODÜLATÖRLER (YENİ TX ZİNCİRİNİ KULLANACAK ŞEKİLDE GÜNCELLENDİ) ---
    
    def gen_qpsk(self, power_db=-10, rsym_range=(100e3, 400e3), rolloff_range=(0.2,0.5), tx_scenario=None):
        Rs = float(torch.empty(1, device=self.device).uniform_(*rsym_range))
        sps = max(2, int(self.fs/Rs))
        n_sym = int(math.ceil(self.N/sps))+16
        bits = torch.randint(0, 2, (2*n_sym,), device=self.device)
        syms = ((2*bits[0::2]-1) + 1j*(2*bits[1::2]-1)).to(torch.float32) / math.sqrt(2.0)
        beta = float(torch.empty(1, device=self.device).uniform_(*rolloff_range))
        taps = rc_impulse_response(beta, sps, span=10, device=self.device)
        iq_base = self._filter_up(syms, taps, sps)
        
        # Zamanlama kayması
        toff0 = int(torch.randint(0, sps, (1,), device=self.device))
        iq_base = torch.roll(iq_base, shifts=toff0)

        # Tüm TX + Kanal zincirini uygula
        tx_scenario['power_db'] = power_db
        iq_final, meta = self._apply_full_tx_chain(iq_base, tx_scenario)

        meta.update(dict(Rs=Rs, sps=sps, rolloff=beta, pulse='RC', time_ofs=int(toff0)))
        return iq_final, meta

    def gen_fm(self, power_db=-10, fdev_range=(3e3, 30e3), fm_range=(1e3, 10e3), tx_scenario=None):
        fm = float(torch.empty(1, device=self.device).uniform_(*fm_range))
        msg = torch.sin(2*math.pi*fm*self.t)
        fdev = float(torch.empty(1, device=self.device).uniform_(*fdev_range))
        phase = 2*math.pi*fdev*torch.cumsum(msg, dim=0)/self.fs
        iq_base = torch.exp(1j*phase).to(torch.complex64)
        
        # Tüm TX + Kanal zincirini uygula (FM için PA simülasyonunu atla, zaten sabit zarflı)
        tx_scenario['power_db'] = power_db
        tx_scenario['use_pa'] = False # FM sabit zarflıdır, PA non-lineerliği önemli değil
        iq_final, meta = self._apply_full_tx_chain(iq_base, tx_scenario)

        meta.update(dict(fm_hz=fm, fdev_hz=fdev))
        return iq_final, meta

    def gen_ofdm(self, power_db=-10, tx_scenario=None):
        Nfft = int(random.choice([64,128,256])); cp = int(random.choice([Nfft//16, Nfft//8]))
        n_sym = max(10, int(self.N // (Nfft+cp)) + 2)
        active = torch.zeros(Nfft, dtype=torch.bool, device=self.device); lo = Nfft//8; hi = Nfft - lo
        active[lo:hi] = True

        lv = torch.tensor([-3,-1,1,3], device=self.device, dtype=torch.float32) # 16QAM
        I = lv[torch.randint(0,4,(n_sym,Nfft), device=self.device)]
        Q = lv[torch.randint(0,4,(n_sym,Nfft), device=self.device)]
        Es = torch.mean(I**2+Q**2); QAM = (I + 1j*Q)/torch.sqrt(Es+1e-12)
        QAM *= active
        
        x = torch.fft.ifft(QAM, dim=1)
        x = torch.cat([x[:, -cp:], x], dim=1).reshape(-1)
        if x.numel() < self.N: x = F.pad(x, (0, self.N-x.numel()))
        iq_base = x[:self.N].to(torch.complex64)

        # OFDM, yüksek PAPR'a sahiptir, PA non-lineerliği için MÜKEMMEL bir adaydır
        tx_scenario['power_db'] = power_db
        tx_scenario['use_pa'] = True # Kesinlikle evet
        tx_scenario['backoff_db'] = random.uniform(4.0, 8.0) # OFDM için daha fazla backoff gerekir
        iq_final, meta = self._apply_full_tx_chain(iq_base, tx_scenario)

        meta.update(dict(Nfft=Nfft, cp=cp, n_sym=n_sym, mod_order=16))
        return iq_final, meta

    def gen_gfsk(self, power_db=-10, rsym_range=(50e3, 200e3), BT=0.5, tx_scenario=None):
        Rs = float(torch.empty(1, device=self.device).uniform_(*rsym_range))
        sps = max(2, int(self.fs/Rs))
        n_sym = int(math.ceil(self.N/sps))+16
        bits = torch.randint(0,2,(n_sym,), device=self.device)*2 - 1
        g = gaussian_pulse(BT, sps, span=6, device=self.device)
        m = torch.zeros(n_sym*sps, device=self.device); m[::sps] = bits.to(torch.float32)
        pad = g.numel()//2
        m = F.conv1d(m.view(1,1,-1), g.view(1,1,-1), padding=pad).view(-1)
        m = m[:self.N] if m.numel()>=self.N else F.pad(m,(0,self.N-m.numel()))
        fdev = float(torch.empty(1, device=self.device).uniform_(0.35*Rs, 0.7*Rs))
        phase = 2*math.pi * fdev * torch.cumsum(m, dim=0)/self.fs
        iq_base = torch.exp(1j*phase).to(torch.complex64)
        
        # GFSK (FM gibi) sabit zarflıdır, PA non-lineerliği önemsizdir
        tx_scenario['power_db'] = power_db
        tx_scenario['use_pa'] = False 
        iq_final, meta = self._apply_full_tx_chain(iq_base, tx_scenario)
        
        meta.update(dict(Rs=Rs, sps=sps, BT=BT, fdev=fdev))
        return iq_final, meta

# =========================
# YÜKSEK SEVİYE ÜRETİCİLER (9.2'DEN)
# =========================

def _get_random_tx_scenario():
    """Her sinyal için rastgele bir TX/Kanal senaryosu seçer"""
    return {
        'environment': random.choice(['urban', 'suburban', 'rural']),
        'velocity_ms': random.uniform(0.0, 80.0), # 0 (statik) - ~300 km/s arası
        'use_pa': True, # Varsayılan olarak açık (modülatör içinde kapatılabilir)
        'backoff_db': random.uniform(1.0, 6.0),
        'pn_profile': random.choice(['1/f3', '1/f2'])
    }

def gen_single(gen: UltimateTorchIQGenerator):
    mod = random.choice(MODS)
    p_db = random.uniform(-20, 0)
    scenario = _get_random_tx_scenario()
    
    if   mod=='FM':   iq, meta = gen.gen_fm(p_db, tx_scenario=scenario)
    elif mod=='OFDM': iq, meta = gen.gen_ofdm(p_db, tx_scenario=scenario)
    elif mod=='GFSK': iq, meta = gen.gen_gfsk(p_db, tx_scenario=scenario)
    else:             iq, meta = gen.gen_qpsk(p_db, tx_scenario=scenario)

    f_center_est, bw_occ, bw_rms, bw_3db = measure_spectral_metrics(iq, gen.fs, p_occ=0.99)
    info = [{
        **meta, "mod": mod, "f_off_hz": 0.0, "rel_power_db": 0.0,
        "f_center_est_hz": f_center_est, "bw_occ99_hz": bw_occ,
        "bw_rms_hz": bw_rms, "bw_3db_hz": bw_3db,
    }]
    return iq, info

def gen_mixed(gen: UltimateTorchIQGenerator, close=True):
    k = random.randint(2, MAX_SIGNAL)
    rel_powers_db = sorted([random.uniform(-18, 0) for _ in range(k)], reverse=True)
    sigs, info = [], []
    for idx in range(k):
        mod = random.choice(MODS); p_db = rel_powers_db[idx]
        scenario = _get_random_tx_scenario()
        
        if   mod=='FM':   s, meta = gen.gen_fm(p_db, tx_scenario=scenario)
        elif mod=='OFDM': s, meta = gen.gen_ofdm(p_db, tx_scenario=scenario)
        elif mod=='GFSK': s, meta = gen.gen_gfsk(p_db, tx_scenario=scenario)
        else:             s, meta = gen.gen_qpsk(p_db, tx_scenario=scenario)

        if close: foff = random.uniform(*CLOSE_OFFSET_HZ) * random.choice([-1, 1])
        else: foff = random.uniform(*FAR_OFFSET_FRAC) * random.choice([-1, 1]) * gen.fs
        
        s = (s * torch.exp(1j*2*math.pi*foff*gen.t)).to(torch.complex64)
        
        f_center_est, bw_occ, bw_rms, bw_3db = measure_spectral_metrics(s, gen.fs, p_occ=0.99)
        row = {
            **meta, "mod": mod, "f_off_hz": float(foff), "rel_power_db": float(p_db),
            "f_center_theory_hz_approx": meta.get("f_center_theory_hz_approx", 0.0) + float(foff),
            "f_center_est_hz": f_center_est, "bw_occ99_hz": bw_occ,
            "bw_rms_hz": bw_rms, "bw_3db_hz": bw_3db,
        }
        sigs.append(s)
        info.append(row)

    mix = torch.zeros(gen.N, device=gen.device, dtype=torch.complex64)
    for s in sigs: mix = mix + s
    return mix, info

def gen_noise(gen: UltimateTorchIQGenerator):
    noise_db = random.uniform(-50, -20)
    p_lin = db2lin(noise_db)
    n = torch.sqrt(torch.tensor(p_lin/2, device=gen.device)) * \
        (torch.randn(gen.N, device=gen.device) + 1j*torch.randn(gen.N, device=gen.device))
    return n.to(torch.complex64), []

# =========================
# ANA VERİ SETİ YAZICI (GÜNCELLENMİŞ RX ZİNCİRİ İLE)
# =========================
def generate_ultimate_realistic_dataset(
    out_dir=OUT_DIR, num_samples=NUM_SAMPLES, proportions=PROPORTIONS,
    fs=FS, duration=DURATION, n_fft=N_FFT, n_overlap=N_OVERLAP, shard_size=SHARD_SIZE, seed=SEED
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = UltimateTorchIQGenerator(fs=fs, duration=duration, device=device)

    print(f"🧠 NIHAI ÜRETİCİ BAŞLATILDI. Cihaz: {device}")

    counts = {k:int(round(v*num_samples)) for k,v in proportions.items()}
    diff = num_samples - sum(counts.values())
    counts['mixed_close'] += diff

    plan = (['noise']*counts['noise'] + ['single']*counts['single'] +
            ['mixed_close']*counts['mixed_close'] + ['mixed_far']*counts['mixed_far'])
    random.shuffle(plan)

    manifest = {
        "created": datetime.now().isoformat(),
        "seed": seed, "baseband_fs": fs, "duration_s": duration,
        "band_limits_hz": [UHF_MIN, UHF_MAX],
        "snr_range_db_total": [0, 30],
        "proportions": proportions,
        "stft": {"n_fft": n_fft, "noverlap": n_overlap, "hop": n_fft-n_overlap},
        "shard_size": shard_size, "num_samples": num_samples, "mods": MODS,
        "feature_channels": ["log_enhanced_power_norm", "phase_norm", "instfreq_spec_norm", "phase_deriv_norm"],
        "impairments_added": [
             "--- TX CHAIN ---",
             "IQ Imbalance (Freq-Independent)", "PA Non-linearity (Rapp AM/AM+AM/PM)",
             "PA Memory Effects (Volterra-like)", "LO Phase Noise (1/f^2, 1/f^3 profiles)",
             "CFO + Linear Drift", "Clock Drift (Resampling)",
             "--- CHANNEL ---",
             "Geometric Multipath (Urban/Suburban/Rural)", "Ricean + Rayleigh (Jakes) Fading", "Per-Path Doppler",
             "--- RX CHAIN ---",
             "AWGN", "External Interference (WiFi, Radar)", 
             "Receiver Thermal Drift (Gain/Phase/Freq)", "ADC Model (Quantization, Clipping, Harmonics)"
        ],
        "version": "V11_ULTIMATE_FULL_CHAIN_SIM",
        "label_fields_per_signal": [
            "mod", "rel_power_db", "f_off_hz", "f_center_est_hz", 
            "bw_occ99_hz", "bw_rms_hz", "bw_3db_hz", "environment", "velocity_ms"
        ],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    shard_idx = 0
    buf_features, buf_iq_raw, buf_labels = [], [], []
    stats = {"noise":0, "single":0, "mixed_close":0, "mixed_far":0}

    pbar = tqdm(plan, desc="Generating ULTIMATE (V11) UHF Dataset", unit="sample")
    for cat in pbar:
        abs_fc_list = []
        if cat == 'single':
            abs_fc_list = [float(np.random.uniform(UHF_MIN, UHF_MAX))]
        elif cat != 'noise':
            k = random.randint(2,4); base_fc = float(np.random.uniform(UHF_MIN, UHF_MAX))
            abs_fc_list = [float(np.clip(base_fc + np.random.uniform(-5e6, 5e6), UHF_MIN, UHF_MAX)) for _ in range(k)]

        # --- Adım 1 & 2 & 3: TX + Kanal Simülasyonu ile Sinyalleri Üret ---
        if   cat=='single':      iq, info = gen_single(gen)
        elif cat=='mixed_close': iq, info = gen_mixed(gen, close=True)
        elif cat=='mixed_far':   iq, info = gen_mixed(gen, close=False)
        else:                    iq, info = gen_noise(gen) # Sadece temel gürültü

        # --- Adım 4 & 5 & 6: RX Zincirini Uygula (Toplam Sinyal Üzerinden) ---
        if cat != 'noise':
            # 4. Termal Gürültü (AWGN) ekle
            snr_db = float(np.random.uniform(0, 30))
            iq = gen.add_awgn_total(iq, snr_db)
        else:
            snr_db = -np.inf # Gürültü örneği zaten gürültü

        # 5. Dış Parazitleri Ekle (AWGN eklendikten sonra, ADC'den önce)
        iq = gen.add_realistic_interference(iq)

        # 6. RX Donanım Efektlerini Ekle
        iq = gen.apply_thermal_drift(iq, temp_range_c=random.uniform(5, 20))
        
        # 7. ADC Simülasyonu (Zincirdeki SON IQ adımı)
        iq = gen.apply_adc_realistic(iq, bits=int(random.choice([10, 12, 14])))
        
        # --- Adım 7: Özellik Çıkarma ---
        # "Filtre" adımı: Welch/Log dönüşümü
        power_ch, ph, ifs, phg = gen.compute_welch_log_spectrogram(iq, n_fft=n_fft, noverlap=n_overlap)
        feature_stack = torch.stack([power_ch, ph, ifs, phg], dim=0).to(torch.float32)

        buf_features.append(to_cpu_np(feature_stack))
        buf_iq_raw.append(to_cpu_np(iq.to(torch.complex64)))
        buf_labels.append({
            "type": cat,
            "num_signals": len(info),
            "signals": info,
            "snr_db_total": snr_db,
            "abs_fc_list_hz": abs_fc_list
        })
        stats[cat] += 1

        # Shard'ı diske yaz
        if len(buf_features) >= shard_size:
            shard_path = os.path.join(out_dir, f"shard_{shard_idx:05d}")
            os.makedirs(shard_path, exist_ok=True)
            np.save(os.path.join(shard_path, "features.npy"), np.stack(buf_features, axis=0))
            np.save(os.path.join(shard_path, "iq_raw.npy"),    np.stack(buf_iq_raw, axis=0))
            with open(os.path.join(shard_path, "labels.pkl"), "wb") as f:
                pickle.dump(buf_labels, f)
            buf_features, buf_iq_raw, buf_labels = [], [], []
            shard_idx += 1

    # Son shard'ı kaydet
    if buf_features:
        shard_path = os.path.join(out_dir, f"shard_{shard_idx:05d}")
        os.makedirs(shard_path, exist_ok=True)
        np.save(os.path.join(shard_path, "features.npy"), np.stack(buf_features, axis=0))
        np.save(os.path.join(shard_path, "iq_raw.npy"),    np.stack(buf_iq_raw, axis=0))
        with open(os.path.join(shard_path, "labels.pkl"), "wb") as f:
            pickle.dump(buf_labels, f)

    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*64)
    print("✅ NIHAI (V11) TAM ZİNCİR RF VERİ SETİ ÜRETİMİ TAMAMLANDI!")
    print("="*64)
    print(f"📂 Çıktı dizini: {out_dir}")
    print(f"📊 Toplam örnek: {sum(stats.values())}")
    print(f"📈 Dağılım: {stats}")

# =========================
# ANA ÇALIŞTIRMA
# =========================
if __name__ == "__main__":
    generate_ultimate_realistic_dataset()
    print("\nTüm işlemler tamamlandı.")

"""RF Zinciri Simülasyonu: Tam bir radyo haberleşme zincirini simüle ediyor - verici (TX), kanal ve alıcı (RX) bozulmalarını sistematik şekilde modellemiş. PA nonlineerlik, IQ dengesizlik, LO faz gürültüsü gibi gerçek donanım etkilerini içeriyor.
Çoklu Modülasyon Desteği: FM, OFDM, GFSK, QPSK gibi farklı modülasyon türlerini destekliyor ve her birini doğru şekilde modelliyor.
Kanal Modeli: Çok yollu yayılım (multipath) ve Doppler etkilerini geometrik/fiziksel temelde modelliyor. Urban/suburban/rural ortamlar için farklı parametreler kullanıyor.
Spektral Analiz: STFT tabanlı spektrogram çıkarma ve çok kanallı özellik üretimi (güç, faz, anlık frekans, faz türevi) yapıyor.
Potansiyel İyileştirme Alanları
Kod Organizasyonu: ~800 satırlık tek dosya oldukça büyük. Modüler yapıya ayrılabilir (TX, RX, kanal, modülasyon sınıfları ayrı dosyalarda).
Bellek Yönetimi: Büyük tensörler GPU belleğinde tutularak bellek taşması riski var. Batch processing veya lazy loading düşünülebilir.
Parametre Validasyonu: Birçok rastgele parametre kullanılıyor ama geçerlilik kontrolleri sınırlı. Fiziksel olarak anlamsız parametre kombinasyonları oluşabilir.
Performans: Scipy sinyallerini PyTorch'a dönüştürme işlemleri gereksiz overhead yaratıyor olabilir.
Genel Değerlendirme
Bu, akademik araştırma veya RF makine öğrenmesi için oldukça sofistike bir araç. Gerçek RF sistemlerindeki bozulmaları iyi modellemiş ve eğitim verisi çeşitliliğini sağlamış. Ancak karmaşıklığı nedeniyle hata ayıklama ve bakım zorlayıcı olabilir."""