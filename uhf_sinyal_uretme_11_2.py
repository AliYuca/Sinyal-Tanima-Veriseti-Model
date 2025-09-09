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
OUT_DIR = r"C:\Users\Osman\Desktop\BITES\sinyal_uhf\uhf_dataset_real_11" 
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

# Mixed offsets - Daha gerçekçi frekans ayırımları
CLOSE_OFFSET_HZ = (10e3, 50e3)  # Yakın kanallar için daha dar aralık
FAR_OFFSET_FRAC = (0.05, 0.15)  # Uzak kanallar için daha makul aralık

# Fiziksel limitler - Gerçekçi RF değerleri
RF_LIMITS = {
    'cfo_hz': (-5e3, 5e3),              # CFO: ±5kHz (kristal toleransı)
    'cfo_slope_hz_s': (-20.0, 20.0),    # CFO drift: ±20Hz/s
    'clock_drift_ppm': (-20.0, 20.0),   # Saat sapması: ±20ppm
    'iq_gain_mismatch': (0.95, 1.05),   # IQ kazanç: ±5%
    'iq_phase_err_deg': (-3.0, 3.0),    # IQ faz hatası: ±3°
    'dc_offset': (-5e-3, 5e-3),         # DC offset: ±5mV (normalize edilmiş)
    'temp_range_c': (5.0, 15.0),        # Sıcaklık değişimi: 5-15°C
    'velocity_ms': (0.0, 30.0),         # Hız: 0-108 km/h (30 m/s)
    'pa_backoff_db': (2.0, 8.0),        # PA backoff: 2-8dB
    'snr_db': (5, 25),                  # SNR aralığı: 5-25dB (daha dar ve gerçekçi)
}

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
    
    # DÜZELTME: torch.tensor kullan
    h_sum_squared = torch.sum(h**2) + torch.tensor(1e-12, device=device, dtype=torch.float32)
    h = h / torch.sqrt(h_sum_squared)
    return h

def gaussian_pulse(BT: float, sps: int, span: int, device):
    L = span*sps + 1
    t = torch.linspace(-span/2, span/2, L, device=device, dtype=torch.float32)
    sigma_sym = 0.32/ max(BT, 1e-3)
    g = torch.exp(-0.5*(t/sigma_sym)**2)
    
    # DÜZELTME: torch.tensor kullan
    g_sum = g.sum() + torch.tensor(1e-12, device=device, dtype=torch.float32)
    g = g / g_sum
    return g

def resample_linear(x: torch.Tensor, factor: float):
    N = x.shape[0]
    new_N = max(1, int(N * factor))  # En az 1 sample garantisi
    
    if new_N == N:
        return x
    
    # Orijinal ve yeni indeksler
    old_indices = torch.linspace(0, N-1, new_N, device=x.device, dtype=torch.float32)
    old_indices = torch.clamp(old_indices, 0.0, N-1.001)  # Sınırları kontrol et
    
    idx0 = old_indices.floor().to(torch.long)
    idx1 = torch.clamp(idx0 + 1, max=N-1)
    frac = (old_indices - idx0.to(torch.float32))
    
    if x.dtype == torch.complex64:
        # Complex tensör için ayrı ayrı interpolasyon
        real_interp = (1-frac) * x.real[idx0] + frac * x.real[idx1]
        imag_interp = (1-frac) * x.imag[idx0] + frac * x.imag[idx1]
        return (real_interp + 1j * imag_interp).to(torch.complex64)
    else:
        # Real tensör için
        return (1-frac) * x[idx0] + frac * x[idx1]

# ---------- SPEKTRAL ÖLÇÜM YARDIMCILARI ----------
def _next_pow2(n: int):
    p = 1
    while p < n: p <<= 1
    return p

def measure_spectral_metrics(iq: torch.Tensor, fs: float, p_occ: float = 0.99):
    device = iq.device
    N = iq.numel()
    
    # Eğer tensor çok kısaysa, minimum değerleri dön
    if N < 8:
        return 0.0, fs/2, fs/4, fs/8
    
    win = torch.hann_window(N, device=device, dtype=torch.float32)
    x = iq * win.to(iq.dtype)
    nfft = _next_pow2(int(2*N))
    X = torch.fft.fft(x, n=nfft)
    X = torch.fft.fftshift(X)
    P = (X.real**2 + X.imag**2).to(torch.float32) + 1e-30
    freqs = torch.linspace(-fs/2, fs/2, steps=nfft, device=device, dtype=torch.float32)
    Psum = torch.sum(P)
    
    if Psum < 1e-20:
        return 0.0, fs/2, fs/4, fs/8
    
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

# ---------- YENİ SPEKTRAL ÖZELLİKLER ----------
def compute_spectral_entropy(iq: torch.Tensor, n_fft: int = 256) -> float:
    """Spektral entropi hesaplar - karışık sinyallerin karmaşıklık ölçüsü"""
    device = iq.device
    
    if len(iq) < 8:
        return 0.0
    
    X = torch.fft.fft(iq, n=min(n_fft, len(iq)))
    P = torch.abs(X)**2
    P_norm = P / (torch.sum(P) + 1e-12)
    # Shannon entropisi (bit cinsinden)
    log_p = torch.log2(P_norm + 1e-12)
    entropy = -torch.sum(P_norm * log_p)
    return float(entropy.item())

def compute_freq_domain_variance(iq: torch.Tensor, fs: float) -> float:
    """Frekans domain'de varyans - spektral enerji yayılımının ölçüsü"""
    device = iq.device
    N = len(iq)
    
    if N < 8:
        return 0.0
    
    freqs = torch.fft.fftfreq(N, 1/fs, device=device)
    X = torch.fft.fft(iq)
    P = torch.abs(X)**2
    P_norm = P / (torch.sum(P) + 1e-12)
    
    # Frekans ağırlıklı merkez
    f_mean = torch.sum(freqs * P_norm)
    # Frekans varyansı
    f_var = torch.sum(P_norm * (freqs - f_mean)**2)
    return float(f_var.item())

def compute_spectral_kurtosis(iq: torch.Tensor) -> float:
    """Spektral kurtosis - spektral dağılımın sivrilik ölçüsü"""
    if len(iq) < 8:
        return 0.0
        
    X = torch.fft.fft(iq)
    P = torch.abs(X)**2
    P_norm = P / (torch.sum(P) + 1e-12)
    
    # Moments
    m1 = torch.sum(P_norm)  # Should be 1
    m2 = torch.sum(P_norm**2)
    m3 = torch.sum(P_norm**3)
    m4 = torch.sum(P_norm**4)
    
    # Kurtosis (excess kurtosis)
    if m2 > 1e-12:
        kurt = (m4 / (m2**2)) - 3.0
    else:
        kurt = 0.0
    
    return float(kurt.item())

# ==========================================================
# NİHAİ IQ ÜRETİCİ VE TÜM BOZULMA KÜTÜPHANESİ
# ==========================================================
class UltimateTorchIQGenerator:
    def __init__(self, fs=FS, duration=DURATION, device=None):
        self.fs = fs
        self.duration = duration
        self.N = int(round(fs*duration))
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.t = torch.arange(self.N, device=self.device, dtype=torch.float32) / fs
        self.pi = math.pi

    # ---- BÖLÜM 1: VERİCİ (TX) VE DONANIM BOZULMALARI - FİZİKSEL LİMİTLERLE ----

    def apply_pa_nonlinearity(self, iq: torch.Tensor, backoff_db=None) -> torch.Tensor:
        """Rapp modeli (AM/AM) ve basit AM/PM ile PA simülasyonu - Fiziksel limitlerle"""
        if backoff_db is None:
            backoff_db = random.uniform(*RF_LIMITS['pa_backoff_db'])
        
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
        am_pm_coeff = self.pi / 6.0  # Doyma noktasında 30 derece kayma (daha gerçekçi)
        phase_shift = am_pm_coeff * (envelope**2) / (1 + envelope**2)
        
        # Orijinal ölçeğe geri dön
        return (envelope_out * torch.exp(1j * (phase + phase_shift)) * (norm_factor / scale_lin)).to(torch.complex64)

    def apply_memory_effects(self, iq: torch.Tensor, memory_depth=2) -> torch.Tensor:
        """Basit Volterra/Hafıza Polinomu modeli - Daha az agresif"""
        # Gerçekçi olmayan ama temsili katsayılar - daha düşük etkiler
        a = torch.tensor([
            [0.98 + 0.005j, -0.02 + 0.01j],  # k=1 (linear term + memory)
            [0.01 - 0.005j, 0.002 - 0.001j], # k=3 (nonlinear term + memory)
        ], device=self.device, dtype=torch.complex64)
        
        output = torch.zeros_like(iq)
        iq_abs_sq = torch.abs(iq)**2

        # k=1 (lineer terimler)
        output += a[0,0] * iq
        for m in range(1, memory_depth):
            delayed = torch.roll(iq, shifts=m)
            delayed[:m] = 0 # Sıfır doldurma
            output += a[0,1] * delayed # Basitleştirilmiş katsayı

        # k=3 (non-lineer terimler) - daha az etki
        output += a[1,0] * iq * iq_abs_sq
        for m in range(1, memory_depth):
            delayed = torch.roll(iq, shifts=m)
            delayed[:m] = 0
            output += a[1,1] * delayed * iq_abs_sq # Basitleştirilmiş
        
        return output.to(torch.complex64)

    def apply_lo_phase_noise(self, iq: torch.Tensor, pn_profile='1/f3') -> torch.Tensor:
        """Spektral şekilli gerçekçi LO Faz Gürültüsü - Daha düşük seviye"""
        N = len(iq)
        noise = torch.randn(N, device=self.device, dtype=torch.float32)
        freqs = torch.fft.fftfreq(N, 1/self.fs, device=self.device)
        freqs[0] = 1e-12 # DC'de sıfıra bölmeyi engelle

        if pn_profile == '1/f3':
            H = 1.0 / (torch.abs(freqs)**1.5) # Güç 1/f^3 -> genlik 1/f^1.5
            pn_power_dbchz = -130  # 1kHz'de dBc/Hz (daha düşük, gerçekçi)
        else: # '1/f2' (Wiener/random walk benzeri)
            H = 1.0 / torch.abs(freqs) # Güç 1/f^2 -> genlik 1/f
            pn_power_dbchz = -120

        H = H * math.sqrt(10**(pn_power_dbchz/10))
        H[0] = 0 # DC bileşeni yok
        
        noise_fft = torch.fft.fft(noise)
        filtered_fft = noise_fft * H.to(torch.complex64)
        phase_noise_deriv = torch.fft.ifft(filtered_fft).real
        
        phase = torch.cumsum(phase_noise_deriv, dim=0) * 2 * math.pi / self.fs
        return (iq * torch.exp(1j * phase)).to(torch.complex64)

    def apply_iq_imbalance_complete(self, iq: torch.Tensor):
        """IQ Dengesizliği - Fiziksel limitlerle"""
        I = iq.real; Q = iq.imag
        gI = float(torch.normal(1.0, 0.02, size=(1,), device=self.device))  # Daha düşük varyans
        gQ = float(torch.normal(1.0, 0.02, size=(1,), device=self.device))
        # Fiziksel limitler içinde tut
        gI = np.clip(gI, *RF_LIMITS['iq_gain_mismatch'])
        gQ = np.clip(gQ, *RF_LIMITS['iq_gain_mismatch'])
        
        eps_deg = float(torch.normal(0.0, 1.0, size=(1,), device=self.device))
        eps_deg = np.clip(eps_deg, *RF_LIMITS['iq_phase_err_deg'])
        eps_rad = math.radians(eps_deg)
        
        dc_i = float(torch.normal(0.0, 1e-3, size=(1,), device=self.device))
        dc_q = float(torch.normal(0.0, 1e-3, size=(1,), device=self.device))
        dc_i = np.clip(dc_i, *RF_LIMITS['dc_offset'])
        dc_q = np.clip(dc_q, *RF_LIMITS['dc_offset'])
        
        I_out = gI * I + dc_i
        Q_out = gQ * (Q*math.cos(eps_rad) + I*math.sin(eps_rad)) + dc_q
        
        out = (I_out + 1j*Q_out).to(torch.complex64)
        meta = dict(I_gain=gI, Q_gain=gQ, iq_phase_err_deg=eps_deg, dc=(dc_i, dc_q))
        return out, meta
        
    def apply_cfo_and_slow_doppler(self, iq: torch.Tensor):
        """CFO ve yavaş Doppler - Fiziksel limitlerle"""
        cfo0 = float(torch.empty(1, device=self.device).uniform_(*RF_LIMITS['cfo_hz']))
        cfo_slope = float(torch.empty(1, device=self.device).uniform_(*RF_LIMITS['cfo_slope_hz_s']))
        
        t = self.t
        cfo_t = cfo0 + cfo_slope * t
        phase = 2*self.pi * torch.cumsum(cfo_t / self.fs, dim=0)
        return (iq * torch.exp(1j*phase)).to(torch.complex64), dict(cfo0_hz=cfo0, cfo_slope_hz_s=cfo_slope)

    def apply_clock_drift(self, iq: torch.Tensor):
        """Saat sapması - Fiziksel limitlerle"""
        drift_ppm = float(torch.empty(1, device=self.device).uniform_(*RF_LIMITS['clock_drift_ppm']))
        factor = 1.0 + drift_ppm * 1e-6
        iq2 = resample_linear(iq, factor=factor)
        
        # Boyut uyumluluğu için kırp veya pad
        if len(iq2) > len(iq):
            iq2 = iq2[:len(iq)]
        elif len(iq2) < len(iq):
            padding = torch.zeros(len(iq) - len(iq2), dtype=iq2.dtype, device=iq2.device)
            iq2 = torch.cat([iq2, padding])
            
        return iq2, dict(clock_drift_ppm=drift_ppm)

    # ---- BÖLÜM 2: KANAL MODELİ - DAHA GERÇEKÇİ PARAMETRELERLE ----

    def apply_multipath_realistic(self, iq: torch.Tensor, 
                                 environment: str, velocity_ms: float) -> torch.Tensor:
        """Geometrik, fizik-tabanlı çok yollu kanal (Ricean/Rayleigh + Jakes Doppler)"""
        
        # Hızı fiziksel limitlere sınırla
        velocity_ms = np.clip(velocity_ms, *RF_LIMITS['velocity_ms'])
        
        if environment == 'urban':
            delays_ns = [0, 50, 100, 300, 800]  # Daha kısa gecikmeler
            powers_db = [0, -2, -4, -10, -15]   # Daha yumuşak güç profili
            angles_deg = [0, 10, -20, 30, -45]
            k_factor_db = 2  # Ricean K-faktörü (LOS zayıf ama var)
        elif environment == 'suburban':
            delays_ns = [0, 30, 100, 400]
            powers_db = [0, -3, -8, -16]
            angles_deg = [0, 15, -25, 35]
            k_factor_db = 5
        else:  # 'rural'
            delays_ns = [0, 20, 200]
            powers_db = [0, -5, -18]
            angles_deg = [0, 8, -12]
            k_factor_db = 8 # Ricean K-faktörü (LOS güçlü)
        
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
                los_component = math.sqrt(float(los_power)) * torch.exp(1j * doppler_phase)
                
                # NLOS (dağınık) bileşen (Jakes spektrumlu Rayleigh)
                rayleigh_fading = self._generate_rayleigh_fading(max_fd, self.N)
                nlos_component = math.sqrt(float(nlos_power)) * rayleigh_fading
                
                channel_tap = los_component + nlos_component
            else:
                # YANSIMA YOLLARI (NLOS): Sadece Rayleigh Fading
                rayleigh_fading = self._generate_rayleigh_fading(max_fd, self.N)
                channel_tap = math.sqrt(float(power_norm)) * rayleigh_fading

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
        N_filt = 128  # Daha küçük filtre (hesaplama tasarrufu)
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

    # ---- BÖLÜM 3: ALICI (RX) VE DIŞ BOZULMALAR - GERÇEKÇİ SEVİYELERLE ----

    def add_awgn_total(self, iq_mix: torch.Tensor, snr_db: float):
        """Alıcının girişindeki termal gürültü (AWGN) - Fiziksel SNR limitleri"""
        snr_db = np.clip(snr_db, *RF_LIMITS['snr_db'])
        sp = torch.mean(torch.abs(iq_mix)**2) + 1e-20
        npow = sp / db2lin(snr_db)
        n = torch.sqrt(npow/2) * (torch.randn(self.N, device=self.device) + 1j*torch.randn(self.N, device=self.device))
        return (iq_mix + n.to(torch.complex64)).to(torch.complex64)

    def apply_thermal_drift(self, iq: torch.Tensor, temp_range_c=None) -> torch.Tensor:
        """Alıcı donanımının sıcaklığa bağlı kayması - Fiziksel limitlerle"""
        if temp_range_c is None:
            temp_range_c = random.uniform(*RF_LIMITS['temp_range_c'])
        
        temp_freq_hz = 0.05  # Daha yavaş sıcaklık değişimi (gerçekçi)
        temp_variation = temp_range_c * torch.sin(2 * math.pi * temp_freq_hz * self.t)
        
        # Daha gerçekçi katsayılar
        gain_temp_coeff = -0.005  # -0.005 dB/°C (tipik RF donanım)
        phase_temp_coeff = 0.2    # 0.2°/°C  
        freq_temp_coeff = -1.0    # -1.0 ppm/°C (kristal)
        
        gain_drift_db = gain_temp_coeff * temp_variation
        phase_drift_deg = phase_temp_coeff * temp_variation
        freq_drift_ppm = freq_temp_coeff * temp_variation
        
        gain_drift_lin = 10**(gain_drift_db / 20.0)
        phase_drift_rad = phase_drift_deg * math.pi / 180.0
        freq_drift_hz = freq_drift_ppm * 1e-6 * self.fs
        
        # Frekans kaymasını faz olarak uygula
        phase_thermal = torch.cumsum(2 * math.pi * freq_drift_hz / self.fs, dim=0)
        
        return (iq * gain_drift_lin * torch.exp(1j * (phase_drift_rad + phase_thermal))).to(torch.complex64)

    def apply_agc_response(self, iq: torch.Tensor, target_power_db=-10, time_constant_ms=10) -> torch.Tensor:
        """Otomatik Kazanç Kontrolü (AGC) simülasyonu - Gerçekçi zaman sabiti"""
        target_power_lin = 10**(target_power_db/10.0)
        tc_samples = int(time_constant_ms * 1e-3 * self.fs)
        alpha = 1.0 / tc_samples  # Eksponansiyel filtre katsayısı
        
        inst_power = torch.abs(iq)**2
        
        # Güç takipçisi (düşük geçiren filtre)
        filtered_power = torch.zeros_like(inst_power)
        filtered_power[0] = inst_power[0]
        for i in range(1, len(inst_power)):
            filtered_power[i] = (1-alpha) * filtered_power[i-1] + alpha * inst_power[i]
        
        # AGC kazancı
        agc_gain = torch.sqrt(target_power_lin / (filtered_power + 1e-12))
        
        return (iq * agc_gain).to(torch.complex64)

    # ---- BÖLÜM 4: MODÜLASYON ÜRETİCİLERİ ----

    def _get_random_tx_scenario(self):
        """Gerçekçi verici senaryoları"""
        scenarios = ['stationary', 'slow_mobile', 'fast_mobile', 'indoor', 'outdoor']
        return random.choice(scenarios)

    def gen_fm(self, power_db: float, tx_scenario: str = 'stationary'):
        """FM sinyali üretir - Gerçekçi parametrelerle"""
        # FM parametreleri - kanaldan bağımsız
        fc_dev = random.uniform(10e3, 75e3)  # Frekans sapması: 10-75 kHz
        f_mod = random.uniform(100, 15000)   # Modülasyon frekansı: 100Hz-15kHz
        
        # Modülasyon sinyali
        mod_signal = torch.sin(2 * math.pi * f_mod * self.t)
        
        # Pre-emphasis (tipik FM)
        if random.random() > 0.3:
            tau = 75e-6  # 75µs pre-emphasis
            s = 2 * math.pi * torch.fft.fftfreq(self.N, 1/self.fs, device=self.device)
            H = (1 + 1j * s * tau) / (1 + 1j * s * tau / 4)  # Basit pre-emphasis
            mod_fft = torch.fft.fft(mod_signal)
            mod_signal = torch.fft.ifft(mod_fft * H).real
        
        # FM modülasyonu
        phase = torch.cumsum(2 * math.pi * fc_dev * mod_signal / self.fs, dim=0)
        iq = torch.exp(1j * phase).to(torch.complex64)
        
        # Güç ayarı - DÜZELTİLEN KISIM
        power_factor = math.sqrt(10**(power_db/10.0))  # math.sqrt kullan, torch.sqrt değil
        iq = iq * power_factor
        
        # Verici bozulmaları - senaryo bazlı
        if tx_scenario in ['slow_mobile', 'fast_mobile']:
            velocity = 5.0 if tx_scenario == 'slow_mobile' else 20.0
            iq, _ = self.apply_multipath_realistic(iq, 'urban', velocity)
        
        # Donanım bozulmaları
        if random.random() > 0.2:
            iq = self.apply_pa_nonlinearity(iq)
        if random.random() > 0.4:
            iq, _ = self.apply_iq_imbalance_complete(iq)
        if random.random() > 0.3:
            iq, _ = self.apply_cfo_and_slow_doppler(iq)
        
        meta = {
            'fc_dev_hz': fc_dev,
            'f_mod_hz': f_mod,
            'tx_scenario': tx_scenario,
            'power_db': power_db
        }
        
        return iq, meta

    def gen_qpsk(self, power_db: float, tx_scenario: str = 'stationary'):
        """QPSK sinyali üretir - Gerçekçi parametrelerle"""
        # QPSK parametreleri
        sym_rate = random.uniform(50e3, 500e3)  # Sembol hızı: 50-500 kbaud
        sps = max(2, int(self.fs / sym_rate))   # Örneklem/sembol (min 2)
        beta = random.uniform(0.2, 0.5)        # Roll-off faktörü
        
        # Sembol sayısı
        n_syms = max(10, self.N // sps)
        
        # QPSK sembolleri (Gray kodlama)
        qpsk_syms = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j], device=self.device) / math.sqrt(2)
        sym_indices = torch.randint(0, 4, (n_syms,), device=self.device)
        symbols = qpsk_syms[sym_indices]
        
        # Upsampling
        upsampled = torch.zeros(n_syms * sps, dtype=torch.complex64, device=self.device)
        upsampled[::sps] = symbols
        
        # Pulse shaping (Root-Raised Cosine)
        span = 8  # Filtre uzunluğu (sembol cinsinden)
        rrc_filter = rc_impulse_response(beta, sps, span, self.device)
        
        # Konvolüsyon ve kırpma
        filtered = F.conv1d(upsampled.view(1, 1, -1), rrc_filter.view(1, 1, -1).to(torch.complex64), padding=len(rrc_filter)//2)
        filtered = filtered.squeeze()
        
        # İstenilen uzunluğa kırp veya pad
        if len(filtered) > self.N:
            filtered = filtered[:self.N]
        elif len(filtered) < self.N:
            padding = torch.zeros(self.N - len(filtered), dtype=torch.complex64, device=self.device)
            filtered = torch.cat([filtered, padding])
        
        # Güç normalize et - DÜZELTİLEN KISIM
        power_factor = math.sqrt(10**(power_db/10.0))  # math.sqrt kullan
        mean_power = torch.mean(torch.abs(filtered)**2) + 1e-12
        iq = filtered * power_factor / torch.sqrt(mean_power)  # burada torch.sqrt OK çünkü tensor
        
        # Verici bozulmaları
        if tx_scenario in ['slow_mobile', 'fast_mobile']:
            velocity = 8.0 if tx_scenario == 'slow_mobile' else 25.0
            iq, _ = self.apply_multipath_realistic(iq, 'suburban', velocity)
        
        # Donanım bozulmaları
        if random.random() > 0.15:
            iq = self.apply_pa_nonlinearity(iq)
        if random.random() > 0.25:
            iq, _ = self.apply_iq_imbalance_complete(iq)
        if random.random() > 0.2:
            iq, _ = self.apply_cfo_and_slow_doppler(iq)
        
        meta = {
            'sym_rate_hz': sym_rate,
            'sps': sps,
            'beta': beta,
            'tx_scenario': tx_scenario,
            'power_db': power_db
        }
        
        return iq, meta

    def gen_gfsk(self, power_db: float, tx_scenario: str = 'stationary'):
        """GFSK sinyali üretir - Gerçekçi parametrelerle"""
        # GFSK parametreleri
        bit_rate = random.uniform(10e3, 250e3)  # Bit hızı: 10-250 kbps
        sps = max(4, int(self.fs / bit_rate))   # Örneklem/bit (min 4 for GFSK)
        BT = random.uniform(0.3, 0.5)          # Gaussian filtre BT
        h = random.uniform(0.3, 0.7)           # Modülasyon indeksi
        
        # Bit sayısı
        n_bits = max(20, self.N // sps)
        
        # Random bits
        bits = torch.randint(0, 2, (n_bits,), device=self.device, dtype=torch.float32) * 2 - 1  # ±1
        
        # Upsampling
        upsampled = torch.zeros(n_bits * sps, dtype=torch.float32, device=self.device)
        upsampled[::sps] = bits
        
        # Gaussian pulse shaping
        span = 4  # Pulse uzunluğu (bit cinsinden)
        g_filter = gaussian_pulse(BT, sps, span, self.device)
        
        # Filtering
        filtered = F.conv1d(upsampled.view(1, 1, -1), g_filter.view(1, 1, -1), padding=len(g_filter)//2)
        filtered = filtered.squeeze()
        
        # İstenilen uzunluğa kırp veya pad
        if len(filtered) > self.N:
            filtered = filtered[:self.N]
        elif len(filtered) < self.N:
            padding = torch.zeros(self.N - len(filtered), dtype=torch.float32, device=self.device)
            filtered = torch.cat([filtered, padding])
        
        # Frequency modulation
        fd = h * bit_rate / 2  # Frekans sapması
        phase = torch.cumsum(2 * math.pi * fd * filtered / self.fs, dim=0)
        iq = torch.exp(1j * phase).to(torch.complex64)
        
        # Güç ayarı - DÜZELTİLEN KISIM
        power_factor = math.sqrt(10**(power_db/10.0))  # math.sqrt kullan
        iq = iq * power_factor
        
        # Verici bozulmaları
        if tx_scenario in ['slow_mobile', 'fast_mobile']:
            velocity = 6.0 if tx_scenario == 'slow_mobile' else 18.0
            iq, _ = self.apply_multipath_realistic(iq, 'suburban', velocity)
        
        # Donanım bozulmaları (GFSK genelde düşük güçlü, az bozulma)
        if random.random() > 0.4:
            iq = self.apply_pa_nonlinearity(iq, backoff_db=random.uniform(4, 10))
        if random.random() > 0.3:
            iq, _ = self.apply_iq_imbalance_complete(iq)
        if random.random() > 0.25:
            iq, _ = self.apply_cfo_and_slow_doppler(iq)
        
        meta = {
            'bit_rate_hz': bit_rate,
            'sps': sps,
            'BT': BT,
            'h': h,
            'fd_hz': fd,
            'tx_scenario': tx_scenario,
            'power_db': power_db
        }
        
        return iq, meta

    def gen_ofdm(self, power_db: float, tx_scenario: str = 'stationary'):
        """OFDM sinyali üretir - Gerçekçi parametrelerle"""
        # OFDM parametreleri
        n_fft = random.choice([64, 128, 256, 512])  # FFT boyutu
        n_used = int(n_fft * random.uniform(0.7, 0.9))  # Kullanılan alt taşıyıcı sayısı
        cp_len = int(n_fft * random.uniform(0.1, 0.25))  # Cyclic prefix uzunluğu
        
        # OFDM sembolleri sayısı
        symbol_len = n_fft + cp_len
        n_ofdm_symbols = max(2, self.N // symbol_len)
        
        # Alt taşıyıcı haritası (merkezi kullan, kenarları boş bırak)
        used_indices = torch.arange(-(n_used//2), n_used//2 + 1, device=self.device)
        used_indices = used_indices[used_indices != 0] + n_fft//2  # DC'yi atla
        
        iq_total = torch.zeros(0, dtype=torch.complex64, device=self.device)
        
        for sym_idx in range(n_ofdm_symbols):
            # QAM sembolleri (QPSK/16-QAM karışımı)
            if random.random() > 0.5:  # QPSK
                constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j], device=self.device) / math.sqrt(2)
            else:  # 16-QAM
                constellation = torch.tensor([
                    -3-3j, -3-1j, -3+1j, -3+3j,
                    -1-3j, -1-1j, -1+1j, -1+3j,
                    +1-3j, +1-1j, +1+1j, +1+3j,
                    +3-3j, +3-1j, +3+1j, +3+3j
                ], device=self.device) / math.sqrt(10)
            
            # Random sembolleri seç
            data_indices = torch.randint(0, len(constellation), (len(used_indices),), device=self.device)
            data_symbols = constellation[data_indices]
            
            # IFFT girişi hazırla
            ifft_input = torch.zeros(n_fft, dtype=torch.complex64, device=self.device)
            ifft_input[used_indices] = data_symbols
            
            # IFFT
            time_domain = torch.fft.ifft(ifft_input) * math.sqrt(n_fft)  # Normalizasyon
            
            # Cyclic prefix ekle
            cp = time_domain[-cp_len:]
            ofdm_symbol = torch.cat([cp, time_domain])
            
            iq_total = torch.cat([iq_total, ofdm_symbol])
        
        # İstenilen uzunluğa kırp veya sıfırla doldur
        if len(iq_total) > self.N:
            iq = iq_total[:self.N]
        else:
            padding = torch.zeros(self.N - len(iq_total), dtype=torch.complex64, device=self.device)
            iq = torch.cat([iq_total, padding])
        
        # Güç normalize et - DÜZELTİLEN KISIM
        power_factor = math.sqrt(10**(power_db/10.0))  # math.sqrt kullan
        mean_power = torch.mean(torch.abs(iq)**2) + 1e-12
        iq = iq * power_factor / torch.sqrt(mean_power)  # burada torch.sqrt OK çünkü tensor
        
        # OFDM'e özel: PAPR azaltma (clipping) - SADECE MAGNITUDE'DA
        if random.random() > 0.3:
            papr_threshold_db = random.uniform(6, 10)  # PAPR eşiği
            mean_power = torch.mean(torch.abs(iq)**2) + 1e-12
            clip_level_squared = mean_power * 10**(papr_threshold_db/10)
            clip_level = torch.sqrt(clip_level_squared)
            magnitude = torch.abs(iq)
            phase = torch.angle(iq)
            # Sadece magnitude'u clip et
            clipped_magnitude = torch.clamp(magnitude, max=float(clip_level))
            iq = clipped_magnitude * torch.exp(1j * phase)
        
        # Verici bozulmaları
        if tx_scenario in ['slow_mobile', 'fast_mobile']:
            velocity = 10.0 if tx_scenario == 'slow_mobile' else 30.0
            iq, _ = self.apply_multipath_realistic(iq, 'urban', velocity)
        
        # Donanım bozulmaları (OFDM PAPR'a duyarlı)
        if random.random() > 0.1:  # PA non-linearity çok kritik
            iq = self.apply_pa_nonlinearity(iq, backoff_db=random.uniform(3, 8))
        if random.random() > 0.2:
            iq, _ = self.apply_iq_imbalance_complete(iq)
        if random.random() > 0.15:
            iq, _ = self.apply_cfo_and_slow_doppler(iq)
        if random.random() > 0.4:  # CFO OFDM için çok kritik
            iq = self.apply_lo_phase_noise(iq)
        
        meta = {
            'n_fft': n_fft,
            'n_used': n_used,
            'cp_len': cp_len,
            'n_ofdm_symbols': n_ofdm_symbols,
            'tx_scenario': tx_scenario,
            'power_db': power_db
        }
        
        return iq, meta


# ---- BÖLÜM 5: ANA ÜRETİM FONKSİYONLARI ----

def _get_random_tx_scenario():
    """Gerçekçi verici senaryoları"""
    scenarios = ['stationary', 'slow_mobile', 'fast_mobile', 'indoor', 'outdoor']
    weights = [0.3, 0.25, 0.15, 0.15, 0.15]  # Stationary daha yaygın
    return random.choices(scenarios, weights=weights)[0]

def gen_single(gen: UltimateTorchIQGenerator):
    mod = random.choice(MODS)
    p_db = random.uniform(-15, 0)  # Daha dar güç aralığı
    scenario = _get_random_tx_scenario()
    
    if   mod=='FM':   iq, meta = gen.gen_fm(p_db, tx_scenario=scenario)
    elif mod=='OFDM': iq, meta = gen.gen_ofdm(p_db, tx_scenario=scenario)
    elif mod=='GFSK': iq, meta = gen.gen_gfsk(p_db, tx_scenario=scenario)
    else:             iq, meta = gen.gen_qpsk(p_db, tx_scenario=scenario)

    # Spektral metrikleri hesapla
    f_center_est, bw_occ, bw_rms, bw_3db = measure_spectral_metrics(iq, gen.fs, p_occ=0.99)
    spectral_entropy = compute_spectral_entropy(iq, n_fft=256)
    freq_variance = compute_freq_domain_variance(iq, gen.fs)
    spectral_kurtosis = compute_spectral_kurtosis(iq)
    
    info = [{
        **meta, "mod": mod, "f_off_hz": 0.0, "rel_power_db": 0.0,
        "f_center_est_hz": f_center_est, "bw_occ99_hz": bw_occ,
        "bw_rms_hz": bw_rms, "bw_3db_hz": bw_3db,
        "spectral_entropy": spectral_entropy,
        "freq_domain_variance": freq_variance,
        "spectral_kurtosis": spectral_kurtosis
    }]
    return iq, info

def gen_mixed(gen: UltimateTorchIQGenerator, close=True):
    k = random.randint(2, MAX_SIGNAL)
    # Daha gerçekçi güç dağılımı (ana sinyal güçlü, diğerleri zayıf)
    main_power = random.uniform(-5, 0)
    other_powers = [main_power + random.uniform(-15, -5) for _ in range(k-1)]
    rel_powers_db = [main_power] + sorted(other_powers, reverse=True)
    
    sigs, info = [], []
    for idx in range(k):
        mod = random.choice(MODS); p_db = rel_powers_db[idx]
        scenario = _get_random_tx_scenario()
        
        if   mod=='FM':   s, meta = gen.gen_fm(p_db, tx_scenario=scenario)
        elif mod=='OFDM': s, meta = gen.gen_ofdm(p_db, tx_scenario=scenario)
        elif mod=='GFSK': s, meta = gen.gen_gfsk(p_db, tx_scenario=scenario)
        else:             s, meta = gen.gen_qpsk(p_db, tx_scenario=scenario)

        if close: 
            foff = random.uniform(*CLOSE_OFFSET_HZ) * random.choice([-1, 1])
        else: 
            foff = random.uniform(*FAR_OFFSET_FRAC) * random.choice([-1, 1]) * gen.fs
        
        s = (s * torch.exp(1j*2*math.pi*foff*gen.t)).to(torch.complex64)
        
        # Her sinyal için spektral metrikleri hesapla
        f_center_est, bw_occ, bw_rms, bw_3db = measure_spectral_metrics(s, gen.fs, p_occ=0.99)
        spectral_entropy = compute_spectral_entropy(s, n_fft=256)
        freq_variance = compute_freq_domain_variance(s, gen.fs)
        spectral_kurtosis = compute_spectral_kurtosis(s)
        
        row = {
            **meta, "mod": mod, "f_off_hz": float(foff), "rel_power_db": float(p_db),
            "f_center_theory_hz_approx": meta.get("f_center_theory_hz_approx", 0.0) + float(foff),
            "f_center_est_hz": f_center_est, "bw_occ99_hz": bw_occ,
            "bw_rms_hz": bw_rms, "bw_3db_hz": bw_3db,
            "spectral_entropy": spectral_entropy,
            "freq_domain_variance": freq_variance,
            "spectral_kurtosis": spectral_kurtosis
        }
        info.append(row); sigs.append(s)
    
    iq_mixed = sum(sigs)
    return iq_mixed, info

def gen_noise_only(gen: UltimateTorchIQGenerator):
    """Sadece gürültü"""
    noise_power_db = random.uniform(-25, -5)
    # DÜZELTİLEN KISIM: math.sqrt kullan
    noise = math.sqrt(10**(noise_power_db/10.0)/2) * (
        torch.randn(gen.N, device=gen.device) + 1j*torch.randn(gen.N, device=gen.device)
    )
    
    # Spektral metrikleri hesapla
    f_center_est, bw_occ, bw_rms, bw_3db = measure_spectral_metrics(noise, gen.fs, p_occ=0.99)
    spectral_entropy = compute_spectral_entropy(noise, n_fft=256)
    freq_variance = compute_freq_domain_variance(noise, gen.fs)
    spectral_kurtosis = compute_spectral_kurtosis(noise)
    
    info = [{
        "mod": "NOISE", "f_off_hz": 0.0, "rel_power_db": noise_power_db,
        "f_center_est_hz": f_center_est, "bw_occ99_hz": bw_occ,
        "bw_rms_hz": bw_rms, "bw_3db_hz": bw_3db,
        "spectral_entropy": spectral_entropy,
        "freq_domain_variance": freq_variance,
        "spectral_kurtosis": spectral_kurtosis,
        "power_db": noise_power_db, "tx_scenario": "noise_only"
    }]
    
    return noise.to(torch.complex64), info

# ---- BÖLÜM 6: STFT VE VERİ KAYDETME ----

def compute_stft_torch(iq: torch.Tensor, fs: float, nperseg=N_FFT, noverlap=N_OVERLAP):
    """PyTorch tabanlı STFT hesaplama"""
    device = iq.device
    
    # Hann penceresi
    window = torch.hann_window(nperseg, device=device, dtype=torch.float32)
    
    # STFT hesapla
    stft_result = torch.stft(
        iq, n_fft=nperseg, hop_length=nperseg-noverlap, 
        win_length=nperseg, window=window,
        return_complex=True, normalized=False
    )
    
    # Magnitude spektrogram (dB)
    magnitude = torch.abs(stft_result)
    magnitude_db = 20 * torch.log10(magnitude + 1e-12)
    
    # Frekans ve zaman eksenleri
    freqs = torch.fft.fftfreq(nperseg, 1/fs, device=device)
    freqs = torch.fft.fftshift(freqs)  # DC'yi merkeze al
    
    times = torch.arange(stft_result.shape[1], device=device, dtype=torch.float32) * (nperseg - noverlap) / fs
    
    # Spektrogramı fftshift ile düzenle
    magnitude_db_shifted = torch.fft.fftshift(magnitude_db, dim=0)
    
    return to_cpu_np(magnitude_db_shifted), to_cpu_np(freqs), to_cpu_np(times)

def create_shard_filename(base_dir: str, shard_idx: int) -> str:
    """Shard dosya adı oluştur"""
    return os.path.join(base_dir, f"shard_{shard_idx:04d}.pkl")

def save_shard(data: List[Dict], base_dir: str, shard_idx: int):
    """Shard'ı kaydet"""
    filepath = create_shard_filename(base_dir, shard_idx)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# ---- BÖLÜM 7: ANA ÜRETİM DÖNGÜSÜ ----

def main():
    set_seed(SEED)
    print(f"UHF Hyper-Realistic Dataset Generator V12")
    print(f"Target: {NUM_SAMPLES:,} samples")
    print(f"Output: {OUT_DIR}")
    
    # Çıkış dizinini oluştur
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Cihaz seçimi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Generator
    gen = UltimateTorchIQGenerator(fs=FS, duration=DURATION, device=device)
    
    # Oranları hesapla
    n_noise = int(NUM_SAMPLES * PROPORTIONS["noise"])
    n_single = int(NUM_SAMPLES * PROPORTIONS["single"])
    n_mixed_close = int(NUM_SAMPLES * PROPORTIONS["mixed_close"])
    n_mixed_far = NUM_SAMPLES - n_noise - n_single - n_mixed_close
    
    print(f"Distribution: Noise={n_noise}, Single={n_single}, Mixed_Close={n_mixed_close}, Mixed_Far={n_mixed_far}")
    
    # Üretim listesi
    generation_tasks = (
        [("noise", i) for i in range(n_noise)] +
        [("single", i) for i in range(n_single)] +
        [("mixed_close", i) for i in range(n_mixed_close)] +
        [("mixed_far", i) for i in range(n_mixed_far)]
    )
    
    # Karıştır
    random.shuffle(generation_tasks)
    
    # Shard değişkenleri
    current_shard = []
    shard_idx = 0
    
    # İstatistikler
    mod_counts = {mod: 0 for mod in MODS + ["NOISE"]}
    scenario_counts = {}
    
    # Üretim döngüsü
    pbar = tqdm(generation_tasks, desc="Generating")
    
    for sample_idx, (sample_type, _) in enumerate(pbar):
        try:
            # Sinyal üret
            if sample_type == "noise":
                iq, meta_list = gen_noise_only(gen)
            elif sample_type == "single":
                iq, meta_list = gen_single(gen)
            elif sample_type == "mixed_close":
                iq, meta_list = gen_mixed(gen, close=True)
            else:  # mixed_far
                iq, meta_list = gen_mixed(gen, close=False)
            
            # Son bozulmalar uygula (alıcı tarafı)
            if random.random() > 0.3:
                snr_db = random.uniform(*RF_LIMITS['snr_db'])
                iq = gen.add_awgn_total(iq, snr_db)
            
            if random.random() > 0.4:
                iq = gen.apply_thermal_drift(iq)
            
            if random.random() > 0.6:
                iq = gen.apply_agc_response(iq)
            
            # STFT hesapla
            spectrogram, freqs, times = compute_stft_torch(iq, gen.fs, N_FFT, N_OVERLAP)
            
            # UHF metadata ekle
            uhf_carrier = random.uniform(UHF_MIN, UHF_MAX)
            
            # Sample metadata oluştur
            sample_data = {
                'spectrogram': spectrogram.astype(np.float32),
                'freqs': freqs.astype(np.float32),
                'times': times.astype(np.float32),
                'iq': to_cpu_np(iq).astype(np.complex64),
                'sample_id': sample_idx,
                'sample_type': sample_type,
                'fs': gen.fs,
                'duration': gen.duration,
                'uhf_carrier_hz': uhf_carrier,
                'timestamp': datetime.now().isoformat(),
                'signals': meta_list,
                'n_signals': len(meta_list)
            }
            
            # İstatistikleri güncelle
            for signal_meta in meta_list:
                mod = signal_meta.get('mod', 'UNKNOWN')
                mod_counts[mod] = mod_counts.get(mod, 0) + 1
                
                scenario = signal_meta.get('tx_scenario', 'unknown')
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
            
            # Shard'a ekle
            current_shard.append(sample_data)
            
            # Shard dolu mu kontrol et
            if len(current_shard) >= SHARD_SIZE:
                save_shard(current_shard, OUT_DIR, shard_idx)
                pbar.set_postfix({
                    'Shard': shard_idx,
                    'Progress': f"{sample_idx+1}/{NUM_SAMPLES}",
                    'Type': sample_type
                })
                current_shard = []
                shard_idx += 1
        
        except Exception as e:
            print(f"\nError at sample {sample_idx}: {e}")
            continue
    
    # Son shard'ı kaydet (eğer dolu değilse)
    if current_shard:
        save_shard(current_shard, OUT_DIR, shard_idx)
    
    # İstatistikleri kaydet
    stats = {
        'total_samples': NUM_SAMPLES,
        'total_shards': shard_idx + 1,
        'shard_size': SHARD_SIZE,
        'proportions': PROPORTIONS,
        'modulation_counts': mod_counts,
        'scenario_counts': scenario_counts,
        'rf_limits': RF_LIMITS,
        'fs': FS,
        'duration': DURATION,
        'n_fft': N_FFT,
        'n_overlap': N_OVERLAP,
        'uhf_range': [UHF_MIN, UHF_MAX],
        'generation_time': datetime.now().isoformat(),
        'seed': SEED
    }
    
    with open(os.path.join(OUT_DIR, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset generation completed!")
    print(f"Total samples: {NUM_SAMPLES:,}")
    print(f"Total shards: {shard_idx + 1}")
    print(f"Modulation distribution:")
    for mod, count in sorted(mod_counts.items()):
        print(f"   {mod}: {count:,}")
    
    print(f"Scenario distribution:")
    for scenario, count in sorted(scenario_counts.items()):
        print(f"   {scenario}: {count:,}")
    
    print(f"Dataset saved to: {OUT_DIR}")

# ---- BÖLÜM 8: YARDIMCI FONKSIYONLAR ----

def load_shard(base_dir: str, shard_idx: int) -> List[Dict]:
    """Shard'ı yükle"""
    filepath = create_shard_filename(base_dir, shard_idx)
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def get_dataset_info(base_dir: str) -> Dict:
    """Dataset bilgilerini al"""
    stats_path = os.path.join(base_dir, 'dataset_stats.json')
    with open(stats_path, 'r') as f:
        return json.load(f)

def list_shards(base_dir: str) -> List[str]:
    """Mevcut shard dosyalarını listele"""
    files = [f for f in os.listdir(base_dir) if f.startswith('shard_') and f.endswith('.pkl')]
    return sorted(files)

# ---- BÖLÜM 9: VERİ YÜKLEYİCİ (DATALOADER) ----

class UHFDatasetLoader:
    """UHF Dataset için PyTorch uyumlu data loader"""
    
    def __init__(self, base_dir: str, mode: str = 'spectrogram'):
        """
        Args:
            base_dir: Dataset dizini
            mode: 'spectrogram', 'iq', 'both'
        """
        self.base_dir = base_dir
        self.mode = mode
        self.stats = get_dataset_info(base_dir)
        self.shard_files = list_shards(base_dir)
        self.total_samples = self.stats['total_samples']
        
        # Index to shard mapping
        self._build_index()
    
    def _build_index(self):
        """Sample index'ten shard'a eşleme oluştur"""
        self.index_to_shard = {}
        sample_idx = 0
        
        for shard_idx, shard_file in enumerate(self.shard_files):
            shard_data = load_shard(self.base_dir, shard_idx)
            shard_size = len(shard_data)
            
            for local_idx in range(shard_size):
                self.index_to_shard[sample_idx] = (shard_idx, local_idx)
                sample_idx += 1
    
    def __len__(self):
        return len(self.index_to_shard)
    
    def __getitem__(self, idx):
        shard_idx, local_idx = self.index_to_shard[idx]
        shard_data = load_shard(self.base_dir, shard_idx)
        sample = shard_data[local_idx]
        
        if self.mode == 'spectrogram':
            return {
                'spectrogram': torch.from_numpy(sample['spectrogram']),
                'freqs': torch.from_numpy(sample['freqs']),
                'times': torch.from_numpy(sample['times']),
                'metadata': sample
            }
        elif self.mode == 'iq':
            return {
                'iq': torch.from_numpy(sample['iq']),
                'metadata': sample
            }
        else:  # both
            return {
                'spectrogram': torch.from_numpy(sample['spectrogram']),
                'iq': torch.from_numpy(sample['iq']),
                'freqs': torch.from_numpy(sample['freqs']),
                'times': torch.from_numpy(sample['times']),
                'metadata': sample
            }

# ---- BÖLÜM 10: VİZUALİZASYON ARAÇLARI ----

def plot_sample_analysis(sample_data: Dict, figsize=(15, 10)):
    """Bir sample'ın detaylı analizini çiz"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Spectrogram
        spec = sample_data['spectrogram']
        freqs = sample_data['freqs'] / 1e6  # MHz
        times = sample_data['times'] * 1e3   # ms
        
        im = axes[0, 0].imshow(spec, aspect='auto', origin='lower',
                              extent=[times[0], times[-1], freqs[0], freqs[-1]],
                              cmap='viridis')
        axes[0, 0].set_title('Spectrogram')
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Frequency (MHz)')
        plt.colorbar(im, ax=axes[0, 0], label='Power (dB)')
        
        # IQ constellation
        iq = sample_data['iq']
        axes[0, 1].scatter(iq.real[::10], iq.imag[::10], alpha=0.6, s=1)
        axes[0, 1].set_title('IQ Constellation')
        axes[0, 1].set_xlabel('I')
        axes[0, 1].set_ylabel('Q')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Power Spectral Density
        from scipy import signal as scipy_signal
        f_psd, psd = scipy_signal.welch(iq, fs=sample_data['fs'], nperseg=256)
        axes[0, 2].semilogy(f_psd/1e6, psd)
        axes[0, 2].set_title('Power Spectral Density')
        axes[0, 2].set_xlabel('Frequency (MHz)')
        axes[0, 2].set_ylabel('PSD')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Time domain
        t_ms = np.arange(len(iq)) / sample_data['fs'] * 1e3
        axes[1, 0].plot(t_ms[:500], np.abs(iq[:500]))
        axes[1, 0].set_title('Envelope (first 500 samples)')
        axes[1, 0].set_xlabel('Time (ms)')
        axes[1, 0].set_ylabel('Magnitude')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Instantaneous frequency
        phase_diff = np.diff(np.unwrap(np.angle(iq)))
        inst_freq = phase_diff * sample_data['fs'] / (2 * np.pi)
        axes[1, 1].plot(t_ms[1:1001], inst_freq[:1000] / 1e3)
        axes[1, 1].set_title('Instantaneous Frequency')
        axes[1, 1].set_xlabel('Time (ms)')
        axes[1, 1].set_ylabel('Frequency (kHz)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Signal info (text)
        info_text = f"Sample ID: {sample_data['sample_id']}\n"
        info_text += f"Type: {sample_data['sample_type']}\n"
        info_text += f"Signals: {sample_data['n_signals']}\n\n"
        
        for i, signal in enumerate(sample_data['signals']):
            info_text += f"Signal {i+1}:\n"
            info_text += f"  Mod: {signal.get('mod', 'N/A')}\n"
            info_text += f"  Power: {signal.get('power_db', 0):.1f} dB\n"
            info_text += f"  Scenario: {signal.get('tx_scenario', 'N/A')}\n"
            if 'f_off_hz' in signal:
                info_text += f"  Offset: {signal['f_off_hz']/1e3:.1f} kHz\n"
            info_text += "\n"
        
        axes[1, 2].text(0.05, 0.95, info_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=8)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
        
    except ImportError:
        print("matplotlib not available for plotting")
        return None

def analyze_dataset_statistics(base_dir: str):
    """Dataset istatistiklerini analiz et"""
    stats = get_dataset_info(base_dir)
    
    print("DATASET ANALYSIS")
    print("=" * 50)
    print(f"Total Samples: {stats['total_samples']:,}")
    print(f"Total Shards: {stats['total_shards']}")
    print(f"Samples per Shard: {stats['shard_size']}")
    print(f"Generation Time: {stats['generation_time']}")
    print()
    
    print("MODULATION DISTRIBUTION:")
    mod_counts = stats['modulation_counts']
    total_signals = sum(mod_counts.values())
    for mod, count in sorted(mod_counts.items()):
        percentage = (count / total_signals) * 100
        print(f"  {mod:8s}: {count:6,} ({percentage:5.1f}%)")
    print()
    
    print("SCENARIO DISTRIBUTION:")
    scenario_counts = stats['scenario_counts']
    total_scenarios = sum(scenario_counts.values())
    for scenario, count in sorted(scenario_counts.items()):
        percentage = (count / total_scenarios) * 100
        print(f"  {scenario:12s}: {count:6,} ({percentage:5.1f}%)")
    print()
    
    print("TECHNICAL PARAMETERS:")
    print(f"  Sampling Rate: {stats['fs']/1e6:.1f} MHz")
    print(f"  Duration: {stats['duration']*1e3:.1f} ms")    
    print(f"  FFT Size: {stats['n_fft']}")
    print(f"  FFT Overlap: {stats['n_overlap']}")
    print(f"  UHF Range: {stats['uhf_range'][0]/1e9:.1f} - {stats['uhf_range'][1]/1e9:.1f} GHz")

if __name__ == "__main__":  
    main()