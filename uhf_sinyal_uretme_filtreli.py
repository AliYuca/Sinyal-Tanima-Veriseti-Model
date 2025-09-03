
# Mods: FM, OFDM, GFSK, QPSK
# YENİ: Makaledeki Welch ve Logaritmik Dönüşüm yöntemleri entegre edildi.

import os, json, pickle, random, math, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

# =========================
# USER SETTINGS
# =========================
OUT_DIR = r"C:\Users\Osman\Desktop\Bites\sinyal_uhf\uhf_dataset_real_test_8" 
NUM_SAMPLES = 15_000
PROPORTIONS = {"noise":0.07, "single":0.15, "mixed_close":0.53, "mixed_far":0.25}
FS = 2_000_000         
DURATION = 1e-3        
N_FFT = 256            
N_OVERLAP = 128        
SHARD_SIZE = 2_000
SEED = 20250814

# UHF sınırlar
UHF_MIN = 300e6
UHF_MAX = 3e9

# ---- modlar ----
MODS = ['FM', 'OFDM', 'GFSK', 'QPSK']

# Mixed offsets
CLOSE_OFFSET_HZ = (5e3, 80e3)     # "yakın"
FAR_OFFSET_FRAC = (0.20, 0.45)    # "uzak" 

# =========================
# Helpers
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
    """Clock drift için doğrusal yeniden örnekleme; uzunluk korunur."""
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

# ---------- Spectral measurement helpers ----------
def _next_pow2(n: int):
    p = 1
    while p < n: p <<= 1
    return p

def measure_spectral_metrics(iq: torch.Tensor, fs: float, p_occ: float = 0.99):
    """
    Spektral ölçüm:
    - f_center_est_hz: spektral ağırlık merkezi
    - bw_occ99_hz    : %99 işgal bant genişliği
    - bw_rms_hz      : 2*std(f) (çift taraflı)
    - bw_3db_hz      : -3 dB genişlik (tepe gücün yarısı)
    """
    device = iq.device
    N = iq.numel()
    win = torch.hann_window(N, device=device, dtype=torch.float32)
    x = iq * win.to(iq.dtype)
    nfft = _next_pow2(int(2*N))  # 2x zero-pad
    X = torch.fft.fft(x, n=nfft)
    X = torch.fft.fftshift(X)
    P = (X.real**2 + X.imag**2).to(torch.float32) + 1e-30
    freqs = torch.linspace(-fs/2, fs/2, steps=nfft, device=device, dtype=torch.float32)

    Psum = torch.sum(P)
    f_center = torch.sum(freqs * P) / Psum

    Pcum = torch.cumsum(P, dim=0) / Psum
    lo_q = (1.0 - p_occ) / 2.0
    hi_q = 1.0 - lo_q
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
        f_lo = float(freqs[above[0]])
        f_hi = float(freqs[above[-1]])
        bw_3db = abs(f_hi - f_lo)
    else:
        bw_3db = 0.0

    return float(f_center.item()), float(bw_occ), float(bw_rms), float(bw_3db)

# =========================
# IQ Generator 
# =========================
class TorchIQGenRealistic:
    def __init__(self, fs=FS, duration=DURATION, device=None):
        self.fs = fs
        self.N = int(round(fs*duration))
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.t = torch.arange(self.N, device=self.device, dtype=torch.float32) / fs
        self.pi = math.pi

    # ---- bozulmalar ----
    def add_phase_noise(self, iq: torch.Tensor, linewidth_hz: float):
        sigma = math.sqrt(2*self.pi*max(linewidth_hz,1.0)/self.fs)
        dphi = sigma*torch.randn(self.N, device=self.device)
        phi = torch.cumsum(dphi, dim=0)
        return (iq * torch.exp(1j*phi)).to(torch.complex64)

    def apply_iq_imbalance_dc(self, iq: torch.Tensor):
        I = iq.real; Q = iq.imag
        gI = float(torch.normal(1.0, 0.07, size=(1,), device=self.device))
        gQ = float(torch.normal(1.0, 0.07, size=(1,), device=self.device))
        eps = float(torch.normal(0.0, math.radians(4.0), size=(1,), device=self.device))
        I2 = gI * I
        Q2 = gQ * (Q*math.cos(eps) + I*math.sin(eps))
        dc = complex(torch.normal(0.0, 2e-3, size=(1,), device=self.device).item(),
                     torch.normal(0.0, 2e-3, size=(1,), device=self.device).item())
        out = (I2 + 1j*(Q2 + dc)).to(torch.complex64)
        meta = dict(I_gain=gI, Q_gain=gQ, iq_phase_err_deg=math.degrees(eps), dc=(dc.real, dc.imag))
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

    def apply_rayleigh_with_path_doppler(self, iq: torch.Tensor, L=None, max_delay=12):
        if L is None: L = int(random.choice([2,3,4]))
        delays = torch.randint(0, max_delay+1, (L,), device=self.device)
        taps0 = (torch.randn(L, device=self.device) + 1j*torch.randn(L, device=self.device)) / math.sqrt(2*L)
        fds = torch.empty(L, device=self.device).uniform_(-100.0, 100.0)
        t = self.t
        y = torch.zeros_like(iq)
        meta_taps = []
        for d, h0, fd in zip(delays, taps0, fds):
            h_t = h0 * torch.exp(1j*2*self.pi*fd*t)
            y = y + torch.roll(iq, shifts=int(d)) * h_t
            meta_taps.append(dict(delay=int(d.item()) if torch.is_tensor(d) else int(d),
                                  tap=(complex(h0.real.item(), h0.imag.item())),
                                  fd_hz=float(fd.item())))
        return y.to(torch.complex64), meta_taps

    def add_awgn_total(self, iq_mix: torch.Tensor, snr_db: float):
        sp = torch.mean(torch.abs(iq_mix)**2) + 1e-20
        npow = sp / db2lin(snr_db)
        n = torch.sqrt(npow/2) * (torch.randn(self.N, device=self.device) + 1j*torch.randn(self.N, device=self.device))
        return (iq_mix + n.to(torch.complex64)).to(torch.complex64)

    # ---- YENİ FONKSİYON: Welch ve Logaritmik Geliştirme ----
    def compute_welch_log_spectrogram(self, iq: torch.Tensor, n_fft=N_FFT, noverlap=N_OVERLAP):
        """
        Bu fonksiyon, makaledeki yaklaşımları kullanarak 4 kanallı bir özellik çıkarır:
        1. Welch-benzeri Güç Spektrogramı (Logaritmik Geliştirme ile)
        2. Faz (Normalleştirilmiş)
        3. Anlık Frekans Spektrogramı (Normalleştirilmiş)
        4. Faz Türevi Spektrogramı (Normalleştirilmiş)
        """
        hop = n_fft - noverlap
        window = torch.hann_window(n_fft, device=self.device)

        # Adım 1: STFT'yi hesapla
        Z = torch.stft(iq, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=window, return_complex=True, center=False, onesided=False)

        # --- YENİ KANAL 1: Welch-benzeri Güç Spektrogramı ve Logaritmik Geliştirme ---
        # 1a: Güç Spektrumunu Hesapla (Welch yöntemindeki periyodogramların eşdeğeri)
        power_spec = torch.abs(Z)**2

        # 1b: Logaritmik Dönüşüm uygula (zayıf sinyalleri geliştirmek için)
        # Makaledeki formül: p_e(k) = log(c*p(k)) / log(c+1). c=1 için bu log(1+p(k))/log(2) olur.
        # torch.log1p(x) = log(1+x) olduğu için bu daha verimlidir.
        c = 1.0
        enhanced_spec = torch.log1p(c * power_spec) / math.log(c + 1)

        # 1c: Min-Max Normalizasyonu ile [0, 1] aralığına ölçekle
        min_val = torch.min(enhanced_spec)
        max_val = torch.max(enhanced_spec)
        power_channel = (enhanced_spec - min_val) / (max_val - min_val + 1e-8)
        power_channel = power_channel.to(torch.float32)

        # --- MEVCUT DİĞER KANALLAR ---
        # Kanal 2: Faz (Normalleştirilmiş)
        phase = torch.angle(Z)
        phase = (phase + math.pi)/(2*math.pi)
        phase = phase.to(torch.float32)

        # Kanal 3: Anlık Frekans Spektrogramı
        prev = torch.roll(iq, shifts=1)
        dphi = torch.angle(iq*torch.conj(prev))
        inst_f = (self.fs/(2*math.pi))*dphi.to(torch.float32)
        Z_if = torch.stft(inst_f, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                          window=window, return_complex=True, center=False, onesided=False)
        if_spec = torch.abs(Z_if)
        q1 = torch.quantile(if_spec, 0.01); q9 = torch.quantile(if_spec, 0.99)
        if_spec = torch.clamp(if_spec, q1, q9)
        if_spec = ((if_spec - if_spec.min())/(if_spec.max()-if_spec.min()+1e-8)).to(torch.float32)

        # Kanal 4: Faz Türevi Spektrogramı
        phase_der = torch.diff(phase, dim=0, prepend=phase[0:1])
        q1 = torch.quantile(phase_der, 0.01); q9 = torch.quantile(phase_der, 0.99)
        phase_der = torch.clamp(phase_der, q1, q9)
        phase_der = ((phase_der - phase_der.min())/(phase_der.max()-phase_der.min()+1e-8)).to(torch.float32)

        F0, T0 = power_channel.shape
        assert phase.shape == (F0, T0) and if_spec.shape == (F0, T0) and phase_der.shape == (F0, T0), \
            f"Shape mismatch: power{power_channel.shape}, ph{phase.shape}, ifs{if_spec.shape}, phg{phase_der.shape}"
        
        return power_channel, phase, if_spec, phase_der

    # ---- upsample + RC ----
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

    # ---- modülasyon  ----
    def gen_qpsk(self, power_db=-10, rsym_range=(100e3, 400e3), rolloff_range=(0.2,0.5)):
        Rs = float(torch.empty(1, device=self.device).uniform_(*rsym_range))
        sps = max(2, int(self.fs/Rs))
        n_sym = int(math.ceil(self.N/sps))+16
        bits = torch.randint(0, 2, (2*n_sym,), device=self.device)
        I = (2*bits[0::2]-1).to(torch.float32)
        Q = (2*bits[1::2]-1).to(torch.float32)
        syms = (I + 1j*Q)/math.sqrt(2)
        beta = float(torch.empty(1, device=self.device).uniform_(*rolloff_range))
        taps = rc_impulse_response(beta, sps, span=10, device=self.device)
        iq = self._filter_up(syms, taps, sps)
        return self._finalize_linear(iq, Rs, power_db, beta, pulse='RC')

    def _finalize_linear(self, iq: torch.Tensor, Rs: float, power_db: float, beta: float, pulse='RC', extra=None):
        sps = max(2, int(self.fs/Rs))
        toff0 = int(torch.randint(0, sps, (1,), device=self.device))
        iq = torch.roll(iq, shifts=toff0)

        iq, iq_meta  = self.apply_iq_imbalance_dc(iq)
        iq, cfo_meta = self.apply_cfo_and_slow_doppler(iq)
        if random.random() < 0.8:
            iq, clk_meta = self.apply_clock_drift(iq)
        else:
            clk_meta = dict(clock_drift_ppm=0.0)
        iq, path_meta = self.apply_rayleigh_with_path_doppler(iq, L=random.choice([2,3,4]), max_delay=12)
        iq = self.add_phase_noise(iq, linewidth_hz=float(torch.empty(1, device=self.device).uniform_(20.0, 2e4)))

        p_cur = torch.mean(torch.abs(iq)**2) + 1e-20
        p_tgt = db2lin(power_db)
        iq = iq * torch.sqrt(torch.tensor(p_tgt, device=self.device, dtype=torch.float32)/p_cur)

        D = self.N / self.fs
        f_center_theory = cfo_meta["cfo0_hz"] + 0.5 * cfo_meta["cfo_slope_hz_s"] * D

        meta = dict(Rs=Rs, sps=sps, rolloff=beta, pulse=pulse, time_ofs=int(toff0),
                    f_center_theory_hz_approx=float(f_center_theory),
                    **iq_meta, **cfo_meta, **clk_meta, rayleigh_paths=path_meta)
        if extra: meta.update(extra)
        return iq.to(torch.complex64), meta

    def gen_fm(self, power_db=-10, fdev_range=(3e3, 30e3), fm_range=(1e3, 10e3)):
        fm = float(torch.empty(1, device=self.device).uniform_(*fm_range))
        msg = torch.sin(2*math.pi*fm*self.t)
        fdev = float(torch.empty(1, device=self.device).uniform_(*fdev_range))
        phase = 2*math.pi*fdev*torch.cumsum(msg, dim=0)/self.fs
        iq = torch.exp(1j*phase).to(torch.complex64)
        iq, cfo_meta = self.apply_cfo_and_slow_doppler(iq)
        iq, clk_meta = self.apply_clock_drift(iq)
        iq, path_meta = self.apply_rayleigh_with_path_doppler(iq)
        iq = self.add_phase_noise(iq, linewidth_hz=float(torch.empty(1, device=self.device).uniform_(20.0, 2e4)))
        p_cur = torch.mean(torch.abs(iq)**2) + 1e-20
        iq = iq * torch.sqrt(torch.tensor(db2lin(power_db), device=self.device)/p_cur)
        D = self.N / self.fs
        f_center_theory = cfo_meta["cfo0_hz"] + 0.5 * cfo_meta["cfo_slope_hz_s"] * D
        meta = dict(fm_hz=fm, fdev_hz=fdev, f_center_theory_hz_approx=float(f_center_theory),
                    **cfo_meta, **clk_meta, rayleigh_paths=path_meta)
        return iq, meta

    def gen_ofdm(self, power_db=-10):
        Nfft = int(random.choice([64,128,256]))
        cp = int(random.choice([Nfft//16, Nfft//8]))
        n_sym = max(10, int(self.N // (Nfft+cp)) + 2)
        active = torch.zeros(Nfft, dtype=torch.bool, device=self.device)
        lo = Nfft//8; hi = Nfft - lo
        active[lo:hi] = True

        use_qpsk = (random.random() < 0.5)
        if use_qpsk:
            mI = torch.randint(0,2,(n_sym, Nfft), device=self.device)*2 - 1
            mQ = torch.randint(0,2,(n_sym, Nfft), device=self.device)*2 - 1
            QAM = (mI + 1j*mQ)/math.sqrt(2)
        else:
            lv = torch.tensor([-3,-1,1,3], device=self.device, dtype=torch.float32)
            I = lv[torch.randint(0,4,(n_sym,Nfft), device=self.device)]
            Q = lv[torch.randint(0,4,(n_sym,Nfft), device=self.device)]
            Es = torch.mean(I**2+Q**2); QAM = (I + 1j*Q)/torch.sqrt(Es+1e-12)
        QAM *= active
        x = torch.fft.ifft(QAM, dim=1)
        x = torch.cat([x[:, -cp:], x], dim=1).reshape(-1)
        if x.numel() < self.N: x = F.pad(x, (0, self.N-x.numel()))
        x = x[:self.N].to(torch.complex64)

        x, cfo_meta = self.apply_cfo_and_slow_doppler(x)
        x, clk_meta = self.apply_clock_drift(x)
        x, path_meta = self.apply_rayleigh_with_path_doppler(x, L=random.choice([2,3,4]), max_delay=12)
        x = self.add_phase_noise(x, linewidth_hz=float(torch.empty(1, device=self.device).uniform_(20.0, 2e4)))
        p_cur = torch.mean(torch.abs(x)**2) + 1e-20
        x = x * torch.sqrt(torch.tensor(db2lin(power_db), device=self.device)/p_cur)
        D = self.N / self.fs
        f_center_theory = cfo_meta["cfo0_hz"] + 0.5 * cfo_meta["cfo_slope_hz_s"] * D
        meta = dict(Nfft=Nfft, cp=cp, n_sym=n_sym, qpsk=use_qpsk,
                    f_center_theory_hz_approx=float(f_center_theory),
                    **cfo_meta, **clk_meta, rayleigh_paths=path_meta)
        return x, meta

    def gen_gfsk(self, power_db=-10, rsym_range=(50e3, 200e3), BT=0.5):
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
        iq = torch.exp(1j*phase).to(torch.complex64)
        iq, cfo_meta = self.apply_cfo_and_slow_doppler(iq)
        iq, clk_meta = self.apply_clock_drift(iq)
        iq, path_meta = self.apply_rayleigh_with_path_doppler(iq)
        iq = self.add_phase_noise(iq, linewidth_hz=float(torch.empty(1, device=self.device).uniform_(20.0, 2e4)))
        p_cur = torch.mean(torch.abs(iq)**2) + 1e-20
        iq = iq * torch.sqrt(torch.tensor(db2lin(power_db), device=self.device)/p_cur)
        D = self.N / self.fs
        f_center_theory = cfo_meta["cfo0_hz"] + 0.5 * cfo_meta["cfo_slope_hz_s"] * D
        meta = dict(Rs=Rs, sps=sps, BT=BT, fdev=fdev,
                    f_center_theory_hz_approx=float(f_center_theory),
                    **cfo_meta, **clk_meta, rayleigh_paths=path_meta)
        return iq, meta

# ---------- Senaryolar ----------
def gen_single(gen: TorchIQGenRealistic):
    mod = random.choice(MODS)
    p_db = random.uniform(-20, 0)
    if   mod=='FM':   iq, meta = gen.gen_fm(p_db)
    elif mod=='OFDM': iq, meta = gen.gen_ofdm(p_db)
    elif mod=='GFSK': iq, meta = gen.gen_gfsk(p_db)
    else:             iq, meta = gen.gen_qpsk(p_db)

    f_center_est, bw_occ, bw_rms, bw_3db = measure_spectral_metrics(iq, gen.fs, p_occ=0.99)

    info = [{
        **meta,
        "mod": mod,
        "f_off_hz": 0.0,
        "rel_power_db": 0.0,
        "f_center_est_hz": f_center_est,
        "bw_occ99_hz": bw_occ,
        "bw_rms_hz": bw_rms,
        "bw_3db_hz": bw_3db,
    }]
    return iq, info

def gen_mixed(gen: TorchIQGenRealistic, close=True):
    k = random.randint(2, 4)
    rel_powers_db = sorted([random.uniform(-18, 0) for _ in range(k)], reverse=True)
    sigs, info = [], []
    for idx in range(k):
        mod = random.choice(MODS); p_db = rel_powers_db[idx]
        if   mod=='FM':   s, meta = gen.gen_fm(p_db)
        elif mod=='OFDM': s, meta = gen.gen_ofdm(p_db)
        elif mod=='GFSK': s, meta = gen.gen_gfsk(p_db)
        else:             s, meta = gen.gen_qpsk(p_db)

        if close:
            foff = random.uniform(*CLOSE_OFFSET_HZ) * random.choice([-1, 1])
        else:
            frac = random.uniform(*FAR_OFFSET_FRAC) * random.choice([-1, 1])
            foff = frac*gen.fs

        s = (s * torch.exp(1j*2*math.pi*foff*gen.t)).to(torch.complex64)

        f_center_est, bw_occ, bw_rms, bw_3db = measure_spectral_metrics(s, gen.fs, p_occ=0.99)

        row = {
            **meta, 
            "mod": mod,
            "f_off_hz": float(foff),
            "rel_power_db": float(p_db),
            "f_center_theory_hz_approx": meta.get("f_center_theory_hz_approx", 0.0) + float(foff),
            "f_center_est_hz": f_center_est,
            "bw_occ99_hz": bw_occ,
            "bw_rms_hz": bw_rms,
            "bw_3db_hz": bw_3db,
        }
        sigs.append(s)
        info.append(row)

    mix = torch.zeros(gen.N, device=gen.device, dtype=torch.complex64)
    for s in sigs: mix = mix + s
    return mix, info

def gen_noise(gen: TorchIQGenRealistic):
    noise_db = random.uniform(-50, -20)
    p_lin = db2lin(noise_db)
    n = torch.sqrt(torch.tensor(p_lin/2, device=gen.device)) * \
        (torch.randn(gen.N, device=gen.device) + 1j*torch.randn(gen.N, device=gen.device))
    return n.to(torch.complex64), []

# =========================
# Veriseti
# =========================
def generate_realistic_uhf_dataset(
    out_dir=OUT_DIR, num_samples=NUM_SAMPLES, proportions=PROPORTIONS,
    fs=FS, duration=DURATION, n_fft=N_FFT, n_overlap=N_OVERLAP, shard_size=SHARD_SIZE, seed=SEED
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = TorchIQGenRealistic(fs=fs, duration=duration, device=device)

    print(f"🧠 Device: {device} (CUDA {'ON' if device.type=='cuda' else 'OFF'})")

    counts = {k:int(round(v*num_samples)) for k,v in proportions.items()}
    diff = num_samples - sum(counts.values())
    if diff != 0: counts['mixed_close'] = counts.get('mixed_close',0) + diff

    plan = (['noise']*counts['noise'] + ['single']*counts['single'] +
            ['mixed_close']*counts['mixed_close'] + ['mixed_far']*counts['mixed_far'])
    random.shuffle(plan)

    manifest = {
        "created": datetime.now().isoformat(),
        "seed": seed,
        "baseband_fs": fs,
        "duration_s": duration,
        "band_limits_hz": [UHF_MIN, UHF_MAX],
        "snr_range_db_total": [0, 30],
        "proportions": proportions,
        "stft": {"n_fft": n_fft, "noverlap": n_overlap, "hop": n_fft-n_overlap, "window": "hann", "onesided": False},
        "shard_size": shard_size,
        "num_samples": num_samples,
        "mods": MODS,
        # YENİ ÖZELLİK KANALLARI AÇIKLAMASI
        "feature_channels": ["log_enhanced_power_norm", "phase_norm", "instfreq_spec_norm", "phase_deriv_norm"],
        "impairments_added": ["IQ imbalance", "DC offset", "Phase noise",
                              "Rayleigh(2-4 paths) + per-path Doppler",
                              "CFO + slow drift", "Clock drift (resample)"],
        "version": "R1.3_torch_gpu_rc_welch_log_measured", 
    }
    manifest.update({
        "label_fields_per_signal": [
            "mod", "rel_power_db",
            "f_off_hz",
            "f_center_est_hz", "f_center_theory_hz_approx",
            "bw_occ99_hz", "bw_rms_hz", "bw_3db_hz"
        ],
        "bandwidth_measurement": {
            "occupied_power": 0.99,
            "rms_bandwidth": "2*sqrt(sum(P*(f-fc)^2)/sum(P))",
            "minus3dB": "width where power >= peak/2"
        }
    })
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    shard_idx = 0
    buf_features, buf_iq_raw, buf_labels = [], [], []
    stats = {"noise":0, "single":0, "mixed_close":0, "mixed_far":0}

    pbar = tqdm(plan, desc="Generating WELCH+LOG UHF dataset", unit="sample")
    for cat in pbar:
        if cat == 'noise':
            abs_fc_list = []
        elif cat == 'single':
            abs_fc_list = [float(np.random.uniform(UHF_MIN, UHF_MAX))]
        else:
            k = random.randint(2,4); base_fc = float(np.random.uniform(UHF_MIN, UHF_MAX))
            abs_fc_list = [float(np.clip(base_fc + np.random.uniform(-5e6, 5e6), UHF_MIN, UHF_MAX)) for _ in range(k)]

        if   cat=='single':      iq, info = gen_single(gen)
        elif cat=='mixed_close': iq, info = gen_mixed(gen, close=True)
        elif cat=='mixed_far':   iq, info = gen_mixed(gen, close=False)
        else:                    iq, info = gen_noise(gen)

        if cat != 'noise':
            snr_db = float(np.random.uniform(0, 30))
            iq = gen.add_awgn_total(iq, snr_db)
        else:
            snr_db = -np.inf

        # DEĞİŞİKLİK: Yeni özellik çıkarma fonksiyonu çağrılıyor
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

        if len(buf_features) >= shard_size:
            shard_path = os.path.join(out_dir, f"shard_{shard_idx:05d}")
            os.makedirs(shard_path, exist_ok=True)
            np.save(os.path.join(shard_path, "features.npy"), np.stack(buf_features, axis=0))
            np.save(os.path.join(shard_path, "iq_raw.npy"),    np.stack(buf_iq_raw, axis=0))
            with open(os.path.join(shard_path, "labels.pkl"), "wb") as f:
                pickle.dump(buf_labels, f)
            buf_features, buf_iq_raw, buf_labels = [], [], []
            shard_idx += 1

        if len(buf_features) % 500 == 0:
            pbar.set_postfix({"Shard": shard_idx, "Type": cat})

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
    print("✅ WELCH+LOGARITHMIC UHF DATASET GENERATION COMPLETE")
    print("="*64)
    print(f"📁 Output dir: {out_dir}")
    print(f"📊 Total samples: {sum(stats.values())}")
    print(f"🧠 Device: {device}")
    print(f"🔧 Feature channels: 4 = [log_power, phase, instfreq_spec, phase_grad]")
    print(f"📈 Sample counts: {stats}")

if __name__ == "__main__":
    generate_realistic_uhf_dataset()