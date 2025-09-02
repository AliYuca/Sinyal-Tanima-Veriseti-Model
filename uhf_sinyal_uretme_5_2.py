# uhf_sinyal_uretme_basitlestirilmis_yorumlu_8.py
import os, json, pickle, random, math, warnings
warnings.filterwarnings('ignore') 

import numpy as np
import torch
import torch.nn.functional as F

# İlerleme çubuğu gösterimi için
from tqdm import tqdm
# Zaman damgası işlemleri için
from datetime import datetime

# KULLANICI AYARLARI
# =========================
OUT_DIR = r"C:\Users\Osman\Desktop\BİTES\sinyal_uhf\uhf_dataset_basitlestirilmis_yorumlu" 
NUM_SAMPLES = 15_000  # sinyal sayısı
PROPORTIONS = {"noise":0.07, "single":0.15, "mixed_close":0.53, "mixed_far":0.25} # Örnek tiplerinin yüzdesel dağılımı
FS = 2_000_000           
DURATION = 1e-3          # Her bir sinyal örneğinin süresi
N_FFT = 256              #  FFT pencere boyutu
N_OVERLAP = 128          
SHARD_SIZE = 2_000       
SEED = 20250814      

# UHF frekans sınırları
UHF_MIN = 300e6 # 300 MHz
UHF_MAX = 3e9   # 3 GHz

# modülasyon tipleri
MODS = ['FM', 'OFDM', 'GFSK', 'QPSK'] 

# Karışık (mixed) sinyal senaryolarında, sinyaller arasındaki frekans farkı aralıkları
CLOSE_OFFSET_HZ = (5e3, 80e3)     # Yakın frekanslı
FAR_OFFSET_FRAC = (0.20, 0.45)    # Uzak frekanslı


# Yardımcı Fonksiyonlar
def set_seed(seed=SEED):
    """Rastgele sayı üreteçlerini sabit bir başlangıç değerine ayarlar."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_cpu_np(x: torch.Tensor):
    """Bir PyTorch tensor'ünü CPU'ya taşıyıp NumPy dizisine çevirir."""
    return x.detach().to('cpu').numpy()

def db2lin(db): 
    """Desibel (dB) cinsinden bir değeri lineer bir orana çevirir."""
    return 10.0 ** (db/10.0)

def rc_impulse_response(beta: float, sps: int, span: int, device):
    """Raised Cosine (RC) filtresinin dürtü yanıtını hesaplar. Genellikle dijital modülasyonlarda darbe şekillendirme için kullanılır."""
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
    """Gauss filtresinin dürtü yanıtını hesaplar. GFSK modülasyonunda kullanılır."""
    L = span*sps + 1
    t = torch.linspace(-span/2, span/2, L, device=device, dtype=torch.float32)
    sigma_sym = 0.32/ max(BT, 1e-3)
    g = torch.exp(-0.5*(t/sigma_sym)**2)
 
    g = g / (g.sum() + 1e-12)
    return g

# ---------- Spektral Ölçüm Yardımcıları ----------
def _next_pow2(n: int):
    """Verilen bir sayıdan büyük veya eşit olan en küçük 2'nin kuvvetini bulur. FFT hesaplamalarında verimlilik için kullanılır."""
    p = 1
    while p < n: p <<= 1
    return p

def measure_spectral_metrics(iq: torch.Tensor, fs: float, p_occ: float = 0.99):
    """
    Bir IQ sinyalinin spektral özelliklerini (merkez frekans, bant genişliği vb.) ölçer.
    Bu, veri setindeki etiketlerin daha doğru olmasını sağlar.
    """
    device = iq.device
    N = iq.numel()
    win = torch.hann_window(N, device=device, dtype=torch.float32)
    x = iq * win.to(iq.dtype)
    nfft = _next_pow2(int(2*N))  # Çözünürlüğü artırmak için zero-padding
    X = torch.fft.fft(x, n=nfft)
    X = torch.fft.fftshift(X)
    P = (X.real**2 + X.imag**2).to(torch.float32) + 1e-30 # Güç Spektrumu
    freqs = torch.linspace(-fs/2, fs/2, steps=nfft, device=device, dtype=torch.float32)

    # Spektral merkez (ağırlıklı ortalama frekans)
    Psum = torch.sum(P)
    f_center = torch.sum(freqs * P) / Psum

    # bant denişliği aralığı belirleme
    Pcum = torch.cumsum(P, dim=0) / Psum
    lo_q = (1.0 - p_occ) / 2.0
    hi_q = 1.0 - lo_q
    il = torch.searchsorted(Pcum, torch.tensor(lo_q, device=device))
    ih = torch.searchsorted(Pcum, torch.tensor(hi_q, device=device))
    il = int(torch.clamp(il, 0, nfft-1)); ih = int(torch.clamp(ih, 0, nfft-1))
    bw_occ = float((freqs[ih] - freqs[il]).abs())

    # RMS bant genişliği
    var = torch.sum(P * (freqs - f_center)**2) / Psum
    bw_rms = float(2.0 * torch.sqrt(torch.clamp(var, min=0.0)))

    # -3dB bant genişliği (tepe gücün yarısı)
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

# IQ Sinyal Üreteci
class TorchIQGenRealistic:
    def __init__(self, fs=FS, duration=DURATION, device=None):
        """Sınıfın başlangıç ayarlarını yapar."""
        self.fs = fs
        self.N = int(round(fs*duration))
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        # Zaman vektörünü oluştur
        self.t = torch.arange(self.N, device=self.device, dtype=torch.float32) / fs
        self.pi = math.pi

    #  BOZULMALAR
    def add_phase_noise(self, iq: torch.Tensor, linewidth_hz: float):
        """Faz Gürültüsü: Osilatörlerdeki kararsızlıklardan kaynaklanan ve sinyalin fazında zamanla meydana gelen küçük, rastgele dalgalanmaları simüle eder."""
        sigma = math.sqrt(2*self.pi*max(linewidth_hz,1.0)/self.fs)
        dphi = sigma*torch.randn(self.N, device=self.device)
        phi = torch.cumsum(dphi, dim=0)
        return (iq * torch.exp(1j*phi)).to(torch.complex64)

    def apply_iq_imbalance(self, iq: torch.Tensor):
        """IQ Dengesizliği: Sinyalin I (gerçek) ve Q (sanal) bileşenleri arasındaki donanımsal kusurları taklit eder. Genlikler ve fazlar tam olarak dengeli olmaz."""
        I = iq.real; Q = iq.imag
        gI = float(torch.normal(1.0, 0.07, size=(1,), device=self.device)) # I için genlik hatası
        gQ = float(torch.normal(1.0, 0.07, size=(1,), device=self.device)) # Q için genlik hatası
        eps = float(torch.normal(0.0, math.radians(4.0), size=(1,), device=self.device)) # Faz hatası (90 dereceden sapma)
        I2 = gI * I
        Q2 = gQ * (Q*math.cos(eps) + I*math.sin(eps))
        out = (I2 + 1j*Q2).to(torch.complex64) # DC bileşeni eklenmeden birleştirilir
        meta = dict(I_gain=gI, Q_gain=gQ, iq_phase_err_deg=math.degrees(eps))
        return out, meta
    
    def apply_rician_fading_with_path_doppler(self, iq: torch.Tensor, L=None, max_delay=12, K_factor=4.0):
        """
        Rician Sönümlemesi: Sönümleme etkisi azaltılmış bir çok-yollu (multipath) yayılım modelidir.
        Bir adet güçlü, doğrudan gelen sinyal (Line-of-Sight) ve birkaç zayıf, yansıyan sinyalden oluşur.
        K_factor: Doğrudan gelen sinyal gücünün yansıyan sinyallerin toplam gücüne oranıdır.
                   Yüksek K değeri, daha az sönümleme (daha kararlı sinyal) anlamına gelir.
        """
        if L is None: L = int(random.choice([2,3])) # Yansıyan (dolaylı) yol sayısı
        delays = torch.randint(1, max_delay+1, (L,), device=self.device) # Yansıyanlar için gecikme (örnek cinsinden)
        
        # Toplam gücü, LoS (doğrudan) ve NLoS (yansıyan) yollar arasında K faktörüne göre paylaştır
        power_los = K_factor / (K_factor + 1.0)
        power_nlos = 1.0 / (K_factor + 1.0)

        y = torch.zeros_like(iq) # Sonuç sinyalini tutmak için boş bir tensor
        meta_taps = [] # Meta verileri kaydetmek için

        # 1. Doğrudan Yolu (Line-of-Sight, LoS) Ekle
        los_tap = torch.sqrt(torch.tensor(power_los, device=self.device))
        y += iq * los_tap # Gecikmesiz ve sönümlemesiz ana yol
        meta_taps.append(dict(delay=0, tap=complex(los_tap.item(), 0), fd_hz=0.0, type='LoS'))

        # 2. Yansıyan Yolları (Non-Line-of-Sight, NLoS - Rayleigh bileşenleri) Ekle
        nlos_taps = (torch.randn(L, device=self.device) + 1j*torch.randn(L, device=self.device)) / math.sqrt(2*L)
        nlos_taps *= torch.sqrt(torch.tensor(power_nlos, device=self.device)) # Güçlerini ayarla
        
        # Her yansıyan yola, hareketlilikten kaynaklanan küçük bir frekans kayması (Doppler) ekle
        fds = torch.empty(L, device=self.device).uniform_(-100.0, 100.0) 
        
        for d, h0, fd in zip(delays, nlos_taps, fds):
            h_t = h0 * torch.exp(1j*2*self.pi*fd*self.t)
            # Geciktirilmiş, sönümlenmiş ve frekansı kaydırılmış sinyali toplama ekle
            y = y + torch.roll(iq, shifts=int(d)) * h_t
            meta_taps.append(dict(delay=int(d.item()), tap=(complex(h0.real.item(), h0.imag.item())), fd_hz=float(fd.item()), type='NLoS'))
            
        return y.to(torch.complex64), meta_taps

    def add_awgn_total(self, iq_mix: torch.Tensor, snr_db: float):
        """AWGN (Eklenir Beyaz Gauss Gürültüsü): Sinyale genel arkaplan gürültüsü ekleyerek belirli bir Sinyal-Gürültü Oranı (SNR) ayarlar."""
        sp = torch.mean(torch.abs(iq_mix)**2) + 1e-20 # Sinyal gücünü hesapla
        npow = sp / db2lin(snr_db) # Gerekli gürültü gücünü hesapla
        # Karmaşık (complex) Gauss gürültüsü oluştur ve sinyale ekle
        n = torch.sqrt(npow/2) * (torch.randn(self.N, device=self.device) + 1j*torch.randn(self.N, device=self.device))
        return (iq_mix + n.to(torch.complex64)).to(torch.complex64)

    # ---- ÖZELLİK ÇIKARIMI  ----
    def compute_spectrograms(self, iq: torch.Tensor, n_fft=N_FFT, noverlap=N_OVERLAP):
        """
        Verilen bir IQ sinyalinden 4 kanallı spektrogram özellikleri çıkarır.
        Bu özellikler, derin öğrenme modelinin girdisi olarak kullanılır.
        Kanallar: Genlik (dB), Faz, Anlık Frekans Spektrumu, Faz Türevi.
        """
        hop = n_fft - noverlap
        window = torch.hann_window(n_fft, device=self.device)

        # STFT hesapla
        Z = torch.stft(iq, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=window, return_complex=True, center=False, onesided=False)
        
        # Kanal 1: Genlik dB cinsinden
        mag = 20*torch.log10(torch.abs(Z)+1e-9)
        mag = torch.clamp(mag, -80.0, 0.0).to(torch.float32)

        # Kanal 2: Normalize faz 
        phase = torch.angle(Z)
        phase = (phase + math.pi)/(2*math.pi)
        phase = phase.to(torch.float32)

        # Kanal 3: Anlık Frekans Spektrumu
        prev = torch.roll(iq, shifts=1)
        dphi = torch.angle(iq*torch.conj(prev))
        inst_f = (self.fs/(2*math.pi))*dphi.to(torch.float32)
        Z_if = torch.stft(inst_f, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                          window=window, return_complex=True, center=False, onesided=False)
        if_spec = torch.abs(Z_if)
        q1 = torch.quantile(if_spec, 0.01); q9 = torch.quantile(if_spec, 0.99)
        if_spec = torch.clamp(if_spec, q1, q9)
        if_spec = ((if_spec - if_spec.min())/(if_spec.max()-if_spec.min()+1e-8)).to(torch.float32)

        # Kanal 4: Fazın Türevi 
        phase_der = torch.diff(phase, dim=0, prepend=phase[0:1])
        q1 = torch.quantile(phase_der, 0.01); q9 = torch.quantile(phase_der, 0.99)
        phase_der = torch.clamp(phase_der, q1, q9)
        phase_der = ((phase_der - phase_der.min())/(phase_der.max()-phase_der.min()+1e-8)).to(torch.float32)
        
        return mag, phase, if_spec, phase_der

    def _filter_up(self, symbols: torch.Tensor, taps: torch.Tensor, sps: int):
        """Sembolleri 'sps' kadar yukarı örnekler (aralara sıfır ekler) ve darbe şekillendirici filtreden geçirir."""
        L = symbols.numel()*sps
        up = torch.zeros(L, device=self.device, dtype=torch.complex64)
        up[::sps] = symbols.to(torch.complex64)
        pad = taps.numel()//2
        I = F.conv1d(up.real.view(1,1,-1), taps.view(1,1,-1), padding=pad).view(-1)
        Q = F.conv1d(up.imag.view(1,1,-1), taps.view(1,1,-1), padding=pad).view(-1)
        y = (I + 1j*Q).to(torch.complex64)
        if y.numel() < self.N: y = F.pad(y, (0, self.N-y.numel()))
        return y[:self.N]

    # ---- MODÜLATÖRLER ----
    
    def gen_qpsk(self, power_db=-10, rsym_range=(100e3, 400e3), rolloff_range=(0.2,0.5)):
        """QPSK modülasyonlu sinyal üretir."""
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
        return self._finalize_signal(iq, Rs, power_db, beta, pulse='RC')

    def _finalize_signal(self, iq: torch.Tensor, Rs: float, power_db: float, beta: float, pulse='RC', extra=None):
        """Üretilen bir sinyale son bozulmaları ekler ve gücünü ayarlar."""
        sps = max(2, int(self.fs/Rs))
        toff0 = int(torch.randint(0, sps, (1,), device=self.device))
        iq = torch.roll(iq, shifts=toff0)

        # BOZULMA ZİNCİRİ
        iq, iq_meta = self.apply_iq_imbalance(iq)
        
        iq, path_meta = self.apply_rician_fading_with_path_doppler(iq) # Azaltılmış sönümleme modeli
        iq = self.add_phase_noise(iq, linewidth_hz=float(torch.empty(1, device=self.device).uniform_(20.0, 2e4)))

        # Sinyal gücünü istenen seviyeye normalize et
        p_cur = torch.mean(torch.abs(iq)**2) + 1e-20
        p_tgt = db2lin(power_db)
        iq = iq * torch.sqrt(torch.tensor(p_tgt, device=self.device, dtype=torch.float32)/p_cur)

        meta = dict(Rs=Rs, sps=sps, rolloff=beta, pulse=pulse, time_ofs=int(toff0),
                    **iq_meta, rician_paths=path_meta)
        if extra: meta.update(extra)
        return iq.to(torch.complex64), meta

    def gen_fm(self, power_db=-10, fdev_range=(3e3, 30e3), fm_range=(1e3, 10e3)):
        """FM modülasyonlu sinyal üretir."""
        fm = float(torch.empty(1, device=self.device).uniform_(*fm_range))
        msg = torch.sin(2*math.pi*fm*self.t)
        fdev = float(torch.empty(1, device=self.device).uniform_(*fdev_range))
        phase = 2*math.pi*fdev*torch.cumsum(msg, dim=0)/self.fs
        iq = torch.exp(1j*phase).to(torch.complex64)
        
        iq, path_meta = self.apply_rician_fading_with_path_doppler(iq)
        iq = self.add_phase_noise(iq, linewidth_hz=float(torch.empty(1, device=self.device).uniform_(20.0, 2e4)))
        
        p_cur = torch.mean(torch.abs(iq)**2) + 1e-20
        iq = iq * torch.sqrt(torch.tensor(db2lin(power_db), device=self.device)/p_cur)
        meta = dict(fm_hz=fm, fdev_hz=fdev, rician_paths=path_meta)
        return iq, meta

# Bu fonksiyonlar tekli, karışık veya gürültü senaryolarına göre sinyalleri üretir.

def gen_single(gen: TorchIQGenRealistic):
    """Tek bir modülasyona sahip sinyal üretir."""
    mod = random.choice(MODS)
    p_db = random.uniform(-20, 0)
    
    # Şimdilik sadece QPSK ve FM için basitleştirilmiş zincir kullanılıyor
    if mod=='FM':   iq, meta = gen.gen_fm(p_db)
    elif mod=='QPSK': iq, meta = gen.gen_qpsk(p_db)
    else: # Diğer modülasyonlar için de bir jeneratör seçilir 
          iq, meta = gen.gen_qpsk(p_db) 

    # Üretilen sinyalin spektral özelliklerini ölç
    f_center_est, bw_occ, bw_rms, bw_3db = measure_spectral_metrics(iq, gen.fs, p_occ=0.99)

    # Etiket bilgilerini oluştur
    info = [{
        **meta,
        "mod": mod, "f_off_hz": 0.0, "rel_power_db": 0.0,
        "f_center_est_hz": f_center_est, "bw_occ99_hz": bw_occ,
        "bw_rms_hz": bw_rms, "bw_3db_hz": bw_3db,
    }]
    return iq, info

def gen_mixed(gen: TorchIQGenRealistic, close=True):
    """Aynı anda birden fazla (2-4) sinyal içeren karışık bir sinyal üretir."""
    k = random.randint(2, 4)
    rel_powers_db = sorted([random.uniform(-18, 0) for _ in range(k)], reverse=True)
    sigs, info = [], []
    for idx in range(k):
        mod = random.choice(MODS); p_db = rel_powers_db[idx]
        if   mod=='FM':   s, meta = gen.gen_fm(p_db)
        else:             s, meta = gen.gen_qpsk(p_db) # Basitleştirme için sadece FM/QPSK

        # Sinyalleri farklı frekanslara kaydır
        if close: foff = random.uniform(*CLOSE_OFFSET_HZ) * random.choice([-1, 1])
        else: foff = random.uniform(*FAR_OFFSET_FRAC) * random.choice([-1, 1]) * gen.fs
        
        s = (s * torch.exp(1j*2*math.pi*foff*gen.t)).to(torch.complex64)

        # Her bir bileşenin özelliklerini ölç
        f_center_est, bw_occ, bw_rms, bw_3db = measure_spectral_metrics(s, gen.fs, p_occ=0.99)
        row = { **meta, "mod": mod, "f_off_hz": float(foff), "rel_power_db": float(p_db),
            "f_center_est_hz": f_center_est, "bw_occ99_hz": bw_occ,
            "bw_rms_hz": bw_rms, "bw_3db_hz": bw_3db, }
        sigs.append(s); info.append(row)

    # Tüm sinyalleri topla
    mix = torch.zeros(gen.N, device=gen.device, dtype=torch.complex64)
    for s in sigs: mix = mix + s
    return mix, info

def gen_noise(gen: TorchIQGenRealistic):
    """Sadece gürültü içeren bir sinyal üretir."""
    noise_db = random.uniform(-50, -20)
    p_lin = db2lin(noise_db)
    n = torch.sqrt(torch.tensor(p_lin/2, device=gen.device)) * \
        (torch.randn(gen.N, device=gen.device) + 1j*torch.randn(gen.N, device=gen.device))
    return n.to(torch.complex64), []



# Bu ana fonksiyon
def generate_realistic_uhf_dataset(
    out_dir=OUT_DIR, num_samples=NUM_SAMPLES, proportions=PROPORTIONS,
    fs=FS, duration=DURATION, n_fft=N_FFT, n_overlap=N_OVERLAP, shard_size=SHARD_SIZE, seed=SEED
):
    # 1. Başlangıç ayarlarını yap
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = TorchIQGenRealistic(fs=fs, duration=duration, device=device)

    print(f"🧠 Cihaz: {device} (CUDA {'AKTİF' if device.type=='cuda' else 'DEVRE DIŞI'})")

    # 2. Üretim planını oluştur
    counts = {k:int(round(v*num_samples)) for k,v in proportions.items()}
    diff = num_samples - sum(counts.values())
    if diff != 0: counts['mixed_close'] = counts.get('mixed_close',0) + diff

    plan = (['noise']*counts['noise'] + ['single']*counts['single'] +
            ['mixed_close']*counts['mixed_close'] + ['mixed_far']*counts['mixed_far'])
    random.shuffle(plan)

    # 3. Manifest dosyası
    manifest = {
        "created": datetime.now().isoformat(),
        "seed": seed,
        "baseband_fs": fs,
        "duration_s": duration,
        "snr_range_db_total": [0, 30],
        "stft": {"n_fft": n_fft, "noverlap": n_overlap, "hop": n_fft-n_overlap},
        "num_samples": num_samples,
        "mods": MODS,
        "impairments_added": ["IQ imbalance", "Phase noise", "Rician Fading (reduced) + per-path Doppler"],
        "version": "R2.1_torch_gpu_simplified_rician_commented",
    }
    
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 4. Veri üretme ve kaydetme döngüsü
    shard_idx = 0
    buf_features, buf_iq_raw, buf_labels = [], [], [] # Geçici depolama listeleri
    stats = {"noise":0, "single":0, "mixed_close":0, "mixed_far":0} # İstatistikler

    pbar = tqdm(plan, desc="Basitleştirilmiş UHF veri seti oluşturuluyor", unit="sample")
    for cat in pbar:
        # Her örnek için rastgele bir UHF taşıyıcı frekansı atanır
        if cat == 'noise': abs_fc_list = []
        elif cat == 'single': abs_fc_list = [float(np.random.uniform(UHF_MIN, UHF_MAX))]
        else:
            k = random.randint(2,4); base_fc = float(np.random.uniform(UHF_MIN, UHF_MAX))
            abs_fc_list = [float(np.clip(base_fc + np.random.uniform(-5e6, 5e6), UHF_MIN, UHF_MAX)) for _ in range(k)]

        # Plana göre ilgili sinyal üretici fonksiyonunu çağır
        if   cat=='single':      iq, info = gen_single(gen)
        elif cat=='mixed_close': iq, info = gen_mixed(gen, close=True)
        elif cat=='mixed_far':   iq, info = gen_mixed(gen, close=False)
        else:                    iq, info = gen_noise(gen)

        # AWGN gürültüsü ekle
        if cat != 'noise':
            snr_db = float(np.random.uniform(0, 30))
            iq = gen.add_awgn_total(iq, snr_db)
        else:
            snr_db = -np.inf

        # Sinyalden 4 kanallı özellikleri çıkar
        mag, ph, ifs, phg = gen.compute_spectrograms(iq, n_fft=n_fft, noverlap=n_overlap)
        feature_stack = torch.stack([mag, ph, ifs, phg], dim=0).to(torch.float32)

        # Üretilen verileri geçici listelere ekle
        buf_features.append(to_cpu_np(feature_stack))
        buf_iq_raw.append(to_cpu_np(iq.to(torch.complex64)))
        buf_labels.append({ "type": cat, "num_signals": len(info), "signals": info,
                            "snr_db_total": snr_db, "abs_fc_list_hz": abs_fc_list })
        stats[cat] += 1

        # Buffer  dolduğunda, verileri shard olarak diske yaz
        if len(buf_features) >= shard_size:
            shard_path = os.path.join(out_dir, f"shard_{shard_idx:05d}")
            os.makedirs(shard_path, exist_ok=True)

            np.save(os.path.join(shard_path, "features.npy"), np.stack(buf_features, axis=0))
            np.save(os.path.join(shard_path, "iq_raw.npy"),    np.stack(buf_iq_raw, axis=0))
            # Etiketleri pickle dosyası olarak kaydet
            with open(os.path.join(shard_path, "labels.pkl"), "wb") as f: pickle.dump(buf_labels, f)

            buf_features, buf_iq_raw, buf_labels = [], [], []
            shard_idx += 1

    # Döngü bittikten sonra buffer'da kalan son verileri de kaydet
    if buf_features:
        shard_path = os.path.join(out_dir, f"shard_{shard_idx:05d}")
        os.makedirs(shard_path, exist_ok=True)
        np.save(os.path.join(shard_path, "features.npy"), np.stack(buf_features, axis=0))
        np.save(os.path.join(shard_path, "iq_raw.npy"),    np.stack(buf_iq_raw, axis=0))
        with open(os.path.join(shard_path, "labels.pkl"), "wb") as f: pickle.dump(buf_labels, f)

    # 5. Sonuçları ekrana yazdır
    print("\n" + "="*50)
    print("✅ BASİTLEŞTİRİLMİŞ VERİ SETİ OLUŞTURMA TAMAMLANDI")
    print(f"📁 Çıktı dizini: {out_dir}")
    print(f"📊 Toplam örnek sayısı: {sum(stats.values())}")

# Betik doğrudan çalıştırıldığında bu ana fonksiyonu çağır
if __name__ == "__main__":
    generate_realistic_uhf_dataset()