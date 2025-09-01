
import numpy as np  
import matplotlib.pyplot as plt  #
from scipy import signal  
import pickle 
import os  
from datetime import datetime  
import multiprocessing as mp  
from tqdm import tqdm  
import h5py  


# Sinyal üretimi için temel parametre ayarları.
class OptimizedSignalGenerator:
    def __init__(self, sample_rate=2e6, duration=0.002):  # Örnekleme frekansı 2MHz, sinyal süresi 2ms
        self.sample_rate = sample_rate                    # Saniyedeki örnek sayısı
        self.duration = duration                          # Her bir sinyalin saniye cinsinden uzunluğu
        self.num_samples = int(sample_rate * duration)    # Toplam örnek sayısı (süre * örnekleme frekansı)
        self.time = np.linspace(0, duration, self.num_samples, endpoint=False)  # Sinyal için zaman vektörü oluştur
        
    # AM modülasyon sinyali
    def generate_am_signal(self, carrier_freq, mod_freq, mod_depth=0.5, amplitude=1.0):
        carrier = np.cos(2 * np.pi * carrier_freq * self.time)  # Taşıyıcı sinyali 
        modulation = np.cos(2 * np.pi * mod_freq * self.time)  # Modüle eden sinyal 
        am_signal = amplitude * (1 + mod_depth * modulation) * carrier  # AM formülü
        
        return am_signal, {
            'type': 'AM',
            'carrier_freq': carrier_freq,
            'mod_freq': mod_freq,
            'mod_depth': mod_depth,
            'amplitude': amplitude,
            'bandwidth': 2 * mod_freq  
        }
    
        # FM modülasyon sinyali
    def generate_fm_signal(self, carrier_freq, mod_freq, freq_deviation=5000, amplitude=1.0):
        
        # FM sinyalinin anlık fazını hesabı
        phase = 2 * np.pi * carrier_freq * self.time + (freq_deviation / mod_freq) * np.sin(2 * np.pi * mod_freq * self.time)
        fm_signal = amplitude * np.cos(phase)  # FM sinyalini oluştur
        
        return fm_signal, {
            'type': 'FM',
            'carrier_freq': carrier_freq,
            'mod_freq': mod_freq,
            'freq_deviation': freq_deviation,
            'amplitude': amplitude,
            'bandwidth': 2 * (freq_deviation + mod_freq)  # Carson Kuralı ile bant genişliği tahmini
        }
     # PSK modülasyon sinyali
    def generate_psk_signal(self, carrier_freq, symbol_rate, num_symbols=4, amplitude=1.0):
    
        num_data_symbols = int(self.duration * symbol_rate)  # Sinyal süresince üretilecek sembol sayısı
        symbols = np.random.choice(num_symbols, size=num_data_symbols)  # Rastgele dijital semboller 
        
        # Sembol sayısına göre faz eşlemesi yap (BPSK, QPSK, 8PSK)
        if num_symbols == 4:  # QPSK için fazlar (0, 90, 180, 270 derece)
            phase_map = {0: 0, 1: np.pi/2, 2: np.pi, 3: 3*np.pi/2}
        elif num_symbols == 2:  # BPSK için fazlar (0, 180 derece)
            phase_map = {0: 0, 1: np.pi}
        elif num_symbols == 8:  # 8PSK için fazlar
            phase_map = {i: i * 2 * np.pi / 8 for i in range(8)}
        
        samples_per_symbol = int(self.sample_rate / symbol_rate)  # Her bir sembolün kaç örnekten oluşacağı
        psk_signal = np.zeros(self.num_samples)  
        
        # Her bir sembol için döngüye gir ve karşılık gelen sinyal parçasını üret
        for i, sym in enumerate(symbols):
            start_idx = i * samples_per_symbol  # Sembol başlangıç indeksi
            end_idx = min(start_idx + samples_per_symbol, self.num_samples)  # Sembol bitiş indeksi
            
            if start_idx < self.num_samples:
                phase = phase_map[sym]  # Sembole karşılık gelen fazı al
                segment_time = self.time[start_idx:end_idx] - (start_idx / self.sample_rate)
                # Taşıyıcı sinyali ilgili fazda oluştur ve ana sinyal dizisine ekle
                psk_signal[start_idx:end_idx] = amplitude * np.cos(
                    2 * np.pi * carrier_freq * segment_time + phase
                )
        
        return psk_signal, {
            'type': f'{num_symbols}PSK',
            'carrier_freq': carrier_freq,
            'symbol_rate': symbol_rate,
            'num_symbols': num_symbols,
            'amplitude': amplitude,
            'bandwidth': symbol_rate, 
            'symbols': symbols
        }
    
    # OFDM modülasyon sinyali
    def generate_ofdm_signal(self, carrier_freq, num_subcarriers=64, symbol_rate=1000, amplitude=1.0):
        """OFDM (Ortogonal Frekans Bölmeli Çoklama) sinyali üretir."""
        subcarrier_spacing = symbol_rate  
        
        # Her alt taşıyıcı için rastgele veri sembolleri oluştur 
        data_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], 
                                       size=(num_subcarriers, int(self.duration * symbol_rate)))
        
        ofdm_signal = np.zeros(self.num_samples, dtype=complex)  
        
        # Her bir OFDM sembolü için döngüye gir
        for symbol_idx in range(data_symbols.shape[1]):
            ofdm_symbol = np.fft.ifft(data_symbols[:, symbol_idx], n=num_subcarriers)
            cp_length = num_subcarriers // 4  
            ofdm_with_cp = np.concatenate([ofdm_symbol[-cp_length:], ofdm_symbol])
            
            # Oluşturulan sembolü ana sinyal dizisine yerleştir
            start_idx = symbol_idx * len(ofdm_with_cp)
            end_idx = min(start_idx + len(ofdm_with_cp), self.num_samples)
            
            if start_idx < self.num_samples:
                length = end_idx - start_idx
                ofdm_signal[start_idx:end_idx] = ofdm_with_cp[:length]
        
        # Karmaşık temel bant sinyalini, taşıyıcı frekansına modüle ederek gerçek sinyale dönüştür.
        real_ofdm = amplitude * np.real(ofdm_signal * np.exp(1j * 2 * np.pi * carrier_freq * self.time[:len(ofdm_signal)]))
        
        # Eğer üretilen sinyal istenen uzunluktan kısaysa, sonunu sıfırlarla doldur.
        if len(real_ofdm) < self.num_samples:
            real_ofdm = np.pad(real_ofdm, (0, self.num_samples - len(real_ofdm)))
        
        return real_ofdm, {
            'type': 'OFDM',
            'carrier_freq': carrier_freq,
            'num_subcarriers': num_subcarriers,
            'symbol_rate': symbol_rate,
            'amplitude': amplitude,
            'bandwidth': num_subcarriers * subcarrier_spacing
        }
    

    def generate_noise_signal(self, noise_type='white', amplitude=1.0):
        """Farklı istatistiksel özelliklere sahip gürültü sinyalleri üretir."""
        if noise_type == 'white':  # Beyaz gürültü: tüm frekanslarda eşit güç
            noise_signal = amplitude * np.random.randn(self.num_samples)
        elif noise_type == 'pink':  # Pembe gürültü: frekans düştükçe gücü artar (1/f gürültüsü)
            freqs = np.fft.fftfreq(self.num_samples, 1/self.sample_rate)
            freqs[0] = 1e-10  # Sıfıra bölme hatasını önle
            pink_spectrum = 1 / np.sqrt(np.abs(freqs))  # Pembe gürültü spektrumu
            white_noise_fft = np.fft.fft(np.random.randn(self.num_samples))
            pink_noise_fft = white_noise_fft * pink_spectrum
            noise_signal = amplitude * np.real(np.fft.ifft(pink_noise_fft)) # Ters FFT ile zaman domenine dön
        elif noise_type == 'colored':  # Renkli gürültü: belirli bir frekans bandında yoğunlaşmış gürültü
            low_freq = np.random.uniform(1000, 10000)
            high_freq = np.random.uniform(low_freq + 5000, 100000)
            white = np.random.randn(self.num_samples)
            # Belirtilen bant aralığı için bir bant geçiren filtre  tasarla
            sos = signal.butter(4, [low_freq, high_freq], btype='band', fs=self.sample_rate, output='sos')
            noise_signal = amplitude * signal.sosfilt(sos, white) # Beyaz gürültüyü filtrele
        
        return noise_signal, {
            'type': f'{noise_type.upper()}_NOISE',
            'amplitude': amplitude,
            'noise_type': noise_type
        }
    
    def add_realistic_impairments(self, clean_signal, impairments=None):
        if impairments is None:
            impairments = ['awgn', 'phase_noise', 'freq_offset']
        
        impaired_signal = clean_signal.copy() # Orijinal sinyali korumak için kopyasını al
        impairment_params = {} # Eklenen bozulmaların parametrelerini saklamak için sözlük
        
        # Bozulma uygulanması
        for impairment in impairments:
            if impairment == 'awgn':  #  Beyaz Gauss Gürültüsü
                snr_db = np.random.uniform(5, 30)  # Rastgele bir SNR seç (5-30 dB)
                signal_power = np.mean(impaired_signal**2)  # Sinyalin gücünü hesapla
                noise_power = signal_power / (10**(snr_db/10))  # Gerekli gürültü gücünü hesapla
                noise = np.sqrt(noise_power) * np.random.randn(len(impaired_signal)) # Gürültüyü oluştur
                impaired_signal += noise  # Gürültüyü sinyale ekle
                impairment_params['awgn_snr_db'] = snr_db
                
            elif impairment == 'phase_noise': # Faz Gürültüsü
                phase_noise_std = np.random.uniform(0.01, 0.1) # Fazdaki sapmanın standart sapmasını rastgele seç
                phase_noise = np.cumsum(np.random.randn(len(impaired_signal)) * phase_noise_std)
                impaired_signal *= np.exp(1j * phase_noise).real # Faz hatasını sinyale uygula
                impairment_params['phase_noise_std'] = phase_noise_std
                
            elif impairment == 'freq_offset': # Frekans Kayması
                freq_offset = np.random.uniform(-1000, 1000)  # -1kHz ile +1kHz arasında rastgele bir frekans kayması seç
                offset_phase = 2 * np.pi * freq_offset * self.time 
                impaired_signal *= np.cos(offset_phase) # Frekans kaymasını sinyale uygula
                impairment_params['freq_offset_hz'] = freq_offset
                

        return impaired_signal, impairment_params
    
    def generate_batch_worker(self, args):
        batch_id, batch_size, signal_types, freq_ranges, add_impairments = args 
        
        batch_data = [] # sinyalleri tutacak liste
        
        # İstenen sayıda sinyal üretmek için döngü
        for i in range(batch_size):
            # Rastgele bir sinyal türü ve frekans bandı seç
            signal_type = np.random.choice(signal_types)
            band_name = np.random.choice(list(freq_ranges.keys()))
            freq_min, freq_max = freq_ranges[band_name]
            
            # Rastgele sinyal parametreleri 
            carrier_freq = np.random.uniform(freq_min, freq_max)
            amplitude = np.random.uniform(0.1, 2.0)
            
            # Seçilen sinyal türüne göre ilgili üretim fonksiyonunu çağır
            if signal_type == 'AM':
                mod_freq = np.random.uniform(100, 8000)
                mod_depth = np.random.uniform(0.1, 0.9)
                signal_data, params = self.generate_am_signal(carrier_freq, mod_freq, mod_depth, amplitude)
            # ... (diğer sinyal türleri için benzer bloklar) ...
            elif signal_type == 'FM':
                mod_freq = np.random.uniform(100, 8000)
                freq_dev = np.random.uniform(1000, 15000)
                signal_data, params = self.generate_fm_signal(carrier_freq, mod_freq, freq_dev, amplitude)
            elif signal_type == 'PSK':
                symbol_rate = np.random.uniform(1000, 50000)
                num_symbols = np.random.choice([2, 4, 8])
                signal_data, params = self.generate_psk_signal(carrier_freq, symbol_rate, num_symbols, amplitude)
            elif signal_type == 'OFDM':
                num_subcarriers = np.random.choice([16, 32, 64, 128])
                symbol_rate = np.random.uniform(500, 5000)
                signal_data, params = self.generate_ofdm_signal(carrier_freq, num_subcarriers, symbol_rate, amplitude)
            elif signal_type == 'NOISE':
                noise_type = np.random.choice(['white', 'pink', 'colored'])
                signal_data, params = self.generate_noise_signal(noise_type, amplitude)
            
            # Bozulma ekleme seçeneği aktifse ve %70 ihtimalle sinyale bozulma ekle
            if add_impairments and np.random.random() > 0.3:
                # 1 ila 3 arasında rastgele sayıda bozulma türü seç
                impairments = np.random.choice(['awgn', 'phase_noise', 'freq_offset', 'multipath'], 
                                             size=np.random.randint(1, 4), replace=False)
                signal_data, impairment_params = self.add_realistic_impairments(signal_data, impairments)
                params.update(impairment_params) # Bozulma parametrelerini ana parametre sözlüğüne ekle
            
            # Ek meta verileri ekle
            params['band'] = band_name
            params['signal_id'] = batch_id * batch_size + i # Her sinyale ID verilir
            params['sample_rate'] = self.sample_rate
            params['duration'] = self.duration
            
            # Sinyali ve parametrelerini içeren sözlüğü listeye ekle
            batch_data.append({'signal': signal_data, 'params': params})
        
        return batch_data # Üretilen sinyal grubunu döndür
    
    def generate_massive_dataset(self, total_samples=500000, output_dir='massive_signals', 
                               batch_size=1000, num_processes=None, use_hdf5=True):
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if num_processes is None:
            num_processes = max(1, mp.cpu_count() - 1)
        
        # Üretilecek sinyal türleri ve frekans aralıklarını tanımla
        signal_types = ['AM', 'FM', 'PSK', 'OFDM', 'NOISE']
        freq_ranges = {
            'HF': (3e6, 30e6),       # 3-30 MHz
            'VHF': (30e6, 300e6),    # 30-300 MHz  
            'UHF': (300e6, 1e9),     # 300MHz-1GHz
            'L_BAND': (1e9, 2e9),    # 1-2 GHz
            'S_BAND': (2e9, 4e9)     # 2-4 GHz (sadece simülasyon için)
        }
        

        print(f"🚀 Büyük Veri Seti Üretimi Başladı")
        print(f"📊 Hedef: {total_samples:,} örnek")
        print(f"⚡ {num_processes} işlemci çekirdeği kullanılıyor")
        print(f"📦 Grup boyutu: {batch_size}")
        print(f"💾 Depolama formatı: {'HDF5' if use_hdf5 else 'Pickle'}")
        print(f"🎯 Sinyal türleri: {signal_types}")
        print(f"📡 Frekans bantları: {list(freq_ranges.keys())}")
        
        # İşçi fonksiyonlarına gönderilecek argümanları hazırla
        num_batches = (total_samples + batch_size - 1) // batch_size # Toplam grup sayısı
        batch_args = []
        
        for batch_id in range(num_batches):
            current_batch_size = min(batch_size, total_samples - batch_id * batch_size) # Son grup daha küçük olabilir
            batch_args.append((batch_id, current_batch_size, signal_types, freq_ranges, True))
        
        # Grupları paralel olarak işle
        all_data = [] # Tüm üretilen verileri tutacak liste
        
        if num_processes > 1: # Eğer birden fazla çekirdek kullanılacaksa
            with mp.Pool(num_processes) as pool: # İşlemci havuzu oluştur
                # tqdm ile ilerleme çubuğu göstererek işçi fonksiyonunu tüm argümanlarla çalıştır
                with tqdm(total=num_batches, desc="Gruplar üretiliyor") as pbar:
                    for batch_data in pool.imap(self.generate_batch_worker, batch_args):
                        all_data.extend(batch_data) # Gelen sonuçları ana listeye ekle
                        pbar.update(1) # İlerleme çubuğunu güncelle
        else: # Sadece tek çekirdek kullanılacaksa (hata ayıklama için)
            for args in tqdm(batch_args, desc="Gruplar üretiliyor"):
                batch_data = self.generate_batch_worker(args)
                all_data.extend(batch_data)
        
        # Veri setini dosyaya kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Dosya adı için zaman damgası
        
        if use_hdf5: # HDF5 formatında kaydet
            dataset_file = f"{output_dir}/massive_dataset_{timestamp}.h5"
            self.save_hdf5_dataset(all_data, dataset_file)
        else: # Pickle formatında kaydet
            dataset_file = f"{output_dir}/massive_dataset_{timestamp}.pkl"
            with open(dataset_file, 'wb') as f:
                pickle.dump(all_data, f, protocol=4)  # Büyük dosyalar için protokol 4
        
        # Veri seti hakkında özet oluştur
        summary = self.generate_dataset_summary(all_data, timestamp, dataset_file)
        summary_file = f"{output_dir}/summary_{timestamp}.json"
        
        import json # JSON formatında kaydetmek için
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2) # Okunabilir formatta JSON dosyasına yaz
        
        # İşlem tamamlandı mesajlarını yazdır
        print(f"\n🎉 Veri seti üretimi tamamlandı!")
        print(f"📊 Üretilen sinyal sayısı: {len(all_data):,}")
        print(f"📁 Veri seti dosyası: {dataset_file}")
        print(f"📋 Özet dosyası: {summary_file}")
        print(f"💾 Dosya boyutu: {os.path.getsize(dataset_file) / (1024**3):.2f} GB")
        
        return all_data, summary
    
    def save_hdf5_dataset(self, dataset, filename):
        """Veri setini HDF5 formatında verimli bir şekilde kaydeder."""
        with h5py.File(filename, 'w') as f: # HDF5 dosyasını yazma modunda aç
            # Sinyaller için bir grup oluştur
            signals_group = f.create_group('signals')
            
            # Verimlilik için dizileri önceden oluştur
            num_signals = len(dataset)
            signal_data = np.zeros((num_signals, self.num_samples), dtype=np.float32) # Sinyal verileri (daha az yer kaplaması için float32)
            
            # Parametre dizileri
            signal_types = []
            carrier_freqs = np.zeros(num_signals)
            amplitudes = np.zeros(num_signals)
            bands = []
            signal_ids = np.zeros(num_signals, dtype=int)
            
            # Veri setindeki her bir eleman için döngüye gir ve dizileri doldur
            for i, item in enumerate(dataset):
                signal_data[i] = item['signal'].astype(np.float32)
                signal_types.append(item['params']['type'])
                carrier_freqs[i] = item['params'].get('carrier_freq', 0) # Eğer taşıyıcı frekans yoksa (örn: gürültü) 0 ata
                amplitudes[i] = item['params']['amplitude']
                bands.append(item['params']['band'])
                signal_ids[i] = item['params']['signal_id']
            
            # Doldurulan NumPy dizilerini HDF5 dosyasına veri setleri olarak kaydet
            signals_group.create_dataset('data', data=signal_data, compression='gzip') # Sıkıştırma uygula
            signals_group.create_dataset('types', data=[s.encode() for s in signal_types]) # Stringleri byte olarak kaydet
            signals_group.create_dataset('carrier_freqs', data=carrier_freqs)
            signals_group.create_dataset('amplitudes', data=amplitudes)
            signals_group.create_dataset('bands', data=[b.encode() for b in bands])
            signals_group.create_dataset('signal_ids', data=signal_ids)
            
            # Dosyanın geneli için meta verileri kaydet
            f.attrs['sample_rate'] = self.sample_rate
            f.attrs['duration'] = self.duration
            f.attrs['num_samples'] = self.num_samples
            f.attrs['total_signals'] = num_signals
    
    def generate_dataset_summary(self, dataset, timestamp, filename):
        """Veri seti hakkında istatistiksel bir özet oluşturur."""
        signal_type_counts = {} # Sinyal türlerinin sayısını tutacak sözlük
        band_counts = {} # Frekans bantlarının sayısını tutacak sözlük
        
        # Veri setini dolaşarak sayımları yap
        for item in dataset:
            sig_type = item['params']['type']
            band = item['params']['band']
            
            signal_type_counts[sig_type] = signal_type_counts.get(sig_type, 0) + 1
            band_counts[band] = band_counts.get(band, 0) + 1
        
        # Tüm bilgileri yapılandırılmış bir sözlük olarak döndür
        return {
            'generation_info': {
                'timestamp': timestamp,
                'total_signals': len(dataset),
                'sample_rate_hz': self.sample_rate,
                'duration_sec': self.duration,
                'samples_per_signal': self.num_samples
            },
            'file_info': {
                'filename': filename,
                'size_bytes': os.path.getsize(filename),
                'size_gb': os.path.getsize(filename) / (1024**3)
            },
            'distribution': {
                'signal_types': signal_type_counts,
                'frequency_bands': band_counts
            },
            'estimated_training_time': { # Model eğitimi için kaba bir zaman tahmini
                'gpu_hours_cnn': len(dataset) / 10000,
                'gpu_hours_transformer': len(dataset) / 5000
            }
        }


if __name__ == "__main__":
    # Sinyal üretici nesnesi
    generator = OptimizedSignalGenerator(sample_rate=2e6, duration=0.002)
    
    dataset, summary = generator.generate_massive_dataset(
        total_samples=500000, # Toplam 500,000 sinyal üret
        batch_size=2000, # Her bir işlemci grubunda 2000 sinyal üret
        num_processes=mp.cpu_count()-1, # Bir çekirdek hariç tümünü kullan
        use_hdf5=True # HDF5 formatında kaydet
    )
    
    # Üretim sonu
    print("\n📈 Veri Seti Özeti:")
    print(f"Sinyal Türleri: {summary['distribution']['signal_types']}")
    print(f"Frekans Bantları: {summary['distribution']['frequency_bands']}")
    print(f"Tahmini CNN Eğitim Süresi: {summary['estimated_training_time']['gpu_hours_cnn']:.1f} saat")