# -*- coding: utf-8 -*-
"""
UHF Sinyal Analiz ve Sınıflandırma Uygulaması
CFO ve Doppler etkisi birleştirilip otomatik hale getirilmiş nihai versiyon.
"""

# Standart kütüphaneler
import random

# Bilimsel hesaplama kütüphaneleri
import numpy as np
import matplotlib.pyplot as plt
import torch

# Web arayüz kütüphaneleri
import gradio as gr

# ==============================================================================
# === BÖLÜM 1: GEREKLİ KODLARDAN FONKSİYONLARI DOĞRUDAN IMPORT ETME ==========
# ==============================================================================

try:
    from uhf_sinyal_uretme_11_2 import (
        UltimateTorchIQGenerator, gen_noise_only, gen_single,
        FS, DURATION, MODS, RF_LIMITS, MAX_SIGNAL,
        CLOSE_OFFSET_HZ, FAR_OFFSET_FRAC,
        _get_random_tx_scenario, measure_spectral_metrics
    )
    from uhf_dataset_psd_donusturme import (
        compute_welch_psd, interp_to_uniform_grid, log_and_norm
    )
except ImportError as e:
    raise ImportError(
        f"HATA: Gerekli bir motor dosyası bulunamadı! ({e}). "
        f"Lütfen tüm .py dosyalarının aynı klasörde olduğundan emin olun."
    )

# ==============================================================================
# === BÖLÜM 2: GÖRSELLEŞTİRME VE ANA MANTIK ================================
# ==============================================================================


def generate_custom_mixed_signal(gen, num_signals, mod_list, distance):
    """
    Özelleştirilmiş karışık sinyal üretir.
    """
    try:
        k = num_signals
        main_power = random.uniform(-5, 0)
        other_powers = [
            main_power + random.uniform(-15, -5) for _ in range(k - 1)
        ]
        rel_powers_db = [main_power] + sorted(other_powers, reverse=True)
        
        sigs, info = [], []
        
        for idx in range(k):
            mod = mod_list[idx]
            p_db = rel_powers_db[idx]
            scenario = _get_random_tx_scenario()
            
            # Modülasyon tipine göre sinyal üretimi
            if mod == 'FM':
                s, meta = gen.gen_fm(p_db, tx_scenario=scenario)
            elif mod == 'OFDM':
                s, meta = gen.gen_ofdm(p_db, tx_scenario=scenario)
            elif mod == 'GFSK':
                s, meta = gen.gen_gfsk(p_db, tx_scenario=scenario)
            else:
                s, meta = gen.gen_qpsk(p_db, tx_scenario=scenario)
            
            # Frekans offset hesaplama
            if distance == "Yakın":
                foff = random.uniform(*CLOSE_OFFSET_HZ) * random.choice([-1, 1])
            else:
                foff = (random.uniform(*FAR_OFFSET_FRAC) * 
                       random.choice([-1, 1]) * gen.fs)
            
            # Sinyal kaydırma
            s_shifted = (s * torch.exp(1j * 2 * np.pi * foff * gen.t)).to(
                torch.complex64
            )
            
            # Spektral metrik ölçümü
            f_center_est, bw_occ, _, _ = measure_spectral_metrics(
                s_shifted, FS
            )
            
            # Metadata güncelleme
            meta['mod'] = mod
            meta['f_off_hz'] = float(foff)
            meta['rel_power_db'] = float(p_db)
            meta['bw_occ99_hz'] = bw_occ
            meta['f_center_est_hz'] = f_center_est
            
            info.append(meta)
            sigs.append(s_shifted)
        
        return sum(sigs), info
        
    except Exception as e:
        raise RuntimeError(f"Karışık sinyal üretiminde hata: {e}")


def create_true_labels_mask(signals_metadata, psd_freq_axis):
    """
    Gerçek sinyal konumları için maske oluşturur.
    """
    mask = np.zeros_like(psd_freq_axis)
    
    for signal in signals_metadata:
        # Gürültü sinyallerini atla
        if (signal.get('mod') == 'NOISE' or 
            'bw_occ99_hz' not in signal):
            continue
            
        center_freq = signal.get('f_off_hz', 0.0)
        bandwidth = signal.get('bw_occ99_hz', 0.0)
        
        if bandwidth > 0:
            start_freq = center_freq - bandwidth / 2
            end_freq = center_freq + bandwidth / 2
            mask[(psd_freq_axis >= start_freq) & 
                 (psd_freq_axis <= end_freq)] = 1.0
    
    return mask


def create_plots(sample_data, psd_final, freqs_for_psd):
    """
    Ana ve ek analiz grafiklerini oluşturur.
    """
    try:
        iq_np = sample_data['iq']
        fs = sample_data['fs']
        
        # --- 1. Ana Grafikler ---
        fig_main = plt.figure(figsize=(12, 8))
        gs = fig_main.add_gridspec(2, 2, height_ratios=[2, 1.5])
        
        # Spektrogram
        ax1 = fig_main.add_subplot(gs[0, :])
        ax1.specgram(iq_np, NFFT=256, Fs=fs, noverlap=128, 
                     cmap='viridis', scale='dB')
        ax1.set_title("Sinyal Spektrogramı")
        ax1.set_xlabel("Zaman (s)")
        ax1.set_ylabel("Frekans (Hz)")
        
        # İşlenmiş Welch PSD
        ax2 = fig_main.add_subplot(gs[1, 0])
        ax2.plot(freqs_for_psd / 1e3, psd_final, color='royalblue')
        ax2.set_title("İşlenmiş Welch PSD")
        ax2.set_xlabel("Frekans (kHz)")
        ax2.set_ylabel("İşlenmiş Güç (dB)")
        ax2.grid(True, alpha=0.5)
        
        # Gerçek sinyal konumu
        ax3 = fig_main.add_subplot(gs[1, 1])
        true_labels_mask = create_true_labels_mask(
            sample_data['signals'], freqs_for_psd
        )
        ax3.plot(freqs_for_psd / 1e3, true_labels_mask, 
                 color='green', linewidth=2)
        ax3.fill_between(freqs_for_psd / 1e3, 0, true_labels_mask, 
                         color='green', alpha=0.3)
        ax3.set_title("Gerçek Sinyal Konumu")
        ax3.set_xlabel("Frekans (kHz)")
        ax3.set_ylabel("Sinyal Varlığı")
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, alpha=0.5)
        
        fig_main.tight_layout()
        
        # --- 2. Ek Analiz Grafikleri ---
        fig_extra = plt.figure(figsize=(12, 8))
        gs_extra = fig_extra.add_gridspec(2, 3)
        times_ms = np.arange(len(iq_np)) / fs * 1e3
        
        # Üst sıra grafikleri
        _create_constellation_plot(fig_extra, gs_extra, iq_np)
        _create_envelope_plot(fig_extra, gs_extra, iq_np, times_ms)
        _create_instant_frequency_plot(fig_extra, gs_extra, iq_np, times_ms, fs)
        
        # Alt sıra grafikleri
        _create_histogram_plots(fig_extra, gs_extra, iq_np)
        _create_phase_plot(fig_extra, gs_extra, iq_np, times_ms)
        
        fig_extra.tight_layout()
        
        return fig_main, fig_extra
        
    except Exception as e:
        raise RuntimeError(f"Grafik oluşturma hatası: {e}")


def _create_constellation_plot(fig, gs, iq_np):
    """
    IQ constellation grafiği oluşturur.
    """
    ax_iq = fig.add_subplot(gs[0, 0])
    ax_iq.scatter(iq_np.real[::10], iq_np.imag[::10], 
                  alpha=0.6, s=1, color='darkcyan')
    ax_iq.set_title('IQ Constellation')
    ax_iq.set_xlabel('I')
    ax_iq.set_ylabel('Q')
    ax_iq.grid(True, alpha=0.3)
    ax_iq.axhline(0, color='grey', lw=0.5)
    ax_iq.axvline(0, color='grey', lw=0.5)


def _create_envelope_plot(fig, gs, iq_np, times_ms):
    """
    Anlık genlik grafiği oluşturur.
    """
    ax_env = fig.add_subplot(gs[0, 1])
    ax_env.plot(times_ms[:500], np.abs(iq_np[:500]), color='coral')
    ax_env.set_title('Anlık Genlik (İlk 500 Örnek)')
    ax_env.set_xlabel('Zaman (ms)')
    ax_env.set_ylabel('Genlik')
    ax_env.grid(True, alpha=0.3)


def _create_instant_frequency_plot(fig, gs, iq_np, times_ms, fs):
    """
    Anlık frekans grafiği oluşturur.
    """
    ax_if = fig.add_subplot(gs[0, 2])
    phase_diff = np.diff(np.unwrap(np.angle(iq_np)))
    inst_freq = phase_diff * fs / (2 * np.pi)
    ax_if.plot(times_ms[1:1001], inst_freq[:1000] / 1e3, 
               color='mediumorchid')
    ax_if.set_title('Anlık Frekans')
    ax_if.set_xlabel('Zaman (ms)')
    ax_if.set_ylabel('Frekans (kHz)')
    ax_if.grid(True, alpha=0.3)


def _create_histogram_plots(fig, gs, iq_np):
    """
    I ve Q değerleri histogramlarını oluşturur.
    """
    # I değerleri histogramı
    ax_i_hist = fig.add_subplot(gs[1, 0])
    ax_i_hist.hist(iq_np.real, bins=100, color='c', density=True)
    ax_i_hist.set_title("I Değerleri Histogramı")
    ax_i_hist.set_xlabel("Genlik")
    ax_i_hist.set_ylabel("Yoğunluk")
    
    # Q değerleri histogramı
    ax_q_hist = fig.add_subplot(gs[1, 1])
    ax_q_hist.hist(iq_np.imag, bins=100, color='m', density=True)
    ax_q_hist.set_title("Q Değerleri Histogramı")
    ax_q_hist.set_xlabel("Genlik")


def _create_phase_plot(fig, gs, iq_np, times_ms):
    """
    Zamanla faz değişimi grafiği oluşturur.
    """
    ax_phase = fig.add_subplot(gs[1, 2])
    ax_phase.plot(times_ms, np.unwrap(np.angle(iq_np)), color='orange')
    ax_phase.set_title("Zamanla Faz Değişimi")
    ax_phase.set_xlabel("Zaman (ms)")
    ax_phase.set_ylabel("Radyan")


def run_analysis(
    signal_type, mod_pool_single,
    num_signals_choice, distance_choice, 
    mod_choice_1, mod_choice_2, mod_choice_3,
    apply_cfo_and_doppler,
    apply_clock_drift, drift_choice, drift_val,
    apply_memory, memory_choice, memory_val,
    snr_choice, snr_val
):
    """
    Ana analiz fonksiyonu - sinyal üretimi ve bozulma uygulaması yapar.
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gen = UltimateTorchIQGenerator(fs=FS, duration=DURATION, device=device)
        iq_torch, meta_list = None, []

        # Sinyal tipi seçimine göre üretim
        if signal_type == "Gürültü":
            iq_torch, meta_list = gen_noise_only(gen)
        elif signal_type == "Tek Sinyal":
            original_mods = list(MODS)
            MODS[:] = mod_pool_single if mod_pool_single else original_mods
            iq_torch, meta_list = gen_single(gen)
            MODS[:] = original_mods
        elif signal_type == "Çoklu Sinyal":
            n_signals = 2 if num_signals_choice == "2 Sinyal" else 3
            mods_to_generate = [mod_choice_1, mod_choice_2, mod_choice_3][:n_signals]
            iq_torch, meta_list = generate_custom_mixed_signal(
                gen, num_signals=n_signals, 
                mod_list=mods_to_generate, 
                distance=distance_choice
            )

        # --- Gelişmiş bozulmaları sırayla uygula ---

        # CFO ve Doppler etkisini otomatik uygula
        if apply_cfo_and_doppler:
            iq_torch, cfo_meta = gen.apply_cfo_and_slow_doppler(iq_torch)
            meta_list.append({'cfo_and_doppler_effects': cfo_meta})

        # Saat sapması uygula
        if apply_clock_drift:
            drift_ppm = (float(drift_val) if drift_choice == "Özel" 
                        else random.uniform(*RF_LIMITS['clock_drift_ppm']))
            iq_torch, _ = gen.apply_clock_drift(iq_torch)
        
        # Hafıza etkileri uygula
        if apply_memory:
            depth = (int(memory_val) if memory_choice == "Özel" 
                    else random.randint(2, 5))
            iq_torch = gen.apply_memory_effects(iq_torch, memory_depth=depth)

        # SNR ayarla ve gürültü ekle
        snr_db = (float(snr_val) if snr_choice == "Özel" 
                 else random.uniform(*RF_LIMITS['snr_db']))
        iq_torch = gen.add_awgn_total(iq_torch, snr_db)

        # Rastgele ek bozulmalar
        if random.random() > 0.4:
            iq_torch = gen.apply_thermal_drift(iq_torch)
        if random.random() > 0.6:
            iq_torch = gen.apply_agc_response(iq_torch)

        # PSD hesaplama ve görselleştirme
        iq_np = iq_torch.detach().cpu().numpy()
        
        f_raw, p_raw = compute_welch_psd(
            iq_np, fs=FS, nperseg=1024, noverlap=512,
            window="hann", scaling="density", onesided=False
        )
        f_uni, p_uni = interp_to_uniform_grid(
            f_raw, p_raw, target_bins=1024, fs=FS
        )
        psd_final = log_and_norm(p_uni, eps=1e-12, mode='none')
        
        sample_data = {'iq': iq_np, 'fs': FS, 'signals': meta_list}
        
        # Özellik metni oluşturma
        true_features_text = _create_features_text(
            signal_type, meta_list, snr_db
        )
        
        main_fig, extra_fig = create_plots(sample_data, psd_final, f_uni)
        
        return main_fig, true_features_text, extra_fig
        
    except Exception as e:
        plt.close('all')
        fig = plt.figure()
        plt.text(0.5, 0.5, f"Hata: {e}", ha='center', va='center', wrap=True)
        return fig, str(e), None
    finally:
        if 'plt' in locals() or 'plt' in globals():
            plt.close('all')


def _create_features_text(signal_type, meta_list, snr_db):
    """
    Sinyal özelliklerini metin olarak formatlar.
    """
    signal_count = len([s for s in meta_list if 'mod' in s])
    true_features_text = (
        f"Gerçek Sinyal Bilgileri:\n- Tip: {signal_type}\n"
        f"- Sinyal Sayısı: {signal_count}\n"
    )
    
    for i, signal in enumerate(meta_list):
        if 'mod' not in signal:
            continue  # Sadece sinyal metalarını göster
            
        offset_key = ('f_center_est_hz' if signal_type == "Tek Sinyal" 
                     else 'f_off_hz')
        
        true_features_text += (
            f"\n-- Sinyal {i+1} --\n"
            f"- Mod: {signal.get('mod','N/A')}\n"
            f"- Ofset: {signal.get(offset_key,0)/1e3:.1f} kHz\n"
            f"- BW: {signal.get('bw_occ99_hz',0)/1e3:.1f} kHz\n"
        )
    
    true_features_text += f"\n-- Genel --\n- SNR: {snr_db:.1f} dB"
    
    return true_features_text


# ==============================================================================
# === BÖLÜM 3: GRADIO ARAYÜZ TANIMI =========================================
# ==============================================================================

def update_control_visibility(sig_type):
    """
    Sinyal tipi seçimine göre kontrol görünürlüğünü günceller.
    """
    return {
        'single_signal_controls': gr.update(visible=(sig_type == "Tek Sinyal")),
        'multi_signal_controls': gr.update(visible=(sig_type == "Çoklu Sinyal"))
    }


def update_mod3_visibility(num_choice):
    """
    3. modülasyon seçeneğinin görünürlüğünü günceller.
    """
    return gr.update(visible=(num_choice == "3 Sinyal"))


def update_advanced_visibility(choice):
    """
    Gelişmiş ayarların görünürlüğünü günceller.
    """
    return gr.update(visible=(choice == "Özel"))


if __name__ == "__main__":
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# UHF Sinyal Analiz ve Sınıflandırma Arayüzü")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Sinyal Üretim Ayarları")
                
                signal_type = gr.Radio(
                    ["Gürültü", "Tek Sinyal", "Çoklu Sinyal"], 
                    label="Sinyal Tipi", 
                    value="Tek Sinyal"
                )
                
                with gr.Group(visible=True) as single_signal_controls:
                    modulation_types = gr.CheckboxGroup(
                        MODS, 
                        label="Modülasyon Havuzu (Tek Sinyal için)", 
                        value=MODS
                    )
                
                with gr.Group(visible=False) as multi_signal_controls:
                    num_signals_choice = gr.Radio(
                        ["2 Sinyal", "3 Sinyal"], 
                        label="Karıştırılacak Sinyal Sayısı", 
                        value="2 Sinyal"
                    )
                    distance_choice = gr.Radio(
                        ["Yakın", "Uzak"], 
                        label="Sinyal Mesafesi", 
                        value="Yakın"
                    )
                    mod_choice_1 = gr.Dropdown(
                        MODS, label="Sinyal 1 Modülasyonu", value="QPSK"
                    )
                    mod_choice_2 = gr.Dropdown(
                        MODS, label="Sinyal 2 Modülasyonu", value="GFSK"
                    )
                    mod_choice_3 = gr.Dropdown(
                        MODS, label="Sinyal 3 Modülasyonu", 
                        value="FM", visible=False
                    )
                
                with gr.Accordion("Gelişmiş Bozulma Ayarları", open=False):
                    # Sadeleştirilmiş CFO
                    apply_cfo_and_doppler = gr.Checkbox(
                        label="CFO ve Doppler Etkisini Aktif Et (Otomatik)", 
                        value=False
                    )
                    
                    with gr.Group():
                        apply_clock_drift = gr.Checkbox(
                            label="Saat Sapması (Clock Drift) Aktif", 
                            value=False
                        )
                        drift_choice = gr.Radio(
                            ["Rastgele", "Özel"], 
                            label="Değer Seçimi", 
                            value="Rastgele"
                        )
                        drift_val = gr.Slider(
                            -20, 20, value=5, label="Özel Sapma (PPM)"
                        )
                    
                    with gr.Group():
                        apply_memory = gr.Checkbox(
                            label="Hafıza Etkileri (Memory Effects) Aktif", 
                            value=False
                        )
                        memory_choice = gr.Radio(
                            ["Rastgele", "Özel"], 
                            label="Etki Derinliği Seçimi", 
                            value="Rastgele"
                        )
                        memory_val = gr.Slider(
                            1, 8, value=2, step=1, 
                            label="Özel Etki Derinliği"
                        )
                    
                    with gr.Group():
                        snr_choice = gr.Radio(
                            ["Rastgele", "Özel"], 
                            label="Sinyal-Gürültü Oranı (SNR)", 
                            value="Rastgele"
                        )
                        snr_val = gr.Slider(
                            0, 30, value=20, label="Özel SNR (dB)"
                        )
                
                run_button = gr.Button("Sinyali Analiz Et", variant="primary")
                
                gr.Markdown("---")
                
                output_feats = gr.Textbox(
                    label="🔍 Sinyal Özellikleri (Gerçek)", 
                    lines=15, 
                    interactive=False, 
                    max_lines=20
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 2. Analiz Sonuçları")
                
                output_primary_plot = gr.Plot(label="Ana Analiz Grafikleri")
                
                with gr.Accordion("Ek Analiz Grafikleri", open=False):
                    output_extra_plot = gr.Plot()

        # --- Arayüz Etkileşimleri ---
        signal_type.change(
            fn=update_control_visibility, 
            inputs=signal_type, 
            outputs=[single_signal_controls, multi_signal_controls]
        )
        
        num_signals_choice.change(
            fn=update_mod3_visibility, 
            inputs=num_signals_choice, 
            outputs=mod_choice_3
        )
        
        drift_choice.change(
            fn=update_advanced_visibility, 
            inputs=drift_choice, 
            outputs=drift_val
        )
        
        memory_choice.change(
            fn=update_advanced_visibility, 
            inputs=memory_choice, 
            outputs=memory_val
        )
        
        snr_choice.change(
            fn=update_advanced_visibility, 
            inputs=snr_choice, 
            outputs=snr_val
        )
        
        all_inputs = [
            signal_type, modulation_types,
            num_signals_choice, distance_choice, 
            mod_choice_1, mod_choice_2, mod_choice_3,
            apply_cfo_and_doppler,
            apply_clock_drift, drift_choice, drift_val,
            apply_memory, memory_choice, memory_val,
            snr_choice, snr_val
        ]
        
        run_button.click(
            fn=run_analysis, 
            inputs=all_inputs, 
            outputs=[output_primary_plot, output_feats, output_extra_plot]
        )
    
    demo.launch()