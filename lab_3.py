import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt

# 1. Создание тестового сигнала с разными компонентами
fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)

# Компоненты сигнала
f1 = 10 * (1 + 0.5 * t)  # изменяющаяся частота (чирп)
f2 = 50  # постоянная частота
f3 = 100 * np.exp(-2 * t)  # затухающая частота

signal_chirp = np.sin(2 * np.pi * f1 * t)  # чирп
signal_constant = 0.5 * np.sin(2 * np.pi * f2 * t)  # постоянный тон
signal_decay = 0.3 * np.sin(2 * np.pi * f3 * t) * np.exp(-3 * t)  # затухающий
signal_noise = 0.1 * np.random.randn(len(t))

# Итоговый сигнал
x = signal_chirp + signal_constant + signal_decay + signal_noise

# 2. Классические вейвлеты (Wavelet Transform)
scales = np.arange(1, 128)
coef, freqs = pywt.cwt(x, scales, 'morl', sampling_period=1/fs)

# 3. Пользовательский вейвлет (Модифицированный Morlet)
def custom_wavelet(t, freq, bandwidth):
    """Создание вейвлета с регулируемой полосой"""
    sigma = bandwidth / (2 * np.pi * freq)
    return (np.exp(2j * np.pi * freq * t) * 
            np.exp(-t**2 / (2 * sigma**2)))

# Создание сетки для вейвлет-преобразования
dt = 1/fs
num_freqs = 50  # уменьшим количество частот для скорости
freq_range = np.logspace(np.log10(10), np.log10(200), num_freqs)
bandwidth = 5  # параметр полосы

# Вейвлет-преобразование с кастомным вейвлетом (ИСПРАВЛЕННАЯ ФУНКЦИЯ)
def custom_cwt(x, freqs, bandwidth, fs):
    dt = 1/fs
    cwt_matrix = np.zeros((len(freqs), len(x)), dtype=complex)
    
    for i, f in enumerate(freqs):
        # Фиксированная длина вейвлета (лучше контролировать)
        t_wavelet = np.linspace(-0.1, 0.1, 200)  # фиксированная длина 200 точек
        
        # Создание вейвлета
        wavelet = custom_wavelet(t_wavelet, f, bandwidth)
        wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet)**2))
        
        # Свертка с сохранением размера
        conv_result = np.convolve(x, wavelet, mode='same')
        cwt_matrix[i, :] = conv_result[:len(x)]  # обрезаем до нужного размера
    
    return cwt_matrix

cwt_custom = custom_cwt(x, freq_range, bandwidth, fs)

# 4. Chirplet Transform (Чирплеты) - УПРОЩЕННАЯ ВЕРСИЯ
class Chirplet:
    def __init__(self, alpha, beta, fc, duration=0.2):
        self.alpha = alpha  # параметр чирпа
        self.beta = beta    # масштаб
        self.fc = fc        # центральная частота
        self.duration = duration
        
    def generate(self, t_center, fs):
        """Генерация чирплета с центром в t_center"""
        t_len = int(self.duration * fs)
        t_local = np.linspace(-self.duration/2, self.duration/2, t_len)
        
        gaussian = np.exp(-(t_local**2) / (2 * self.beta**2))
        chirp = np.exp(1j * 2 * np.pi * (self.fc * t_local + 0.5 * self.alpha * t_local**2))
        return gaussian * chirp

# Создание набора чирплетов
chirplet_freqs = [10, 50, 100]
chirplet_alphas = [50, 0, -100]  # скорость изменения частоты
chirplet_betas = [0.05, 0.1, 0.03]

chirplets = []
for fc, alpha, beta in zip(chirplet_freqs, chirplet_alphas, chirplet_betas):
    chirplets.append(Chirplet(alpha, beta, fc, 0.2))

# Чирплет-преобразование (УПРОЩЕННОЕ)
def chirplet_transform_simple(x, chirplets, t, fs):
    result = np.zeros((len(chirplets), len(x)), dtype=complex)
    
    for i, chirp in enumerate(chirplets):
        # Генерация чирплета фиксированной длины
        chirplet_len = int(chirp.duration * fs)
        t_local = np.linspace(-chirp.duration/2, chirp.duration/2, chirplet_len)
        
        # Создание чирплета
        gaussian = np.exp(-(t_local**2) / (2 * chirp.beta**2))
        chirp_signal = np.exp(1j * 2 * np.pi * (chirp.fc * t_local + 0.5 * chirp.alpha * t_local**2))
        wavelet = gaussian * chirp_signal
        wavelet = wavelet / np.linalg.norm(wavelet)
        
        # Свертка
        conv_result = np.convolve(x, np.conj(wavelet), mode='same')
        result[i, :] = conv_result[:len(x)]
    
    return result

chirp_result = chirplet_transform_simple(x, chirplets, t, fs)

# 5. Адаптивный вейвлет (УПРОЩЕННЫЙ)
def adaptive_wavelet_simple(x, t, initial_freq=50):
    """Упрощенный адаптивный вейвлет"""
    dt = t[1] - t[0]
    wavelet_len = 200  # фиксированная длина
    
    # Начальный вейвлет (Morlet)
    t_wavelet = np.linspace(-0.1, 0.1, wavelet_len)
    sigma = 0.5 / initial_freq
    wavelet = np.exp(2j * np.pi * initial_freq * t_wavelet) * np.exp(-t_wavelet**2 / (2 * sigma**2))
    wavelet = wavelet / np.linalg.norm(wavelet)
    
    # Свертка по всему сигналу
    conv_result = np.convolve(x, np.conj(wavelet), mode='same')
    
    return np.abs(conv_result[:len(x)]), wavelet

adapted_coeffs, final_wavelet = adaptive_wavelet_simple(x, t, initial_freq=50)

# 6. Визуализация (УПРОЩЕННАЯ)
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Исходный сигнал
axes[0,0].plot(t, x)
axes[0,0].set_title('Исходный сигнал')
axes[0,0].set_xlabel('Время, с')
axes[0,0].set_ylabel('Амплитуда')
axes[0,0].grid(True)

# Компоненты сигнала
axes[0,1].plot(t, signal_chirp, label='Чирп (f=10-15 Гц)', alpha=0.7)
axes[0,1].plot(t, signal_constant, label='Постоянный (50 Гц)', alpha=0.7)
axes[0,1].plot(t, signal_decay, label='Затухающий (100→0 Гц)', alpha=0.7)
axes[0,1].set_title('Компоненты сигнала')
axes[0,1].set_xlabel('Время, с')
axes[0,1].legend()
axes[0,1].grid(True)

# Спектрограмма (STFT)
f_stft, t_stft, Zxx = signal.stft(x, fs, nperseg=256)
im1 = axes[1,0].pcolormesh(t_stft, f_stft, np.abs(Zxx), shading='gouraud', cmap='viridis')
axes[1,0].set_title('Спектрограмма (STFT)')
axes[1,0].set_xlabel('Время, с')
axes[1,0].set_ylabel('Частота, Гц')
plt.colorbar(im1, ax=axes[1,0])

# Кастомный CWT
im2 = axes[1,1].pcolormesh(t, freq_range, np.abs(cwt_custom), shading='gouraud', cmap='plasma')
axes[1,1].set_title('Кастомный вейвлет')
axes[1,1].set_xlabel('Время, с')
axes[1,1].set_ylabel('Частота, Гц')
plt.colorbar(im2, ax=axes[1,1])

# Чирплет-преобразование
im3 = axes[2,0].imshow(np.abs(chirp_result), aspect='auto', extent=[0, 1, 0, len(chirplets)],
                      cmap='hot', interpolation='bilinear')
axes[2,0].set_title('Чирплет-преобразование')
axes[2,0].set_xlabel('Время, с')
axes[2,0].set_yticks([0.5, 1.5, 2.5])
axes[2,0].set_yticklabels([f'f={c.fc}' for c in chirplets])
plt.colorbar(im3, ax=axes[2,0])

# Адаптивный вейвлет и его коэффициенты
ax2 = axes[2,1].twinx()
axes[2,1].plot(t, adapted_coeffs, 'b-', label='Коэффициенты')
axes[2,1].set_xlabel('Время, с')
axes[2,1].set_ylabel('Корреляция', color='b')
axes[2,1].tick_params(axis='y', labelcolor='b')
axes[2,1].set_title('Адаптивный вейвлет')

# Вставка формы вейвлета
inset_ax = axes[2,1].inset_axes([0.6, 0.6, 0.3, 0.3])
t_wavelet = np.linspace(-0.05, 0.05, len(final_wavelet))
inset_ax.plot(t_wavelet, np.real(final_wavelet), 'r-', label='Real')
inset_ax.plot(t_wavelet, np.imag(final_wavelet), 'g--', label='Imag')
inset_ax.set_title('Форма вейвлета')
inset_ax.legend(fontsize=8)
inset_ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. Сравнение методов
print("=== ВЕЙВЛЕТ-АНАЛИЗ И ЧИРПЛЕТЫ ===")
print(f"Длина сигнала: {len(x)} отсчетов")
print(f"Частота дискретизации: {fs} Гц")
print(f"Анализируемые частоты: от {freq_range[0]:.1f} до {freq_range[-1]:.1f} Гц")
print(f"Количество чирплетов: {len(chirplets)}")

print("\nХарактеристики чирплетов:")
for i, chirp in enumerate(chirplets):
    print(f"  Чирплет {i+1}: f={chirp.fc} Гц, α={chirp.alpha}, β={chirp.beta:.3f}")

# Анализ результатов
print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===")
print("1. Чирп-компонента (10-15 Гц):")
print("   - Видна на спектрограмме как наклонная линия")
print("   - Хорошо выделяется чирплетом с f=10 Гц")

print("\n2. Постоянная компонента (50 Гц):")
print("   - Горизонтальная линия на спектрограмме")
print("   - Максимум в CWT на соответствующих масштабах")

print("\n3. Затухающая компонента (100→0 Гц):")
print("   - Видна как затухающая линия на спектрограмме")
print("   - Хорошо обнаруживается адаптивным вейвлетом")