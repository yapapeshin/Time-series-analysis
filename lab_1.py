import numpy as np
import matplotlib.pyplot as plt

# 1. Создание тестового сигнала
Fs = 1000  # Частота дискретизации
t = np.linspace(0, 1, Fs, endpoint=False)  # Время 1 секунда

# Сигнал: сумма 3 синусоид + шум
f1, A1 = 50, 2.0    # 50 Гц, амплитуда 2
f2, A2 = 120, 1.5   # 120 Гц, амплитуда 1.5
f3, A3 = 300, 0.8   # 300 Гц, амплитуда 0.8

signal = (A1 * np.sin(2 * np.pi * f1 * t) + 
          A2 * np.sin(2 * np.pi * f2 * t) + 
          A3 * np.sin(2 * np.pi * f3 * t) +
          0.5 * np.random.randn(len(t)))

# 2. Прямое преобразование Фурье (ТУДА)
n = len(signal)
fft_result = np.fft.fft(signal)
freqs = np.fft.fftfreq(n, 1/Fs)

# Односторонний спектр
magnitude = 2.0/n * np.abs(fft_result[:n//2])
freqs_one_sided = freqs[:n//2]

# 3. Анализ частотных параметров
peaks = []
peak_threshold = 0.1
for i in range(1, len(magnitude)-1):
    if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
        if magnitude[i] > peak_threshold:
            peaks.append(i)

print("Найденные частотные компоненты:")
for idx in peaks[:5]:  # Показываем топ-5
    print(f"Частота: {freqs_one_sided[idx]:.1f} Гц, Амплитуда: {magnitude[idx]:.2f}")

# 4. Обратное преобразование Фурье
# Фильтрация оставляем только значимые частоты
filtered_fft = np.zeros_like(fft_result)
for idx in peaks:
    filtered_fft[idx] = fft_result[idx]
    filtered_fft[-idx] = fft_result[-idx]  # Симметричная часть

reconstructed = np.fft.ifft(filtered_fft).real

# 5. Визуализация
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Исходный сигнал
axes[0].plot(t[:200], signal[:200], 'b', alpha=0.7, label='Исходный')
axes[0].set_title('Исходный сигнал (фрагмент)')
axes[0].set_xlabel('Время (с)')
axes[0].set_ylabel('Амплитуда')
axes[0].grid(True)
axes[0].legend()

# Амплитудный спектр
axes[1].plot(freqs_one_sided, magnitude, 'r')
axes[1].set_title('Амплитудный спектр')
axes[1].set_xlabel('Частота (Гц)')
axes[1].set_ylabel('Амплитуда')
axes[1].set_xlim(0, Fs/2)
axes[1].grid(True)

# Восстановленный сигнал
axes[2].plot(t[:200], reconstructed[:200], 'g', alpha=0.7, label='Восстановленный')
axes[2].set_title('Восстановленный сигнал (фрагмент)')
axes[2].set_xlabel('Время (с)')
axes[2].set_ylabel('Амплитуда')
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()

# 6. Основные параметры
print("\n=== ОСНОВНЫЕ ПАРАМЕТРЫ ===")
print(f"Частота дискретизации: {Fs} Гц")
print(f"Длина сигнала: {n} отсчетов")
print(f"Длительность: {t[-1]:.3f} с")
print(f"Разрешение по частоте: {Fs/n:.2f} Гц")
print(f"Максимальная частота (Найквиста): {Fs/2} Гц")