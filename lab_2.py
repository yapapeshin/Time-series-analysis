import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Параметры
Fs = 1000  # Частота дискретизации
t_total = 0.1  # Общее время
t = np.linspace(0, t_total, int(Fs * t_total), endpoint=False)

# Исходный аналоговый сигнал
f_signal = 50  
A = 1.0
original_signal = A * np.sin(2 * np.pi * f_signal * t)

# 1. Дискретизация по Котельникову (Fs > 2*f_max)
Fs_nyquist = 2.2 * f_signal  # чуть выше Найквиста
Fs_under = 1.8 * f_signal    # ниже Найквиста 

# Выборка точек для интерполяции
samples_nyquist = int(t_total * Fs_nyquist)
samples_under = int(t_total * Fs_under)

t_nyquist = np.linspace(0, t_total, samples_nyquist)
t_under = np.linspace(0, t_total, samples_under)

signal_nyquist = A * np.sin(2 * np.pi * f_signal * t_nyquist)
signal_under = A * np.sin(2 * np.pi * f_signal * t_under)

# 2. Различные методы интерполяции
methods = {
    'sinc (идеальная)': None,
    'кубическая': 'cubic',
    'линейная': 'linear',
    'ближайший': 'nearest'
}

# 3. теорема Котельникова
def sinc_reconstruction(t_samples, y_samples, t_output):
    reconstructed = np.zeros_like(t_output)
    T = 1/Fs_nyquist  # период дискретизации
    
    for i, y in enumerate(y_samples):
        reconstructed += y * np.sinc((t_output - t_samples[i]) / T)
    return reconstructed

# Интерполяция на плотную сетку
t_dense = np.linspace(0, t_total, 1000)

# Вычисления
reconstructed_sinc = sinc_reconstruction(t_nyquist, signal_nyquist, t_dense)
reconstructed_cubic = interp1d(t_nyquist, signal_nyquist, kind='cubic', bounds_error=False, fill_value='extrapolate')(t_dense)
reconstructed_linear = interp1d(t_nyquist, signal_nyquist, kind='linear', bounds_error=False, fill_value='extrapolate')(t_dense)

# 4. Анализ ошибок
original_dense = A * np.sin(2 * np.pi * f_signal * t_dense)

error_sinc = np.mean(np.abs(reconstructed_sinc - original_dense))
error_cubic = np.mean(np.abs(reconstructed_cubic - original_dense))
error_linear = np.mean(np.abs(reconstructed_linear - original_dense))

print("=== СРАВНЕНИЕ МЕТОДОВ ИНТЕРПОЛЯЦИИ ===")
print(f"Частота сигнала: {f_signal} Гц")
print(f"Частота Найквиста: {2*f_signal} Гц")
print(f"Частота дискретизации: {Fs_nyquist:.1f} Гц (выше Найквиста в {Fs_nyquist/(2*f_signal):.2f} раз)")
print(f"\nСредняя ошибка восстановления:")
print(f"Sinc-интерполяция: {error_sinc:.6f}")
print(f"Кубическая сплайн: {error_cubic:.6f}")
print(f"Линейная: {error_linear:.6f}")

# 5. Демонстрация алиасинга
original_under = A * np.sin(2 * np.pi * f_signal * t_dense)
reconstructed_under = interp1d(t_under, signal_under, kind='cubic', bounds_error=False, fill_value='extrapolate')(t_dense)
aliasing_freq = abs(Fs_under - f_signal)  # возникает ложная частота

# 6. Визуализация
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# График 1: Исходный vs дискретизированный (выше Найквиста)
axes[0,0].plot(t_dense, original_dense, 'b-', alpha=0.5, label='Исходный')
axes[0,0].plot(t_nyquist, signal_nyquist, 'ro', markersize=8, label=f'Отсчеты (Fs={Fs_nyquist:.0f} Гц)')
axes[0,0].set_title(f'Дискретизация ВЫШЕ частоты Найквиста\n(Fs = {Fs_nyquist:.0f} Гц > {2*f_signal} Гц)')
axes[0,0].set_xlabel('Время, с')
axes[0,0].set_ylabel('Амплитуда')
axes[0,0].grid(True)
axes[0,0].legend()

# График 2: Разные методы интерполяции
axes[0,1].plot(t_dense, original_dense, 'k--', alpha=0.3, label='Исходный')
axes[0,1].plot(t_dense, reconstructed_sinc, 'r-', label='Sinc (идеальная)')
axes[0,1].plot(t_dense, reconstructed_cubic, 'g-', label='Кубическая')
axes[0,1].plot(t_dense, reconstructed_linear, 'b-', label='Линейная')
axes[0,1].set_title('Сравнение методов интерполяции')
axes[0,1].set_xlabel('Время, с')
axes[0,1].grid(True)
axes[0,1].legend()

# График 3: Дискретизация НИЖЕ Найквиста (алиасинг)
axes[1,0].plot(t_dense, original_dense, 'b-', alpha=0.5, label=f'Исходный ({f_signal} Гц)')
axes[1,0].plot(t_under, signal_under, 'ro', markersize=8, label=f'Отсчеты (Fs={Fs_under:.0f} Гц)')
axes[1,0].plot(t_dense, reconstructed_under, 'r--', label=f'Восстановленный (~{aliasing_freq:.0f} Гц)')
axes[1,0].set_title(f'АЛИАСИНГ: Fs НИЖЕ Найквиста\n(Fs = {Fs_under:.0f} Гц < {2*f_signal} Гц)')
axes[1,0].set_xlabel('Время, с')
axes[1,0].set_ylabel('Амплитуда')
axes[1,0].grid(True)
axes[1,0].legend()

# График 4: Sinc-функции
axes[1,1].clear()
T = 1/Fs_nyquist
for i in range(3):
    t_sinc = np.linspace(t_nyquist[i] - 3*T, t_nyquist[i] + 3*T, 200)
    sinc_i = signal_nyquist[i] * np.sinc((t_sinc - t_nyquist[i]) / T)
    axes[1,1].plot(t_sinc, sinc_i, '--', alpha=0.5)
    
axes[1,1].plot(t_nyquist, signal_nyquist, 'ro', markersize=8, label='Отсчеты')
axes[1,1].plot(t_dense, reconstructed_sinc, 'r-', linewidth=2, label='Сумма sinc-функций')
axes[1,1].set_title('Принцип sinc-интерполяции\n(сумма сдвинутых sinc-функций)')
axes[1,1].set_xlabel('Время, с')
axes[1,1].grid(True)
axes[1,1].legend()

plt.tight_layout()
plt.show()

# 7. Проверка теоремы Котельникова
print("\n=== ТЕОРЕМА КОТЕЛЬНИКОВА ===")
print("1. Сигнал должен быть с ограниченным спектром (f_max)")
print(f"2. Частота дискретизации Fs > 2*f_max = {2*f_signal} Гц")
print(f"3. Восстановление: x(t) = Σ x[n] * sinc(π(t-nT)/T)")
print(f"4. T = 1/Fs = {1/Fs_nyquist:.6f} с - период дискретизации")
print(f"\nПроверка: Fs={Fs_nyquist:.1f} Гц > {2*f_signal} Гц - {'✓ условие выполнено' if Fs_nyquist > 2*f_signal else '✗ условие НЕ выполнено'}")