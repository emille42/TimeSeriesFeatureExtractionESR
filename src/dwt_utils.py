import pywt
import matplotlib.pyplot as plt


# Декомпозиция сигнала до заданного уровня
def transform_signal_with_dwt(signal, wavelet='haar', level=1):
    transformed_signal = pywt.wavedec(signal, wavelet=wavelet, level=level)
    # Результат преобразования возвращает сам преобразованный сигнал и коэффициенты, необходимые для его восстановления
    # Преобразованный сигнал хранится в первом элементе массива
    transformed_signal = transformed_signal[0]
    return transformed_signal

# Декомпозиция датасета до заданного уровня
def transfrom_dataset_with_dwt(data, wavelet='haar', level=1):
  transformed_df = data.copy()
  keys = transformed_df.columns
  for row in transformed_df.iterrows():
    signal = row[1].to_numpy()
    index = row[0]
    transformed_signal = transform_signal_with_dwt(signal=signal, wavelet=wavelet, level=level)
    values = {k : v for k,v in zip(keys, transformed_signal)}
    transformed_df.loc[index] = values
  transformed_df = transformed_df.dropna(axis=1)
  return transformed_df


def decompose_signal_plot(signal, level=5):
   fig, ax = plt.subplots(figsize=(6,1))
   ax.set_title("Исходный сигнал: ")
   ax.plot(signal)
   plt.show()
    
   data = signal
   waveletname = 'haar'
 
   fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(6,6))
   for i in range(level):
       (data, coeff_d) = pywt.dwt(data, waveletname)
       axes[i, 0].plot(data, 'r')
       axes[i, 1].plot(coeff_d, 'g')
       axes[i, 0].set_ylabel("Уровень {}".format(i + 1), fontsize=12, rotation=90)
       axes[i, 0].set_yticklabels([])
       if i == 0:
           axes[i, 0].set_title("Approximation coefficients", fontsize=14)
           axes[i, 1].set_title("Detail coefficients", fontsize=14)
       axes[i, 1].set_yticklabels([])
   plt.tight_layout()
   plt.show()