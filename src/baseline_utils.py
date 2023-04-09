
from sklearn.metrics import classification_report
from scipy.fftpack import fft
import numpy as np

# Функция для трансформирования сигнала при помощи FFT
def transform_signal_with_fft(signal, N):
  # Используем scale_factor для масштабирования сигнала
  # Умножаем на 2 т.к. далее будем брать только половину значений FFT
  scale_factor = 2.0/N
  fft_output = fft(signal)
  # Поскольку результат преобразования - комплексное число, используем np.abs, чтобы вычислить его модуль
  # Поскольку результат FFT при вещественных значениях входного сигнала зеркальный, возьмем только его половину
  fft_output = scale_factor*np.abs(fft_output[0:N//2])
  return fft_output

# Функция для трансформирования датасета при помощи FFT
def transfrom_dataset_with_fft(data, N):
  transformed_df = data.copy()
  keys = transformed_df.columns[0:N//2]
  for row in transformed_df.iterrows():
    signal = row[1].to_numpy()
    index = row[0]
    transformed_signal = transform_signal_with_fft(signal, N)
    values = {k : v for k,v in zip(keys, transformed_signal)}
    transformed_df.loc[index] = values
  transformed_df = transformed_df.dropna(axis=1)
  return transformed_df


# Функия для тренировки модели и оценки качества классификации
def train_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    report = classification_report(Y_test, pred)
    print(report)