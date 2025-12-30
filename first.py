import numpy as np
import matplotlib.pyplot as plt

# Фиксация генератора случайных чисел для воспроизводимости
np.random.seed(42)

# Число эпох
epochs = np.arange(1, 51)

# Синтетическая функция потерь
train_loss = np.exp(-epochs / 15) + 0.04 * np.random.rand(len(epochs))
val_loss = np.exp(-epochs / 14) + 0.05 * np.random.rand(len(epochs))

# Построение графика
plt.figure(figsize=(6, 4))
plt.plot(epochs, train_loss, label="Обучающая выборка")
plt.plot(epochs, val_loss, label="Валидационная выборка")

plt.xlabel("Число эпох")
plt.ylabel("Значение функции потерь")
plt.title("Зависимость функции потерь от числа эпох")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
