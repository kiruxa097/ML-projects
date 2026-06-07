import matplotlib.pyplot as plt

# Создаем холст
fig, ax = plt.subplots(figsize=(10,5))

# Настраиваем оси
ax.set_title("Мой прогресс в ML")
ax.set_xlabel("День обучения")
ax.set_ylabel("Количетсво выученных функций")

# Рисуем сетку
ax.grid(True)

plt.show()