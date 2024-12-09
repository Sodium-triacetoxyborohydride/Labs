import numpy as np
import matplotlib.pyplot as plt
import random


# Инициализация сетки
def initialize_grid(size):
    return np.zeros((size, size))


# Генерация матрицы вероятностей переходов
def generate_transition_matrix(size):
    matrix = np.random.rand(size, size, 4)  # 4 направления: вверх, вниз, влево, вправо
    for i in range(size):
        for j in range(size):
            matrix[i, j] /= np.sum(matrix[i, j])  # Нормировка вероятностей
            if random.random() > 0.7:  # с вероятностью 0.3 оставляем 0
                zero_index = random.randint(0, 3)
                matrix[i, j][zero_index] = 0
                matrix[i, j] /= np.sum(matrix[i, j])  # повторная нормировка
    return matrix


# Выбор случайной позиции, исключая указанную
def select_random_position(size, exclude=None):
    while True:
        pos = (random.randint(0, size - 1), random.randint(0, size - 1))
        if pos != exclude:
            return pos


# Реализация случайного блуждания
def perform_random_walk(size, transition_matrix, start_pos, end_pos):
    steps = 0
    current_pos = start_pos
    path = [current_pos]

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # направления: вверх, вниз, влево, вправо

    while current_pos != end_pos:
        i, j = current_pos
        move = np.random.choice(4, p=transition_matrix[i, j])  # выбор направления на основе вероятностей
        ni, nj = i + directions[move][0], j + directions[move][1]

        # проверка выхода за границы
        if 0 <= ni < size and 0 <= nj < size:
            current_pos = (ni, nj)
            path.append(current_pos)
            steps += 1

    return steps, path


# Проведение эксперимента
def conduct_experiment(size, num_animals):
    transition_matrix = generate_transition_matrix(size)

    end_pos = select_random_position(size)
    steps_list = []

    for _ in range(num_animals):
        start_pos = select_random_position(size, exclude=end_pos)
        steps, path = perform_random_walk(size, transition_matrix, start_pos, end_pos)
        steps_list.append(steps)

    plt.hist(steps_list, bins=30, edgecolor='black')
    plt.title('Histogram of Steps to Reach Sensor')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    plt.show()

    # Визуализация пути
    if size <= 3:
        plt.figure()
        for i in range(len(path) - 1):
            plt.plot([path[i][1], path[i + 1][1]], [path[i][0], path[i + 1][0]], 'bo-')
        plt.plot(end_pos[1], end_pos[0], 'ro', label='Sensor')
        plt.plot(start_pos[1], start_pos[0], 'go', label='Start')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title('Path of an Animal')
        plt.show()


# Запуск эксперимента с заданными параметрами
conduct_experiment(size=10, num_animals=100)
