import numpy as np


# сумма двух строк (векторов)
def sum_raw(raw1, raw2):
    ans = []
    for i in range(len(raw1)):
        ans.append(raw1[i] + raw2[i])
    return ans


# Умножение вектора на скаляр
def vect_mult_number(v, a):
    return [el * a for el in v]


# Прямое правило для приведения симплексной матрицы
def rect_rule(solve_element, solve_column_ind, solve_raw_ind, symplex_matrix):
    r = int(solve_raw_ind)
    s = int(solve_column_ind)
    for i in range(len(symplex_matrix)):  # Приведение всех строк матрицы
        if i != r:
            f = symplex_matrix[i][s] / solve_element  # Вычисляем фактор
            for j in range(len(symplex_matrix[i])):
                symplex_matrix[i][j] -= symplex_matrix[r][j] * f  # Корректируем строки
    return symplex_matrix


# Проверка, является ли симплексная таблица оптимальной
def is_optimal(symplex_matrix, factor):
    last_row = symplex_matrix[-1, :-1]  # Последняя строка, кроме последнего элемента
    if factor == 'min':
        return np.all(last_row <= 0)  # Для минимизации все значения должны быть <= 0
    else:
        return np.all(last_row >= -1e-6)  # Для максимизации допускаем небольшое отклонение от 0



# Определение базиса в симплексной таблице
def find_basis(symplex_matrix):
    num_rows, num_cols = symplex_matrix.shape
    basis_indices = []
    for col_index in range(num_cols - 1):
        col = symplex_matrix[:, col_index]
        # Проверяем, является ли столбец базисным (единичная матрица)
        if np.sum(col == 1) == 1 and np.sum(col == 0) == num_rows - 1:
            basis_indices.append(col_index)
    return basis_indices


# Симплекс-метод для нахождения оптимального решения
def symplex_method(symplex_matrix, var_names, row_base_vars, factor, show_solving):
    f = show_solving
    if factor == 'max':
        print('\n', '\n', '-' * 10, 'МАКСИМИЗАЦИЯ', '\n', '-' * 10, '\n')
    elif factor == 'min':
        print('\n', '\n', '-' * 20, 'МИНИМИЗАЦИЯ', '-' * 20, '\n')
    else:
        print('ОШИБКА', '\n')
        return 0

    iteration = 0
    max_iterations = 1000

    while not is_optimal(symplex_matrix, factor):
        if f:
            print('z-строка:\n', symplex_matrix[-1, :-1])
        if iteration > max_iterations:
            print("Превышено максимальное количество итераций. Возможное зацикливание.")
            break

        z_vect = symplex_matrix[-1, :-1]

        # Выбор разрешающего столбца
        if factor == 'min':
            solve_column_ind = np.argmax(z_vect)
        else:
            solve_column_ind = np.argmax(z_vect)

        if f:
            print('Решающий столбец: ', solve_column_ind)

        min_ratio = float('inf')
        solve_raw_ind = -1
        for i in range(len(symplex_matrix) - 1):
            if symplex_matrix[i][solve_column_ind] > 0:
                ratio = symplex_matrix[i][-1] / symplex_matrix[i][solve_column_ind]
                if abs(ratio) < min_ratio:
                    min_ratio = abs(ratio)
                    solve_raw_ind = i

        if solve_raw_ind == -1:
            print('Задача не имеет ограниченного решения.')
            return

        if f:
            print('Решающая строка', solve_raw_ind)

        solve_element = symplex_matrix[solve_raw_ind][solve_column_ind]

        if f:
            print('Решающий элемент: ', solve_element)

        row_base_vars[solve_raw_ind] = var_names[solve_column_ind]


        symplex_matrix = rect_rule(solve_element, solve_column_ind, solve_raw_ind, symplex_matrix)
        symplex_matrix[solve_raw_ind] /= solve_element
        if f:
            print("Текущая симплекс-матрица:\n", symplex_matrix)
            print("Текущий базис:\n", row_base_vars)

    # Вывод значений переменных и целевой функции
    solution = symplex_matrix[:, -1]
    z_value = symplex_matrix[-1, -1]
    if f:
        print('\n', '-' * 20, 'РЕЗУЛЬТАТ', '-' * 20)
        print('\n', "\nТекущий базис:", row_base_vars)
        print("Оптимальное решение:", solution[:-1])
        if factor=='min':
            print("Значение целевой функции (z):", z_value)
        else:
           print("Значение целевой функции (z):", -z_value)

    res = []
    for i in range(len(row_base_vars)):
        if row_base_vars[i][0] == 'x':
            res.append([row_base_vars[i][1:], float(solution[i])])
    return res



# M-метод для решения задачи с искусственными переменными
def M_method(z, x, b, signs, M, factor, show_solving):
    f = show_solving
    limit_num = len(x)

    A = np.array(x)
    B = np.array(b)

    # Обработка знаков неравенств и создание дополнительных столбцов
    additional_cols = []
    for i in range(limit_num):
        extra_col = np.zeros(limit_num)
        if signs[i] == '<=':
            extra_col[i] = 1
            additional_cols.append(extra_col)
        elif signs[i] == '>=':
            extra_col[i] = -1
            additional_cols.append(extra_col)

    # Добавление дополнительных столбцов S в матрицу A
    if additional_cols:
        A = np.hstack((A, np.array(additional_cols).T))

    # Создание искусственной единичной матрицы
    artificial_matrix = np.eye(limit_num)

    symplex_matrix = np.hstack((A, artificial_matrix.T, B.reshape(-1, 1)))

    # Преобразование вектора коэффициентов целевой функции
    z_vect = np.hstack(
        (vect_mult_number(z, -1), [0] * len(additional_cols), np.ones(artificial_matrix.shape[0]) * (-M), [0]))

    symplex_matrix = np.vstack((symplex_matrix, z_vect))

    if f:
        print("Начальная симплекс-матрица:")
        print(symplex_matrix)

    for i in range(limit_num):
        symplex_matrix[limit_num] = sum_raw(symplex_matrix[limit_num], symplex_matrix[i] * (M))
    if f:
        print(symplex_matrix)

    var_names = [f"x{i + 1}" for i in range(len(z))] + [f"s{i + 1}" for i in
                                                        range(len(additional_cols))] + [f"r{i + 1}" for i in range(
        artificial_matrix.shape[0])] + ["b"]
    row_base_vars = []
    indexes = find_basis(symplex_matrix)
    for i in range(len(indexes)):
        row_base_vars.append(var_names[indexes[i]])

    # Выполнение симплекс-метода
    return symplex_method(symplex_matrix, var_names, row_base_vars, factor, show_solving)


# Вспомогательная функция для отображения результатов
def show_res(res, data, proteins, fats, carbs, prices, calories):
    sum_prot = 0
    sum_fat = 0
    sum_carb = 0
    sum_price = 0
    sum_calories = 0
    print('\n', '-' * 20, 'РЕЗУЛЬТАТ', '-' * 20)
    print("\nРекомендуемый набор блюд:\n")
    for i in range(len(res)):
        number = int(res[i][0]) - 1
        colvo = res[i][1]
        print(str(i + 1) + '.', data[number][0], '-', round(colvo, 2), 'шт.')
        sum_prot += proteins[number] * colvo
        sum_fat += fats[number] * colvo
        sum_carb += carbs[number] * colvo
        sum_price += prices[number] * colvo
        sum_calories += calories[number] * colvo

    print('\nCтоимость набора:', round(sum_price, 2), 'руб.')
    print('Белки:', round(sum_prot, 2), 'г.')
    print('Жиры:', round(sum_fat, 2), 'г.')
    print('Углеводы:', round(sum_carb, 2), 'г.')
    print('Калории:', round(sum_calories, 2))


## Чтение данных из текстового файла
file_path = 'menu.txt'
with open(file_path, encoding='utf-8') as file:
    lines = file.readlines()

# Пропускаем заголовок
data = [line.strip().split(',') for line in lines[1:]]

# Извлечение данных и преобразование в числа
names = [item[0] for item in data]
calories = np.array([int(item[1]) for item in data])
proteins = np.array([int(item[2]) for item in data])
fats = np.array([int(item[3]) for item in data])
carbs = np.array([int(item[4]) for item in data])
prices = np.array([int(item[5]) for item in data])

x, b, signs = [], [], []

# проверка ввода y/n
def get_valid_input(prompt, valid_options):
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in valid_options:
            return user_input
        else:
            print(f"Пожалуйста, введите {' или '.join(valid_options)}.")

# проверка ввода чисел для КБЖУ
def get_valid_number(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Пожалуйста, введите числовое значение.")

# считываем ограничения от пользователя
use_calories = get_valid_input("Использовать ограничение по калориям? (y/n): ", ['y', 'n']) == 'y'
use_proteins = get_valid_input("Использовать ограничение по белкам? (y/n): ", ['y', 'n']) == 'y'
use_fats = get_valid_input("Использовать ограничение по жирам? (y/n): ", ['y', 'n']) == 'y'
use_carbs = get_valid_input("Использовать ограничение по углеводам? (y/n): ", ['y', 'n']) == 'y'

# заносим ограничения в списки
if use_calories:
    D = get_valid_number("Введите количество калорий: ")
    deltaD = D * 0.1
    x.append(calories)
    x.append(calories)
    b.append(D - deltaD)
    b.append(D + deltaD)
    signs.append('>=')
    signs.append('<=')

if use_proteins:
    A = get_valid_number("Введите количество белков: ")
    deltaA = A * 0.1
    x.append(proteins)
    x.append(proteins)
    b.append(A - deltaA)
    b.append(A + deltaA)
    signs.append('>=')
    signs.append('<=')

if use_fats:
    B = get_valid_number("Введите количество жиров: ")
    deltaB = B * 0.1
    x.append(fats)
    x.append(fats)
    b.append(B - deltaB)
    b.append(B + deltaB)
    signs.append('>=')
    signs.append('<=')

if use_carbs:
    C = get_valid_number("Введите количество углеводов: ")
    deltaC = C * 0.1
    x.append(carbs)
    x.append(carbs)
    b.append(C - deltaC)
    b.append(C + deltaC)
    signs.append('>=')
    signs.append('<=')

# целевая функция: минимизация или максимизация стоимости
z = prices
M = 1000

# поиск минимальной стоимости
print("\nМинимальная стоимость:")
res=M_method(z,x,b,signs,M,'min',False)
show_res(res,data,proteins,fats,carbs,prices,calories)

# поиск максимальной стоимости
print("\nМаксимальная стоимость:")
res=M_method(z,x,b,signs,M,'max',False)
show_res(res,data,proteins,fats,carbs,prices,calories)

