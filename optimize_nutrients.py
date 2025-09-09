import csv
import numpy as np
from scipy.optimize import nnls

# Key name for complex digestible nutrients containing newline in CSV header
COMPLEX_KEY = 'Сложные \nперевариваемые'

NUTRIENT_KEYS = [
    'Белки',
    'Насыщенные',
    'НЕнасыщенные',
    'Простые',
    COMPLEX_KEY,
    'Растворимая',
    'Нерастворимая',
    'ККал',
]


def load_products(path: str):
    """Load product information from CSV.

    Returns a list of dicts with nutrient composition per 100g, step size and
    maximal allowable weight.
    """
    products = []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nutrients = [float(row[key]) for key in NUTRIENT_KEYS]
            step = float(row['Шаг'])
            max_weight = float(row['Макс. порций']) * step
            products.append(
                {
                    'name': row['Продукт'],
                    'nutrients': nutrients,
                    'step': step,
                    'max_weight': max_weight,
                    'weight': 0.0,
                }
            )
    return products


def solve_nonnegative_least_squares(target, products):
    """Solve NNLS for the given target and products returning step counts."""
    K = len(target)
    N = len(products)
    A = np.zeros((K, N))
    for j, product in enumerate(products):
        s = product['step']
        p = np.array(product['nutrients'])
        A[:, j] = (s / 100.0) * p  # nutrient contribution of one step
    b = np.array(target)
    steps, _ = nnls(A, b)
    return steps


def calculate_calories(values):
    """Compute calories from macronutrient values."""
    proteins = values['Белки']
    fats = values['Насыщенные'] + values['НЕнасыщенные']
    carbs = values['Простые'] + values[COMPLEX_KEY]
    fiber = values['Растворимая'] + values['Нерастворимая']
    return proteins * 4 + fats * 9 + carbs * 4 + fiber * 1.5


def optimize(target, products, alpha_percent):
    """Optimize product weights for a fixed alpha (percentage of residual)."""
    K = len(target)
    r = target[:]
    s = [p['step'] for p in products]
    max_weight = [p['max_weight'] for p in products]
    p_matrix = [p['nutrients'] for p in products]
    x = [0.0] * len(products)
    alpha = alpha_percent / 100.0

    while True:
        target_vec = [alpha * rk if rk > 0 else 0.0 for rk in r]
        if all(val <= 0 for val in target_vec):
            break
        delta_steps = solve_nonnegative_least_squares(target_vec, products)

        any_added = False
        for j in range(len(products)):
            available_steps = (max_weight[j] - x[j]) / s[j]
            if available_steps <= 0:
                continue
            d = min(delta_steps[j], available_steps)
            d = int(round(d))
            if d <= 0:
                continue
            if d > available_steps:
                d = int(available_steps)
                if d <= 0:
                    continue
            add_weight = d * s[j]
            x[j] += add_weight
            any_added = True
            for k in range(K):
                r[k] -= (p_matrix[j][k] / 100.0) * add_weight
        if not any_added or all(val <= 0 for val in r):
            break

    rmse = (sum(val * val for val in r) / K) ** 0.5
    return x, r, rmse


def main():
    products = load_products('Nutrients DB.csv')
    print('Введите целевые значения для нутриентов (в граммах):')
    target = []
    values = {}
    for key in NUTRIENT_KEYS[:-1]:
        label = key.replace('\n', ' ')
        while True:
            try:
                val = float(input(f'{label}: '))
                break
            except ValueError:
                print('Введите числовое значение')
        target.append(val)
        values[key] = val

    calories = calculate_calories(values)
    target.append(calories)

    while True:
        try:
            start_alpha = int(input('Начальное значение альфа (1-100): '))
            if 1 <= start_alpha <= 100:
                break
            print('Введите число от 1 до 100')
        except ValueError:
            print('Введите целое число')

    while True:
        try:
            runs_per_alpha = int(input('Количество прогонов для каждого альфа (0-100): '))
            if 0 <= runs_per_alpha <= 100:
                break
            print('Введите число от 0 до 100')
        except ValueError:
            print('Введите целое число')

    best = None
    repeats = max(1, runs_per_alpha)
    for alpha in range(start_alpha, 101):
        for run in range(1, repeats + 1):
            weights, residual, rmse = optimize(target, products, alpha)
            if best is None or rmse < best['rmse']:
                best = {
                    'alpha': alpha,
                    'run': run,
                    'weights': weights,
                    'residual': residual,
                    'rmse': rmse,
                }

    achieved = [t - r for t, r in zip(target, best['residual'])]
    print('Сравнение нутриентов:')
    print(f"{'Нутриент':<20}{'Цель':>10}{'Рацион':>10}")
    for key, tgt, act in zip(NUTRIENT_KEYS, target, achieved):
        label = key.replace('\n', ' ')
        print(f"{label:<20}{tgt:>10.1f}{act:>10.1f}")

    print(
        f"\nМинимальная RMSE: {best['rmse']:.4f} при Альфа={best['alpha']} (прогон {best['run']})"
    )
    print('Продукты и граммовки:')
    for prod, w in zip(products, best['weights']):
        if w > 0:
            print(f"- {prod['name']}: {w:.1f} г")


if __name__ == '__main__':
    main()
