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


def optimize(target, products):
    K = len(target)
    r = target[:]
    s = [p['step'] for p in products]
    max_weight = [p['max_weight'] for p in products]
    p_matrix = [p['nutrients'] for p in products]
    x = [0.0] * len(products)

    while True:
        alpha_list = []
        for k in range(K):
            if r[k] > 0:
                ratios = []
                for j in range(len(products)):
                    val = (s[j] / 100.0) * p_matrix[j][k]
                    if val > 0:
                        ratios.append(val / r[k])
                if ratios:
                    alpha_list.append(min(ratios))
        if not alpha_list:
            break
        alpha = max(alpha_list)
        target_vec = [alpha * rk for rk in r]
        delta_steps = solve_nonnegative_least_squares(target_vec, products)

        any_added = False
        for j in range(len(products)):
            available_steps = (max_weight[j] - x[j]) / s[j]
            if available_steps < 1:
                continue
            d = min(delta_steps[j], available_steps)
            if d < 1:
                d = 1
            else:
                d = int(round(d))
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
    print('Введите целевые значения для нутриентов (в граммах или ккал):')
    target = []
    for key in NUTRIENT_KEYS:
        label = key.replace('\n', ' ')
        while True:
            try:
                val = float(input(f'{label}: '))
                break
            except ValueError:
                print('Введите числовое значение')
        target.append(val)

    weights, residual, rmse = optimize(target, products)
    print('Подобранные веса:')
    for prod, w in zip(products, weights):
        if w > 0:
            print(f"- {prod['name']}: {w:.1f} г")
    print('RMSE:', round(rmse, 4))


if __name__ == '__main__':
    main()
