import csv
import math
import random
import numpy as np


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

WEIGHTS = {
    'Белки': 2,
    'Насыщенные': 1,
    'НЕнасыщенные': 1,
    'Простые': 1,
    COMPLEX_KEY: 1,
    'Растворимая': 1,
    'Нерастворимая': 1,
    'ККал': 3,
}


def js_round(x: float) -> int:
    """JavaScript-like rounding (half away from zero)."""
    return int(math.floor(x + 0.5))


def load_products(path: str):
    """Load product information from CSV."""
    products = []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nutrient_map = {key: float(row[key]) for key in NUTRIENT_KEYS}
            step = float(row['Шаг'])
            max_weight = float(row['Макс. порций']) * step
            products.append(
                {
                    'name': row['Продукт'],
                    'nutrients': nutrient_map,
                    'step': step,
                    'max_weight': max_weight,
                }
            )
    return products


def calculate_calories(values):
    proteins = values['Белки']
    fats = values['Насыщенные'] + values['НЕнасыщенные']
    carbs = values['Простые'] + values[COMPLEX_KEY]
    fiber = values['Растворимая'] + values['Нерастворимая']
    return proteins * 4 + fats * 9 + carbs * 4 + fiber * 1.5


def run_iterative_optimization(var_idxs, resid, products, residual_fraction):
    resid_vec = resid.copy()
    var_add = {idx: 0.0 for idx in var_idxs}
    step_vals = {idx: products[idx]['step'] for idx in var_idxs}
    max_vals = {idx: products[idx]['max_weight'] for idx in var_idxs}
    nut_keys = list(WEIGHTS.keys())
    active = var_idxs[:]
    iteration = 0
    max_iter = 10
    while active and iteration < max_iter:
        active = [
            idx
            for idx in active
            if step_vals[idx] > 0 and (max_vals[idx] - var_add[idx]) >= step_vals[idx] / 2
        ]
        if not active:
            break
        m = len(active)
        pmat = [[products[idx]['nutrients'][k] / 100.0 for k in nut_keys] for idx in active]
        G = np.zeros((m, m))
        b_vec = np.zeros(m)
        for i in range(m):
            sum_b = 0.0
            for k, key in enumerate(nut_keys):
                w = WEIGHTS[key]
                target_scaled = resid_vec[key] * residual_fraction
                sum_b += w * target_scaled * pmat[i][k]
            b_vec[i] = sum_b
            for j in range(m):
                sum_g = 0.0
                for k, key in enumerate(nut_keys):
                    w = WEIGHTS[key]
                    sum_g += w * pmat[i][k] * pmat[j][k]
                G[i, j] = sum_g
        active_idxs = active[:]
        active_G = G.copy()
        active_B = b_vec.copy()
        sol = None
        while True:
            m2 = len(active_idxs)
            if m2 == 0:
                break
            try:
                sol_vec = np.linalg.solve(active_G, active_B)
            except np.linalg.LinAlgError:
                break
            neg = [i for i, val in enumerate(sol_vec) if val < 0]
            if not neg:
                sol = sol_vec
                break
            mask = [i for i in range(m2) if i not in neg]
            active_idxs = [active_idxs[i] for i in mask]
            active_B = active_B[mask]
            active_G = active_G[np.ix_(mask, mask)]
        if sol is None:
            break
        any_positive = False
        for i, idx in enumerate(active_idxs):
            grams = sol[i]
            if grams <= 0:
                continue
            remaining = max_vals[idx] - var_add[idx]
            if remaining <= 0:
                continue
            if grams > remaining:
                grams = remaining
            step = step_vals[idx]
            if step > 0:
                rounded = js_round(grams / step) * step
                if rounded > remaining:
                    rounded = math.floor(remaining / step) * step
                if rounded < 0:
                    rounded = 0
                grams = rounded
            if grams <= 0:
                continue
            var_add[idx] += grams
            any_positive = True
            p = products[idx]['nutrients']
            for key in nut_keys:
                resid_vec[key] -= (p[key] / 100.0) * grams
        if not any_positive:
            break
        iteration += 1
    return var_add, resid_vec


def evaluate(target, products, var_idxs, alpha):
    resid = {k: target[k] for k in target}
    var_map, resid_vec = run_iterative_optimization(var_idxs, resid, products, alpha)
    totals = {k: target[k] - resid_vec[k] for k in target}
    err_sum = 0.0
    n_keys = 0
    for k in target:
        diff = target[k] - totals[k]
        err_sum += WEIGHTS[k] * diff * diff
        n_keys += 1
    rmse = math.sqrt(err_sum / n_keys) if n_keys else 0.0
    weights = [0.0] * len(products)
    for idx, grams in var_map.items():
        weights[idx] = grams
    residual = [resid_vec[k] for k in NUTRIENT_KEYS]
    return weights, residual, rmse


def main():
    products = load_products('Nutrients DB.csv')
    print('Введите целевые значения для нутриентов (в граммах):')
    target = {}
    for key in NUTRIENT_KEYS[:-1]:
        label = key.replace('\n', ' ')
        while True:
            try:
                val = float(input(f'{label}: '))
                break
            except ValueError:
                print('Введите числовое значение')
        target[key] = val
    target['ККал'] = calculate_calories(target)

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
    n_products = len(products)
    indices_all = list(range(n_products))
    for alpha in range(start_alpha, 101):
        for run in range(1, repeats + 1):
            indices = indices_all[:]
            random.shuffle(indices)
            weights, residual, rmse = evaluate(target, products, indices, alpha / 100.0)
            ordered = [0.0] * n_products
            for idx, w in zip(indices, weights):
                ordered[idx] = w
            if best is None or rmse < best['rmse']:
                best = {
                    'alpha': alpha,
                    'run': run,
                    'weights': ordered,
                    'residual': residual,
                    'rmse': rmse,
                }

    achieved = [target[k] - r for k, r in zip(NUTRIENT_KEYS, best['residual'])]
    print('Сравнение нутриентов:')
    print(f"{'Нутриент':<20}{'Цель':>10}{'Рацион':>10}")
    for key, tgt, act in zip(NUTRIENT_KEYS, [target[k] for k in NUTRIENT_KEYS], achieved):
        label = key.replace('\n', ' ')
        print(f"{label:<20}{tgt:>10.1f}{act:>10.1f}")

    print(
        f"\nМинимальная RMSE: {best['rmse']:.4f} при Альфа={best['alpha']} (прогон {best['run']})"
    )
    print('Продукты и граммовки:')
    for prod, w in zip(products, best['weights']):
        if w > 0:
            print(f"- {prod['name']}: {w:.2f} г")


if __name__ == '__main__':
    main()

