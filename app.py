import csv
import os
from flask import Flask, jsonify, request

CSV_FILE = os.environ.get('NUTRIENTS_DB', 'Nutrients DB.csv')
FIELDNAMES = ['Продукт', 'Белки', 'Насыщенные', 'НЕнасыщенные', 'Простые',
              'Сложные перевариваемые', 'Растворимая', 'Нерастворимая',
              'ККал', 'Макс. порций', 'Шаг']

app = Flask(__name__, static_folder='static', static_url_path='')

def _sanitize(row):
    def clean_key(k: str) -> str:
        return " ".join(k.replace('\r', ' ').replace('\n', ' ').split())
    return {clean_key(k): v for k, v in row.items()}

def read_products():
    products = []
    with open(CSV_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            products.append(_sanitize(row))
    return products

def write_products(products):
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, FIELDNAMES)
        writer.writeheader()
        for p in products:
            writer.writerow(p)

@app.route('/api/products', methods=['GET'])
def get_products():
    return jsonify(read_products())

@app.route('/api/products', methods=['POST'])
def add_product():
    data = request.get_json()
    if not data or any(field not in data or data[field] == '' for field in FIELDNAMES):
        return jsonify({'error': 'Все поля обязательны'}), 400
    products = read_products()
    if any(p['Продукт'] == data['Продукт'] for p in products):
        return jsonify({'error': 'Продукт уже существует'}), 400
    products.append({field: data[field] for field in FIELDNAMES})
    write_products(products)
    return jsonify({'status': 'created'}), 201

@app.route('/api/products/<string:name>', methods=['PUT'])
def edit_product(name):
    data = request.get_json()
    if not data or any(field not in data or data[field] == '' for field in FIELDNAMES):
        return jsonify({'error': 'Все поля обязательны'}), 400
    products = read_products()
    for idx, p in enumerate(products):
        if p['Продукт'] == name:
            products[idx] = {field: data[field] for field in FIELDNAMES}
            write_products(products)
            return jsonify({'status': 'updated'})
    return jsonify({'error': 'Не найдено'}), 404

@app.route('/api/products/<string:name>', methods=['DELETE'])
def delete_product(name):
    products = read_products()
    new_products = [p for p in products if p['Продукт'] != name]
    if len(new_products) == len(products):
        return jsonify({'error': 'Не найдено'}), 404
    write_products(new_products)
    return jsonify({'status': 'deleted'})

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)
