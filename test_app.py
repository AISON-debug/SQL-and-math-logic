import json
import shutil
from pathlib import Path
import importlib

import pytest

FIELDNAMES = ["Продукт","Белки","Насыщенные","НЕнасыщенные","Простые",
              "Сложные перевариваемые","Растворимая","Нерастворимая",
              "ККал","Макс. порций","Шаг"]

@pytest.fixture
def client(tmp_path, monkeypatch):
    src = Path(__file__).with_name('Nutrients DB.csv')
    dst = tmp_path / 'db.csv'
    shutil.copy(src, dst)
    monkeypatch.setenv('NUTRIENTS_DB', str(dst))
    import app
    importlib.reload(app)
    return app.app.test_client()


def sample_product(name):
    return {fn: '1' for fn in FIELDNAMES} | {"Продукт": name}


def test_crud_flow(client):
    resp = client.get('/api/products')
    assert resp.status_code == 200
    data = resp.get_json()
    base_count = len(data)

    new_prod = sample_product('Тест')
    resp = client.post('/api/products', json=new_prod)
    assert resp.status_code == 201

    resp = client.get('/api/products')
    assert len(resp.get_json()) == base_count + 1

    updated = sample_product('Тест')
    updated['Белки'] = '2'
    resp = client.put('/api/products/Тест', json=updated)
    assert resp.status_code == 200

    resp = client.delete('/api/products/Тест')
    assert resp.status_code == 200

    resp = client.get('/api/products')
    assert len(resp.get_json()) == base_count
