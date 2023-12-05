from fastapi.testclient import TestClient
from main import app



client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_read_item():
    response = client.get("/items")
    assert response.status_code == 200
    assert response.json() == [{"name": "Foo"}, {"name": "Bar"}]
