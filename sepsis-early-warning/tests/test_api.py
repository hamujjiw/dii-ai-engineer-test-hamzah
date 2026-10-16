from fastapi.testclient import TestClient

from api.main import app

HIGH = {"heart_rate": 122, "respiratory_rate": 29, "temperature": 39.2, "wbc_count": 18.1,
        "lactate_level": 4.7, "age": 83, "num_comorbidities": 4}
LOW = {"heart_rate": 83, "respiratory_rate": 15, "temperature": 36.8, "wbc_count": 6.8,
       "lactate_level": 1.0, "age": 36, "num_comorbidities": 1}


def client():
    return TestClient(app)


def test_health():
    with client() as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_high_risk_patient():
    with client() as c:
        r = c.post("/predict", json=HIGH)
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"sepsis_risk_prediction", "risk_probability"}
        assert body["sepsis_risk_prediction"] == 1 and body["risk_probability"] > 0.5


def test_low_risk_patient():
    with client() as c:
        body = c.post("/predict", json=LOW).json()
        assert body["sepsis_risk_prediction"] == 0 and body["risk_probability"] < 0.5


def test_lactate_may_be_missing():
    with client() as c:
        payload = {**LOW, "lactate_level": None}
        assert c.post("/predict", json=payload).status_code == 200
        payload.pop("lactate_level")
        assert c.post("/predict", json=payload).status_code == 200


def test_rejects_implausible_value():
    with client() as c:
        assert c.post("/predict", json={**LOW, "temperature": 80}).status_code == 422


def test_rejects_missing_field():
    with client() as c:
        payload = {k: v for k, v in LOW.items() if k != "heart_rate"}
        assert c.post("/predict", json=payload).status_code == 422
