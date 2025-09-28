import glob
import os
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


def find_sample(pattern):
    files = glob.glob(pattern)
    return files[0] if files else None


@pytest.mark.skipif(
    not (os.path.isdir("MRI") and os.path.isdir("PET")),
    reason="Local MRI/PET folders not found; skipping local integration test.")
def test_predict_endpoint_local():
    mri_sample = find_sample("MRI/test/*/*")
    pet_sample = find_sample("PET/test/*/*")
    if not mri_sample or not pet_sample:
        pytest.skip("Samples not found in MRI/PET test folders")

    client = TestClient(app)
    with open(mri_sample, "rb") as mf, open(pet_sample, "rb") as pf:
        resp = client.post(
            "/predict",
            files={
                "mri_file": (os.path.basename(mri_sample), mf, "image/jpeg"),
                "pet_file": (os.path.basename(pet_sample), pf, "image/jpeg"),
            },
            data={"patient_id": "it_local_test"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "predictions" in data["result"]
    assert "fusion" in data["result"]["predictions"]
