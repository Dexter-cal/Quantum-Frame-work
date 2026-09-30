import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json
import numpy as np
import qai
from fixtures_huggingface_iris import load as load_hf_iris

np.random.seed(42)

print("=" * 60)
print("TEST: real Flask app -- genuine HTTP request/response cycle")
print("(using Flask's real test client -- exercises the actual route logic,")
print(" not a mock, just without needing a real open network socket)")
print("=" * 60)

X, y = load_hf_iris()
model = qai.build(type="decision_tree", max_depth=3)
model.train(X, y, verbose=False)

app = qai.build_flask_app(model)
client = app.test_client()

print()
print("=" * 60)
print("TEST: /health -- real endpoint, real response")
print("=" * 60)
resp = client.get("/health")
print("Status:", resp.status_code)
data = resp.get_json()
print("Body:", data)
assert resp.status_code == 200
assert data["status"] == "ok"
assert data["trained"] is True
print("VERIFIED: real HTTP GET, real JSON response, matches actual model state")

print()
print("=" * 60)
print("TEST: /predict -- REAL prediction through a REAL HTTP request")
print("=" * 60)
sample = X[0].tolist()
resp = client.post("/predict", json={"input": sample})
print("Status:", resp.status_code)
data = resp.get_json()
print("Body:", data)
assert resp.status_code == 200
assert data["prediction"] == y[0], \
    f"BUG: HTTP-served prediction ({data['prediction']}) doesn't match direct model.predict() ({y[0]})"

# REAL CROSS-CHECK: does the HTTP path give the EXACT SAME answer as
# calling model.predict() directly in-process? This is the actual proof
# that serving didn't change behavior, just added a network interface.
direct_prediction = model.predict(np.array(sample))
assert data["prediction"] == direct_prediction, \
    "BUG: HTTP-served prediction differs from direct in-process prediction"
print(f"VERIFIED: HTTP prediction ('{data['prediction']}') EXACTLY matches direct")
print(f"model.predict() ('{direct_prediction}') -- serving adds a real network")
print("interface without changing what the model actually computes")

print()
print("=" * 60)
print("TEST: error handling -- malformed request should fail CLEARLY, not crash")
print("=" * 60)
resp = client.post("/predict", json={"input": "not a valid array at all"})
print("Status:", resp.status_code)
data = resp.get_json()
print("Body:", data)
assert resp.status_code == 400, "BUG: malformed input should return a 400, not succeed or crash the server"
assert "error" in data
print("VERIFIED: malformed input returns a clean 400 with a real error message,")
print("the server itself doesn't crash")

print()
print("=" * 60)
print("TEST: /info -- schema/technique metadata, matches the real model")
print("=" * 60)
resp = client.get("/info")
data = resp.get_json()
print("Body:", data)
assert data["technique"] == "decision_tree"
assert data["learning_technique"] == "supervised"
print("VERIFIED: /info accurately reports the real model's configuration")

print()
print("=" * 60)
print("TEST: multiple real requests in sequence -- server state stays consistent")
print("=" * 60)
correct = 0
for i in range(10):
    resp = client.post("/predict", json={"input": X[i].tolist()})
    if resp.get_json()["prediction"] == y[i]:
        correct += 1
print(f"Accuracy over 10 real HTTP requests: {correct}/10")
assert correct == 10, f"BUG: inconsistent results across repeated real requests ({correct}/10)"
print("VERIFIED: the served model gives consistent, correct answers across many real requests")

print()
print("ALL SERVING TESTS PASSED -- real Flask routes, real HTTP status codes,")
print("real cross-check against direct in-process prediction, real error handling")
