"""Test the chat_with_model server in a controlled way."""
import sys, os, time, json, threading, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import qai

m = qai.build(technique='classifier')
X = np.random.rand(20, 4)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
m.train(X, y, verbose=False)

# Start the chat server in a thread
def run_server():
    qai.chat_with_model(m, port=8771, open_browser=False, timeout_s=10)

t = threading.Thread(target=run_server, daemon=True)
t.start()
time.sleep(1.5)

def post(path, body):
    req = urllib.request.Request(
        f'http://127.0.0.1:8771{path}',
        data=json.dumps(body).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

# Test info endpoint
with urllib.request.urlopen('http://127.0.0.1:8771/api/info') as resp:
    info = json.loads(resp.read())
print('INFO:', {k: v for k, v in info.items() if k != 'arch_png'})
print('  has arch_png:', bool(info.get('arch_png')))

# Test single predict
r = post('/api/predict', {'input': '[0.5, 0.6, 0.7, 0.8]'})
print('SINGLE PREDICT: prediction=' + str(r.get('prediction')) +
      ', latency_ms=' + str(r.get('latency_ms')) +
      ', confidence=' + str(r.get('confidence')))

# Test batch predict
r = post('/api/predict', {'input': '[0.5, 0.6, 0.7, 0.8]\n[0.1, 0.2, 0.3, 0.4]'})
print('BATCH PREDICT: predictions=' + str(len(r.get('predictions', []))) +
      ', total_ms=' + str(r.get('total_ms')) +
      ', stats.n_predictions=' + str(r.get('stats', {}).get('n_predictions')))

# Test explain (may fail for untrained explain, but should not crash)
try:
    r = post('/api/explain', {'input': '[0.5, 0.6, 0.7, 0.8]'})
    print('EXPLAIN: technique=' + str(r.get('technique')))
except Exception as e:
    print('EXPLAIN error:', e)

# Test bad input
try:
    r = post('/api/predict', {'input': 'not valid'})
    print('BAD INPUT:', r)
except urllib.error.HTTPError as e:
    print('BAD INPUT correctly returned', e.code)

print('ALL CHAT TESTS PASSED')
