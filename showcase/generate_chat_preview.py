"""Generate a screenshot of the chat UI as a static HTML for the showcase."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
import warnings; warnings.filterwarnings('ignore')
import qai
import numpy as np

# Build and train a small classifier
m = qai.build(technique="classifier")
X = np.random.rand(50, 4)
y = (X[:, 0] + X[:, 1] > 1).astype(int)
m.train(X, y, verbose=False)

# Get the chat HTML by making a single GET to the chat endpoint
import threading, time, urllib.request
def run():
    try:
        qai.chat_with_model(m, port=8773, open_browser=False, timeout_s=5)
    except Exception:
        pass

t = threading.Thread(target=run, daemon=True)
t.start()
time.sleep(1.0)

# Get the HTML
with urllib.request.urlopen("http://127.0.0.1:8773/") as resp:
    html = resp.read().decode("utf-8")

# Save it as a static HTML showcase
out = os.path.join(os.path.dirname(__file__), "chat_preview.html")
with open(out, "w") as f:
    f.write(html)
print(f"Saved: {out} ({len(html)} bytes)")
