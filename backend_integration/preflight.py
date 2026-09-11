"""SkyGuard AI release preflight. Fails closed on missing intelligence layers."""
import importlib.util
import os, sys

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND=os.path.join(ROOT,"backend_integration")
for path in (ROOT,BACKEND):
    if path not in sys.path:
        sys.path.insert(0,path)

print("SkyGuard AI FINAL PREFLIGHT")
print("Python:",sys.version.split()[0])
for name in ["numpy","pandas","sklearn","fastapi","uvicorn","shap","streamlit"]:
    print(f"{name:12}", "OK" if importlib.util.find_spec(name) else "MISSING")

from main import runtime_status
runtime=runtime_status()
print("Agent 2     ", runtime["agent2"])
print("Agent 3     ", runtime["agent3"])
print("Agent 4     ", runtime["agent4"])
print("Agent 5     ", runtime["agent5"])

model=os.path.join(ROOT,"agent3_ml_detection","agent3_ml_detection_repo","models","lstm_autoencoder.keras")
print("LSTM artifact", "OK" if os.path.exists(model) else "MISSING")

if not runtime["ready"]:
    raise SystemExit("PREFLIGHT FAILED: one or more required real agents are unavailable.")
print("PREFLIGHT PASS: all required runtime agents are available.")
