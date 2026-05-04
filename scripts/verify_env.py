# scripts/verify_env.py
import sys, importlib

required = {
    "torch": "2.2",
    "torchvision": "0.17",
    "flwr": "1.8",
    "numpy": "1.26",
    "PIL": None,
    "sklearn": None,
    "fastapi": "0.109",
    "cv2": None,
    "matplotlib": None,
    "streamlit": "1.35",
    "plotly": None,
    "tqdm": None,
    "yaml": None,
}

all_ok = True
print("Checking packages...\n")
for mod, ver in required.items():
    try:
        m = importlib.import_module(mod)
        installed = getattr(m, "__version__", "ok")
        status = "✓" if (ver is None or installed.startswith(ver)) else "⚠"
        print(f"  {status}  {mod}: {installed}")
    except Exception as e:
        print(f"  ✗  {mod}: ERROR — {e}")
        all_ok = False

import torch
print(f"\n  GPU available : {torch.cuda.is_available()}")
print(f"  Device        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (fine for this project)'}")

print()
if all_ok:
    print("✅  CHECKPOINT 0 PASSED — Environment is ready!")
else:
    print("❌  CHECKPOINT 0 FAILED — See errors above")
    sys.exit(1)