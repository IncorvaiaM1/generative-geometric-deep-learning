"""Train NICE then VAE and write sample grids."""

import os
import subprocess
import sys

os.makedirs("results", exist_ok=True)

cmds = [
    [sys.executable, "-u", "train_nice.py"],
    [sys.executable, "-u", "train_vae.py"],
]

for cmd in cmds:
    print("=" * 80, flush=True)
    print("Running", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
