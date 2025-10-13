#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
import platform

print("🌟 Setting up MAGIF… 🪄")

# detect OS
system = platform.system()
if system == "Windows":
    venv_activate = r"magif-env\Scripts\activate"
    venv_python = r"magif-env\Scripts\python.exe"
else:
    venv_activate = "./magif-env/bin/activate"
    venv_python = "./magif-env/bin/python"

# create virtual environment
if not os.path.exists("magif-env"):
    print("🌟 Creating virtual environment…")
    subprocess.check_call([sys.executable, "-m", "venv", "magif-env"])
else:
    print("✅ Virtual environment already exists.")

# upgrade pip & tools
print("🌟 Upgrading pip & setuptools…")
subprocess.check_call([venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

# install dependencies in editable mode
print("🌟 Installing project in editable mode…")
subprocess.check_call([venv_python, "-m", "pip", "install", "-e", "."])

# check for SWI-Prolog
print("🌟 Checking for SWI-Prolog…")
if shutil.which("swipl") is not None:
    subprocess.run(["swipl", "--version"])
else:
    print("⚠️ SWI-Prolog is not installed.")
    print("   Please install it manually:")
    if system == "Linux":
        print("   Ubuntu/Debian: sudo apt install swi-prolog")
    elif system == "Darwin":
        print("   macOS (Homebrew): brew install swi-prolog")
    else:
        print("   Windows: https://www.swi-prolog.org/Download.html")

print("✨ Setup complete, time for some MAGiF! 🪄")

print(f"\nTo activate your virtual environment, run:\n")
if system == "Windows":
    print(r"   magif-env\Scripts\activate")
else:
    print("   source magif-env/bin/activate")
print("\nTo deactivate, just type: deactivate")

