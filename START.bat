@echo off
chcp 65001 >nul
echo.
echo  ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ███████╗██╗███╗   ███╗
echo  ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔════╝██║████╗ ████║
echo  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║███████╗██║██╔████╔██║
echo  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║╚════██║██║██║╚██╔╝██║
echo  ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝███████║██║██║ ╚═╝ ██║
echo  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝╚═╝     ╚═╝
echo.
echo  Brain Simulation Platform — Windows Setup ^& Launcher
echo  =====================================================
echo.

REM ── Step 1: Check Python ──────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found!
    echo.
    echo  Please install Python 3.10+ from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] Python %PYVER% found

REM ── Step 2: Create virtual environment ───────────────────────────────
if not exist ".venv" (
    echo  [..] Creating virtual environment...
    python -m venv .venv
    echo  [OK] Virtual environment created
) else (
    echo  [OK] Virtual environment already exists
)

REM ── Step 3: Activate venv ────────────────────────────────────────────
call .venv\Scripts\activate.bat
echo  [OK] Virtual environment activated

REM ── Step 4: Upgrade pip ──────────────────────────────────────────────
echo  [..] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM ── Step 5: Install dependencies ─────────────────────────────────────
echo  [..] Installing NeuroSim dependencies (this takes ~2 minutes)...
echo       numpy scipy numba plotly dash click pyyaml rich tqdm h5py
echo.
pip install ^
    numpy>=1.24 ^
    scipy>=1.10 ^
    numba>=0.57 ^
    plotly>=5.15 ^
    dash>=2.11 ^
    click>=8.1 ^
    pyyaml>=6.0 ^
    rich>=13 ^
    tqdm>=4.65 ^
    h5py>=3.8 ^
    --quiet

if errorlevel 1 (
    echo  [ERROR] Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)
echo  [OK] All dependencies installed

REM ── Step 6: Add neurosim to path ─────────────────────────────────────
set PYTHONPATH=%~dp0python;%PYTHONPATH%

REM ── Step 7: Run the simulation + launch dashboard ────────────────────
echo.
echo  [..] Running simulation and launching dashboard...
echo.
python run_dashboard.py

pause
