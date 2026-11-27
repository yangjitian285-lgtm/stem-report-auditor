@echo off
chcp 65001 >nul
cd /d "%~dp0"
title STEM Report Auditor
where python >nul 2>nul
if errorlevel 1 (
  echo Python が見つかりません。https://www.python.org/downloads/ からインストールしてください。
  echo インストール画面で「Add python.exe to PATH」にチェックを入れてください。
  pause
  exit /b 1
)
python -c "import numpy" >nul 2>nul
if errorlevel 1 (
  echo 初回のみ、必要な部品をインストールします...
  python -m pip install -r requirements.txt
)
python app.py
pause
