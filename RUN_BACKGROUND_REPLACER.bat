@echo off
cd /d "%~dp0"
python replace_background.py
if errorlevel 1 (
  echo.
  echo Background replacement failed. Read the error above.
  pause
  exit /b 1
)
echo.
echo Background replacement complete. Open the output folder.
pause
