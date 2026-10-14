@echo off
cd /d "%~dp0"
set "BACKGROUND_PREVIEW_SECONDS=30"
python replace_background.py
if errorlevel 1 (
  echo.
  echo 30-second preview failed. Read the error above.
  pause
  exit /b 1
)
echo.
echo 30-second preview complete. Open the output folder.
pause
