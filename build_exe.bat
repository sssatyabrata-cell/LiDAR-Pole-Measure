@echo off
setlocal
cd /d "%~dp0"

set "EXE_PATH=%cd%\dist\lidar_pole_measure.exe"
set "ONEDIR_EXE=%cd%\dist\lidar_pole_measure\lidar_pole_measure.exe"
set "TEMP_DIST=%cd%\dist_new"
set "TEMP_EXE=%TEMP_DIST%\lidar_pole_measure.exe"
set "TEMP_ONEDIR_EXE=%TEMP_DIST%\lidar_pole_measure\lidar_pole_measure.exe"
set "ASSETS_DIR=%cd%\assets"
set "ICON_PNG=%cd%\assets\pole_icon.png"
set "ICON_ICO=%cd%\assets\pole_icon.ico"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3 -m venv .venv
)

echo Closing any running lidar_pole_measure.exe...
taskkill /F /IM lidar_pole_measure.exe >nul 2>&1

if exist "%TEMP_DIST%" rmdir /S /Q "%TEMP_DIST%" >nul 2>&1
if exist "%cd%\dist\lidar_pole_measure_prev.exe" del /F /Q "%cd%\dist\lidar_pole_measure_prev.exe" >nul 2>&1

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

echo Detecting icon source in assets...
python -c "from pathlib import Path; import sys; root=Path(r'%ASSETS_DIR%'); stems=('pole_icon','app_icon','icon'); cand=[]; [cand.append(p) for s in stems for p in root.glob(s+'.*') if p.is_file()]; [cand.append(p) for p in root.iterdir() if p.is_file() and p.stem.lower() in stems and p not in cand]; print(cand[0] if cand else '')" > "%temp%\lpm_icon_src.txt"
set "ICON_SRC="
set /p ICON_SRC=<"%temp%\lpm_icon_src.txt"
if not "%ICON_SRC%"=="" (
  echo Preparing ICO from: %ICON_SRC%
  if exist "%ICON_ICO%" del /F /Q "%ICON_ICO%" >nul 2>&1
  if /I "%ICON_SRC%"=="%ICON_ICO%" (
    if exist "%ICON_ICO%" (
      echo Using existing ICO: %ICON_ICO%
    ) else (
      echo Existing ICO path was detected but file is missing.
      set "ICON_SRC="
    )
  ) else (
    python -c "from PIL import Image; img=Image.open(r'%ICON_SRC%').convert('RGBA'); img.save(r'%ICON_ICO%', format='ICO', sizes=[(16,16),(24,24),(32,32),(40,40),(48,48),(64,64),(128,128),(256,256)])"
  )
)
if exist "%ICON_ICO%" (
  echo Embedding icon: %ICON_ICO%
) else (
  echo Icon not found. Place icon as pole_icon/app_icon/icon in assets.
)

if exist "%ICON_ICO%" (
  python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --name lidar_pole_measure ^
    --distpath "%TEMP_DIST%" ^
    --icon "%ICON_ICO%" ^
    --add-data "assets;assets" ^
    --paths src ^
    --collect-all sklearn ^
    --collect-all scipy ^
    --collect-all joblib ^
    --collect-all threadpoolctl ^
    --collect-all pyproj ^
    --collect-all shapely ^
    --collect-all shapefile ^
    run_app.py
) else (
  python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --name lidar_pole_measure ^
    --distpath "%TEMP_DIST%" ^
    --add-data "assets;assets" ^
    --paths src ^
    --collect-all sklearn ^
    --collect-all scipy ^
    --collect-all joblib ^
    --collect-all threadpoolctl ^
    --collect-all pyproj ^
    --collect-all shapely ^
    --collect-all shapefile ^
    run_app.py
)

if errorlevel 1 (
  echo.
  echo Build failed.
  pause
  exit /b 1
)

if exist "%TEMP_EXE%" (
  if not exist "%cd%\dist" mkdir "%cd%\dist"
  copy /Y "%TEMP_EXE%" "%EXE_PATH%" >nul
  rmdir /S /Q "%TEMP_DIST%" >nul 2>&1
  echo.
  echo Build complete. EXE path:
  echo %EXE_PATH%
  pause
  exit /b 0
)

echo.
echo Onefile EXE not found in dist. Trying onedir fallback...
if exist "%ICON_ICO%" (
  python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onedir ^
    --windowed ^
    --name lidar_pole_measure ^
    --distpath "%TEMP_DIST%" ^
    --icon "%ICON_ICO%" ^
    --add-data "assets;assets" ^
    --paths src ^
    --collect-all sklearn ^
    --collect-all scipy ^
    --collect-all joblib ^
    --collect-all threadpoolctl ^
    --collect-all pyproj ^
    --collect-all shapely ^
    --collect-all shapefile ^
    run_app.py
) else (
  python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onedir ^
    --windowed ^
    --name lidar_pole_measure ^
    --distpath "%TEMP_DIST%" ^
    --add-data "assets;assets" ^
    --paths src ^
    --collect-all sklearn ^
    --collect-all scipy ^
    --collect-all joblib ^
    --collect-all threadpoolctl ^
    --collect-all pyproj ^
    --collect-all shapely ^
    --collect-all shapefile ^
    run_app.py
)

if errorlevel 1 (
  echo.
  echo Build failed in both onefile and onedir modes.
  echo Check antivirus quarantine and Windows Defender history.
  pause
  exit /b 1
)

if exist "%TEMP_ONEDIR_EXE%" (
  if exist "%cd%\dist\lidar_pole_measure" rmdir /S /Q "%cd%\dist\lidar_pole_measure" >nul 2>&1
  xcopy /E /I /Y "%TEMP_DIST%\lidar_pole_measure" "%cd%\dist\lidar_pole_measure" >nul
  rmdir /S /Q "%TEMP_DIST%" >nul 2>&1
  echo.
  echo Build complete (onedir). EXE path:
  echo %cd%\dist\lidar_pole_measure\lidar_pole_measure.exe
  pause
  exit /b 0
)

echo.
echo Build finished but no EXE was found.
echo Possible reasons: antivirus quarantine, file lock, or permission issues.
echo Check:
echo 1) Windows Security -> Protection history
echo 2) %cd%\dist
echo 3) %cd%\build
pause
