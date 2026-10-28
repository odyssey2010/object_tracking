@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 exit /b 1

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Python virtual environment not found. Follow the installation steps in Yolo_Segmentation.md. 1>&2
    popd
    exit /b 1
)

".venv\Scripts\python.exe" "yolo_seg.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
