@echo off
setlocal

for %%I in ("%~dp0.") do set "ROOT=%%~fI"
set "BUILD_DIR=%ROOT%\build"

if not defined OpenCV_DIR (
  set "OpenCV_DIR="
  for %%D in (
    "%ROOT%\..\opencv_prebuilt\build\x64\vc16\lib"
    "%ROOT%\..\opencv-camera\opencv_prebuilt\build\x64\vc16\lib"
    "C:\dev\opencv_camera_20260626_2_opencv_prebuilt\opencv_prebuilt\build\x64\vc16\lib"
    "C:\dev\opencv-camera\opencv_prebuilt\build\x64\vc16\lib"
  ) do (
    if exist "%%~D\OpenCVConfig.cmake" (
      set "OpenCV_DIR=%%~D"
      goto OpenCVFound
    )
  )
)
:OpenCVFound

if not defined VISP_DIR (
  set "VISP_DIR="
  for %%D in (
    "%ROOT%\dependencies\visp-install\x64\vc18\lib"
    "%ROOT%\..\dependencies\visp-install\x64\vc18\lib"
    "%ROOT%\..\visp-install\x64\vc18\lib"
    "%ROOT%\..\..\dependencies\visp-install\x64\vc18\lib"
  ) do (
    if exist "%%~D\VISPConfig.cmake" (
      set "VISP_DIR=%%~D"
      goto VISPFound
    )
  )
)
:VISPFound

where cmake >nul 2>nul
if errorlevel 1 (
  echo Error: CMake was not found on PATH.
  exit /b 1
)
if not exist "%OpenCV_DIR%\OpenCVConfig.cmake" (
  echo Error: OpenCVConfig.cmake was not found. Set OpenCV_DIR to the OpenCV build directory before running build.bat.
  exit /b 1
)
if not exist "%VISP_DIR%\VISPConfig.cmake" (
  echo Error: VISPConfig.cmake was not found. Set VISP_DIR to the ViSP install directory before running build.bat.
  exit /b 1
)

cmake -S "%ROOT%" -B "%BUILD_DIR%" -A x64 "-DOpenCV_DIR=%OpenCV_DIR%" "-DVISP_DIR=%VISP_DIR%"
if errorlevel 1 exit /b %ERRORLEVEL%

cmake --build "%BUILD_DIR%" --config Release --parallel 8
exit /b %ERRORLEVEL%
