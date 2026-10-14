@echo off
setlocal

for %%I in ("%~dp0.") do set "ROOT=%%~fI"
set "EXE=%ROOT%\build\Release\object_tracking.exe"
set "MODEL=%ROOT%\models\dope_mustard_bottle.onnx"
set "CONFIG=%ROOT%\models\mustard.xml"
set "OBJECT_MODEL=%ROOT%\models\mustard.cao"
set "CAMERA_INDEX=%~1"
if not defined CAMERA_INDEX set "CAMERA_INDEX=0"

if not exist "%EXE%" (
  echo Error: "%EXE%" was not found. Run build.bat first.
  exit /b 1
)
if not exist "%MODEL%" (
  echo Error: DOPE ONNX model was not found: "%MODEL%"
  exit /b 1
)
if not exist "%CONFIG%" (
  echo Error: ViSP tracker configuration was not found: "%CONFIG%"
  exit /b 1
)
if not exist "%OBJECT_MODEL%" (
  echo Error: ViSP object model was not found: "%OBJECT_MODEL%"
  exit /b 1
)

"%EXE%" "%MODEL%" "%CONFIG%" "%OBJECT_MODEL%" "%CAMERA_INDEX%"
exit /b %ERRORLEVEL%
