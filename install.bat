@echo off
chcp 65001 >nul
cd /d %~dp0
echo Installing dependencies (Aliyun mirror)...
python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
echo.
echo Done. Run start.bat to launch.
pause
