@echo off
cd /d "%~dp0"
rem 3단계에서 아래 줄 끝에 --live 를 붙이면 실제 주문이 나갑니다.
python live_trade.py >> live_console.log 2>&1
