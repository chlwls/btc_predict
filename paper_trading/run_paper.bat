@echo off
cd /d "%~dp0"
python paper_trade.py >> paper_console.log 2>&1
python paper_trade_multi.py >> paper_console.log 2>&1
