@echo off
cd /d "%~dp0vendor\llama2.c"
run.exe out\model.bin -t 1.0 -p 0.9 -n 256 -i "Once upon a time"
