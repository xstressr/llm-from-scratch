@echo off
REM Compile llm.c train_gpt2.c with MSVC Build Tools 2022. Does not touch CUDA binaries.
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS%" || exit /b 1
cd /d "%~dp0vendor\llm.c" || exit /b 1
cl /Idev /nologo /O2 /Oi /Ot /fp:fast /MD /openmp /EHsc /wd4996 /TP train_gpt2.c /Fe:train_gpt2.exe
REM C4849 (collapse on parallel for) is expected on MSVC; OpenMP still runs.
