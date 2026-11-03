@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"

set "CLI=%SCRIPT_DIR%\run_2d.py"

rem mode: search (default, the staged search) or scan (the tau x Ninter scan)
if "%mode%"=="" set "mode=search"
if "%nworkers%"=="" set "nworkers=4"
if not "%mode%"=="search" goto scan
set "SEARCH="
if not "%outdir%"=="" set "SEARCH=%SEARCH% --outdir=%outdir%"
if not "%seed%"=="" set "SEARCH=%SEARCH% --seed=%seed%"
if not "%mass%"=="" set "SEARCH=%SEARCH% --mass=%mass%"
if not "%seeds_per_rung%"=="" set "SEARCH=%SEARCH% --seeds_per_rung=%seeds_per_rung%"
if not "%explore_steps%"=="" set "SEARCH=%SEARCH% --explore_steps=%explore_steps%"
if not "%max_chains%"=="" set "SEARCH=%SEARCH% --max_chains=%max_chains%"
if "%committor_stage%"=="1" set "SEARCH=%SEARCH% --committor"
if not "%committor_iterations%"=="" set "SEARCH=%SEARCH% --committor_iterations=%committor_iterations%"
echo mode       : search
python "%CLI%" --nworkers=%nworkers%%SEARCH%
goto end

:scan
if "%outdir%"=="" set "outdir=."
if "%num_trials%"=="" set "num_trials=10"
if "%n_max%"=="" set "n_max=10000"
if "%threshold%"=="" set "threshold=0.5"
if "%n_compress%"=="" set "n_compress=10"

set "EXTRA="
if not "%seed%"=="" set "EXTRA=%EXTRA% --seed=%seed%"
if not "%mass%"=="" set "EXTRA=%EXTRA% --mass=%mass%"
rem space-separated lists; tau in fs, kt in the energy unit of the potential
if not "%tau%"=="" set "EXTRA=%EXTRA% --tau %tau%"
if not "%kt%"=="" set "EXTRA=%EXTRA% --kt %kt%"
if not "%ninter%"=="" set "EXTRA=%EXTRA% --ninter %ninter%"

echo outdir     : %outdir%
echo num_trials : %num_trials%
echo n_max      : %n_max%
echo threshold  : %threshold%
echo.

python "%CLI%" --scan --outdir="%outdir%" --num_trials="%num_trials%" --n_max="%n_max%" --threshold="%threshold%" --n_compress="%n_compress%"%EXTRA%

:end
endlocal
