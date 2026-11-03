@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"

set "CLI=%SCRIPT_DIR%\run_treax_expedition.py"

rem stages: search (default), search+committor, or explore (one MCTSS run)
if "%stages%"=="" set "stages=search"
if "%nworkers%"=="" set "nworkers=4"
if "%tau%"=="" set "tau=1."
if "%steps%"=="" set "steps=10"
if "%eq%"=="" set "eq=0"
if "%nprocs%"=="" set "nprocs=1"
rem method is left unset on purpose: the CLI then picks the default of whichever
rem back end is in use, gfn2-xtb on xtb and hf-3c on orca
if "%accuracy%"=="" set "accuracy=1."
if "%etemp%"=="" set "etemp=300."
rem t12 is left unset on purpose: 10 for the explore stage, 3 for the search
rem charge and mult are left unset on purpose: the CLI then takes them from the
rem reaction's meta.json, and falls back to 0 and 1 when there is none
if "%threshold%"=="" set "threshold=1."
rem calculator is left unset on purpose: the search takes it from meta.json,
rem the explore stage falls back to xtb
if "%step_scale%"=="" set "step_scale=1."

rem Input: an .xyz, a folder with reactant.xyz and product.xyz, or a reaction
rem name resolved under reaction\input. Default is the sn2 reaction shipped here.
if "%file%"=="" set "file=sn2"

rem xtb is threaded with OpenMP, ORCA runs separate MPI processes, and the
rem search runs worker processes of its own. Giving both the same count at once
rem would oversubscribe the machine.
set "OMP_NUM_THREADS=1"
if "%stages%"=="explore" if "%calculator%"=="xtb" set "OMP_NUM_THREADS=%nprocs%"
if "%stages%"=="explore" if "%calculator%"=="" set "OMP_NUM_THREADS=%nprocs%"

rem Optional flags: the ORCA back end, moving every atom, the GFN2-xTB screen
set "EXTRA="
if not "%method%"=="" set "EXTRA=%EXTRA% --method=%method%"
if not "%orcapath%"=="" set "EXTRA=%EXTRA% --orcapath=%orcapath%"
if not "%basis%"=="" set "EXTRA=%EXTRA% --basis=%basis%"
if not "%state%"=="" set "EXTRA=%EXTRA% --state=%state%"
if not "%seed%"=="" set "EXTRA=%EXTRA% --seed=%seed%"
if not "%charge%"=="" set "EXTRA=%EXTRA% --charge=%charge%"
if not "%mult%"=="" set "EXTRA=%EXTRA% --mult=%mult%"
if not "%outdir%"=="" set "EXTRA=%EXTRA% --outdir=%outdir%"
if "%all%"=="1" set "EXTRA=%EXTRA% --all"
if "%surrogate_xtb%"=="1" set "EXTRA=%EXTRA% --surrogate_xtb"
if not "%calculator%"=="" set "EXTRA=%EXTRA% --calculator=%calculator%"
if not "%t12%"=="" set "EXTRA=%EXTRA% --t12=%t12%"
rem search options
if "%screen%"=="1" set "EXTRA=%EXTRA% --screen"
if not "%seeds_per_rung%"=="" set "EXTRA=%EXTRA% --seeds_per_rung=%seeds_per_rung%"
if not "%explore_steps%"=="" set "EXTRA=%EXTRA% --explore_steps=%explore_steps%"
if not "%max_chains%"=="" set "EXTRA=%EXTRA% --max_chains=%max_chains%"
if not "%crossing_votes%"=="" set "EXTRA=%EXTRA% --crossing_votes=%crossing_votes%"
if not "%committor_iterations%"=="" set "EXTRA=%EXTRA% --committor_iterations=%committor_iterations%"
if "%orcapath%"=="" if not "%TREAX_ORCA%"=="" set "EXTRA=%EXTRA% --orcapath=%TREAX_ORCA%"

echo file      : %file%
echo stages    : %stages%
echo back end  : %calculator%
echo method    : %method% %basis%
echo tau       : %tau%
echo steps     : %steps%
echo step scale: %step_scale%
echo.

python "%CLI%" --stages="%stages%" --nworkers="%nworkers%" --file="%file%" --tau="%tau%" --steps="%steps%" --eq="%eq%" --nprocs="%nprocs%" --threshold="%threshold%" --step_scale="%step_scale%" --accuracy="%accuracy%" --etemp="%etemp%"%EXTRA%

endlocal
