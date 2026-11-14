@echo off
REM  run_all.bat - runs all scripts as a batch
setlocal
set "C=%~dp0"
set "EX=%C%04_extract\extract_scene_table.py"
set "GL=--source acolite_aoi2_glint --tag aoi2_glint --masks-from aoi2_noglint"

echo. & echo  [1/11] masks (rectangle AOI)
python "%EX%" masks || goto :fail
echo. & echo  [2/11] extract - no glint correction
python "%EX%" extract || goto :fail
echo. & echo  [3/11] extract - glint corrected
python "%EX%" extract %GL% || goto :fail
echo. & echo  [4/11] cloud cover per scene (L1C cloud mask)
python "%C%06_ancillary\cloud_fraction.py" || goto :fail
echo. & echo  [5/11] scene table - no glint
python "%EX%" summary || goto :fail
echo. & echo  [6/11] scene table - glint
python "%EX%" summary %GL% || goto :fail
echo. & echo  [7/11] RQ1 seasonal cycle
python "%C%05_analysis\rq1_seasonal.py" || goto :fail
echo. & echo  [8/11] RQ2 + RQ3 confounds
python "%C%05_analysis\rq2_rq3.py" || goto :fail
echo. & echo  [9/11] RQ4 depth (ATL24 photons + per-scene extraction)
python "%C%05_analysis\rq4_depth.py" photons || goto :fail
python "%C%05_analysis\rq4_depth.py" extract || goto :fail
echo. & echo  [10/11] RQ4 analysis
python "%C%05_analysis\rq4_depth.py" analyse || goto :fail
echo. & echo  [11/11] Figures and tables
python "%C%07_figures\make_figures.py" || goto :fail
python "%C%07_figures\make_fig1_study_area.py" || goto :fail

echo. & echo DONE
goto :eof

:fail
echo. & echo STOPPED: the step above failed.
exit /b 1