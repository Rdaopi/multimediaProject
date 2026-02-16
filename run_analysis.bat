@echo off
SET ENV_NAME=openvoice_env

echo [1/3] Attivazione ambiente Conda: %ENV_NAME%...
call conda activate %ENV_NAME%

if %errorlevel% neq 0 (
    echo ❌ Errore: Ambiente Conda '%ENV_NAME%' non trovato.
    pause
    exit /b
)

echo [2/3] Avvio della Pipeline (Estrazione + Analisi)...
python main.py

if %errorlevel% neq 0 (
    echo ❌ Si e verificato un errore durante l'esecuzione.
    pause
    exit /b
)

echo [3/3] Operazione completata con successo!
pause