@echo off

IF NOT EXIST venv (
    @REM Create a virtual environment
    python -m venv venv
    @REM Activate the virtual environment
    call venv\Scripts\activate
    @REM Install the requirements
    pip install -r requirements.txt
) ELSE (
    @REM Activate the virtual environment
    call venv\Scripts\activate
)

@REM Run the application in another terminal
@REM start "Flask App" cmd /k python -m src.app
start "Flask App" cmd /k flask run

echo Waiting for page to load...
timeout /t 5 >nul
echo Opening webpage...

@REM Open the application in the browser
start http://localhost:5000


