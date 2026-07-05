# AMD Hackathon Team 0-1

Python project bootstrap with local virtual environment and core ML/API dependencies.

## Setup

1. Create virtual environment:

```powershell
py -3 -m venv venv
```

2. Activate environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run once per terminal session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

3. Install dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Installed Dependencies

- torch==2.12.1
- transformers==5.13.0
- fastapi==0.139.0
- uvicorn==0.50.0
- pydantic==2.13.4

## Quick Check

```powershell
.\venv\Scripts\python.exe -m pip show torch transformers fastapi uvicorn pydantic
```

## Run API

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

## TODO Status

- [x] Create local virtual environment
- [x] Install project dependencies
- [x] Add requirements.txt
- [x] Add .gitignore for virtual environments
- [x] Add application source code scaffold