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

## Hosted AI Backend (AMD GPU Credits + 3 Gemma Models)

This repo now includes a backend pattern for hosting multiple Gemma models behind a single API.

Architecture:

- Frontend (Next.js/React) sends requests to `/chat`
- FastAPI backend applies routing + rate limits
- vLLM serves one of 3 Gemma models on AMD GPU

### 1. Configure model endpoints

Set environment variables (PowerShell):

```powershell
$env:VLLM_BASE_URL = "http://YOUR_AMD_GPU_HOST:8000"
$env:VLLM_API_KEY = ""
$env:GEMMA_SMALL_MODEL = "gemma-4-small"
$env:GEMMA_MEDIUM_MODEL = "gemma-4-medium"
$env:GEMMA_LARGE_MODEL = "gemma-4-large"
$env:RATE_LIMIT_PER_HOUR = "10"
```

### 2. Start backend API

```powershell
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### 3. Chat endpoint

```powershell
curl -X POST http://127.0.0.1:8080/chat `
	-H "Content-Type: application/json" `
	-H "x-user-id: demo-user" `
	-d '{"message":"Explain this algorithm","task_type":"code","max_tokens":256,"temperature":0.2}'
```

Response includes:

- `answer`
- `routed_model`
- `route_reason`
- `latency_ms`
- `usage`

### 4. Router behavior

- Short/simple requests route to small model
- Coding requests route to medium model
- Long/reasoning-heavy requests route to large model

Inspect routing decision:

```powershell
curl "http://127.0.0.1:8080/router/decision?message=prove%20this%20theorem&task_type=default"
```

### 5. Rate limiting and usage logs

- Default limit: 10 requests/hour per user (`x-user-id`, fallback to client IP)
- Usage logs written to `logs/usage.jsonl`
- Includes user id, model used, route reason, latency, and token usage

## Run Local LLM API

```powershell
.\venv\Scripts\python.exe -m uvicorn local.app:app --reload --port 8001
```

## Latency Optimization (<5-10s target)

The local endpoint now defaults to `speed_mode=true`, which:

- Applies prompt compression by task type
- Uses profile-based token limits and caps fast responses
- Runs generation in inference mode with cache enabled

Request example:

```powershell
curl -X POST http://127.0.0.1:8001/local-llm `
	-H "Content-Type: application/json" `
	-d '{"prompt":"Summarize AMD MI300 architecture.","task_type":"summary","speed_mode":true}'
```

## Token Efficiency Logs

Every inference writes structured metrics to `logs.jsonl` and is queryable via:

```powershell
curl "http://127.0.0.1:8001/logs/token-efficiency?limit=20"
```

Metrics include:

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `tokens_per_second`
- `ms_per_output_token`
- `prompt_compression_ratio` (local)

## Local vs Cloud Speed Comparison

Use benchmark endpoint:

```powershell
curl -X POST http://127.0.0.1:8001/benchmark/compare `
	-H "Content-Type: application/json" `
	-d '{"prompt":"Explain ROCm in 5 bullet points.","task_type":"summary"}'
```

Optional cloud env vars (OpenAI-compatible chat-completions endpoint):

```powershell
$env:CLOUD_LLM_ENDPOINT = "https://your-provider/v1/chat/completions"
$env:CLOUD_LLM_API_KEY = "your_api_key"
$env:CLOUD_LLM_MODEL = "gpt-4o-mini"
```

If cloud is not configured, comparison still returns local metrics and reports cloud as unavailable.

## TODO Status

- [x] Create local virtual environment
- [x] Install project dependencies
- [x] Add requirements.txt
- [x] Add .gitignore for virtual environments
- [x] Add application source code scaffold