from fastapi import FastAPI
from pydantic import BaseModel
from local.infer import compare_local_vs_cloud, generate, get_recent_efficiency_logs

app = FastAPI()


class Request(BaseModel):
    prompt: str
    task_type: str = "default"
    speed_mode: bool = True


@app.post("/local-llm")
def local_llm(req: Request):
    return generate(req.prompt, req.task_type, req.speed_mode)


@app.post("/benchmark/compare")
def benchmark_compare(req: Request):
    return compare_local_vs_cloud(req.prompt, req.task_type)


@app.get("/logs/token-efficiency")
def token_efficiency_logs(limit: int = 20):
    return {"events": get_recent_efficiency_logs(limit=limit)}