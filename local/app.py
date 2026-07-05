from fastapi import FastAPI
from pydantic import BaseModel
from local.infer import generate

app = FastAPI()


class Request(BaseModel):
    prompt: str
    task_type: str = "default"


@app.post("/local-llm")
def local_llm(req: Request):
    return generate(req.prompt, req.task_type)