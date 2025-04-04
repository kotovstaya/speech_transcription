import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel


class Input(BaseModel):
    chunk: List[float]


load_dotenv()

app = FastAPI()


@app.post("/get_answer/")
async def inference(inp_obj: Input) -> Dict[str, Any]:
    resp = requests.post(
        os.getenv("LLM_ENDPOINT"),
        json={"chunk": inp_obj.chunk},
        headers={"Content-Type": "application/json"},
    )
    return {
        "chunk": inp_obj.chunk,
        "response": resp.json()["response"],
        "status": resp.status_code,
    }
