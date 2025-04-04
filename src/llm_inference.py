import os
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from local_llm import CustomLLM

load_dotenv()


class Query(BaseModel):
    chunk: List[float]


model = CustomLLM(
    model_name=os.getenv("LOCAL_LLM_MODEL"),
    torch_dtype=torch.float32 if os.getenv("TORCH_DTYPE") == "FLOAT32" else torch.bfloat16,  # noqa: E501,
    device_map=os.getenv("DEVICE"),
)


app = FastAPI()


@app.post("/inference/")
async def inference(query: Query) -> Dict[str, Any]:
    response = model.generate(query.chunk)
    return {"response": response}
