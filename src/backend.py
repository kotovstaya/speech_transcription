import os
from typing import Any, Dict, List
import requests
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from backend_utils import get_resampled_and_normalized_audio
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class Input(BaseModel):
    user_id: str
    chunk: List[float]
    sending: bool = False
    target_dB: float = -10
    norm_algo: str = "rms"
    model_sr: int = 16000
    chunk_size_sec: int = 30
    target_peak: float = 0.95
    original_sr: int = 16000


USER_2_CHUNKS = {}

load_dotenv()

app = FastAPI()


@app.post("/get_answer/")
async def inference(inp_obj: Input) -> Dict[str, Any]:
    global USER_2_CHUNKS
    response = ""
    if inp_obj.sending:
        USER_2_CHUNKS.setdefault(inp_obj.user_id, []).append(inp_obj.chunk)
    else:
        audio = np.hstack(USER_2_CHUNKS[inp_obj.user_id])
        logger.info(f"inp_obj: {inp_obj}")
        audio = get_resampled_and_normalized_audio(
            audio, inp_obj.norm_algo, inp_obj.target_peak, inp_obj.target_dB, inp_obj.model_sr, inp_obj.original_sr,  # noqa: E501
        )
        response = ""
        chunk_length = inp_obj.chunk_size_sec * inp_obj.model_sr
        for i, chunk in enumerate([audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]):  # noqa: E501
            logger.info(f"chunk: {i}")
            resp = requests.post(
                os.getenv("LLM_ENDPOINT"),
                json={"chunk": chunk.tolist()},
                headers={"Content-Type": "application/json"},
            )
            response += resp.json()["response"]
        USER_2_CHUNKS.pop(inp_obj.user_id)
        logger.info(f"return response for {inp_obj.user_id} : {response}")
    return {
        "response": response,
        "status": 200,
    }
