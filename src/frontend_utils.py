import os
import requests
import io
import librosa
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def get_transcription(
        *,
        uploaded_file,
        user_id: str,
        chunk_size_sec: int,
        target_dB: float,
        norm_algo: str,
        target_peak: float,
) -> str:
    model_sr = 16000
    file_bytes = io.BytesIO(uploaded_file.read())
    audio, original_sr = librosa.load(file_bytes, sr=None)
    chunk_length = chunk_size_sec * original_sr
    for i, chunk in enumerate([audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]):  # noqa: E501
        logger.info(f"chunk: {i}")
        requests.post(
            os.getenv("BACKEND_ENDPOINT"),
            json={"user_id": user_id, "chunk": chunk.tolist(), "sending": True},
            headers={"Content-Type": "application/json"},
        )
    logger.info(f"run backend processor for user: {user_id}")
    resp = requests.post(
        os.getenv("BACKEND_ENDPOINT"),
        json={
            "user_id": user_id,
            "chunk": [],
            "sending": False,
            "target_dB": target_dB,
            "norm_algo": norm_algo,
            "model_sr": model_sr,
            "original_sr": original_sr,
            "chunk_size_sec": chunk_size_sec,
            "target_peak": target_peak,
        },
        headers={"Content-Type": "application/json"},
    )
    return resp.json()["response"]