from abc import ABC
from typing import List

from dotenv import load_dotenv
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
)

load_dotenv()


class BaseCustomLLM(ABC):
    ...


class CustomLLM(BaseCustomLLM):
    def __init__(
        self,
        model_name: str,
        torch_dtype,
        device_map: str = "cpu",
        sampling_rate: int = 16000,
        max_new_tokens: int = 440,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> None:

        super().__init__()
        self.sampling_rate = sampling_rate
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            # use_safetensors=True
        ).to(self.device_map)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def generate(self, chunk: List[float]) -> str:
        input_features = self.processor(
            chunk,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).input_features.to(self.device_map)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
        }

        predicted_ids = self.model.generate(input_features, **gen_kwargs)
        chunk_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()  # noqa: E501
        return chunk_text
