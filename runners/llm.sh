#!/bin/bash

echo "Starting local LLM"
uvicorn llm_inference:app --host 0.0.0.0 --port 4321 --workers 1
