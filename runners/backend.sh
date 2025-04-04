#!/bin/bash

echo "Starting backend..."
uvicorn backend:app --host 0.0.0.0 --port 8009 --workers 1
