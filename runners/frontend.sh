#!/bin/bash

echo "Starting frontend..."
streamlit run frontend.py --server.port 8010 --server.address 0.0.0.0
