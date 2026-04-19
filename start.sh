#!/bin/bash
echo "Starting LexSearch..."
cd ~/lexsearch/backend
export GROQ_API_KEY=$(grep GROQ_API_KEY ~/.zshrc | cut -d'=' -f2)
uvicorn app.main:app --reload --port 8000
