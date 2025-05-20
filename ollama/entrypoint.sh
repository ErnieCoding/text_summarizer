#!/bin/bash
set -e

ollama serve &

sleep 5

ollama pull qwen2.5:14b
ollama pull qwen2.5:32b

wait