FROM python:3.12-slim

# System deps (tmux required by OpenHands TerminalTool)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tmux git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# PyTorch (CPU — GPU not needed for tabular ML)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# ML libraries
COPY requirements.v4.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# OpenHands SDK
RUN pip install --no-cache-dir openhands-ai==1.6.0

# Agent runner + prompts (baked into image)
COPY agent_runner.py /app/agent_runner.py
COPY prompts_dir_v4/ /app/prompts/

# Workspace and data dirs
RUN mkdir -p /workspace /data && chmod 777 /workspace

WORKDIR /workspace
CMD ["sleep", "infinity"]
