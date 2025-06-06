FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Update the package list and install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    psmisc

# Install elan and Lean 4
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- --default-toolchain v4.9.0-rc1 -y
ENV PATH=/root/.elan/bin:$PATH
RUN elan toolchain install v4.9.0-rc1

# Install Mathlib for DeepSeek-Prover-V1.5
RUN git clone https://github.com/xinhjBrant/mathlib4.git /workspace/mathlib4
WORKDIR /workspace/mathlib4
RUN lake build
RUN lake build repl
WORKDIR /workspace

# Install COPRA
RUN git clone --recurse-submodules https://github.com/liuchengwucn/copra.git /workspace/copra
WORKDIR /workspace/copra/src/tools/repl
RUN lake build repl
WORKDIR /workspace/copra/data/test/lean4_proj
RUN lake build
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

LABEL org.opencontainers.image.source=https://github.com/liuchengwucn/Safe
LABEL org.opencontainers.image.description="Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification - Official Implementation & Dataset"