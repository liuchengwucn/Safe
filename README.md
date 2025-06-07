# Safe (ACL 2025 Main)

**TL;DR**: Formally verifying LLM mathematical reasoning using the Lean 4 formal language!

The official implementation of our paper **Safe** (Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification) and its associated datasets **FormalStep**. 

[Paper](https://www.arxiv.org/abs/2506.04592)
[Code](https://github.com/liuchengwucn/Safe)
[Dataset](https://huggingface.co/datasets/liuchengwu/FormalStep)

## Configuration Guide

To run this project, you need to locally deploy both a 7B reasoning model and a 7B prover using vLLM. We recommend using a server with at least 2 NVIDIA GPUs for optimal performance.

The Lean environment setup can be complex, so we **strongly recommend** using our provided Docker image to simplify configuration. For those interested in the manual setup process, please refer to our `Dockerfile` and `compose.yml`. Below is a step-by-step Docker-based configuration guide.

### Step 1: Clone the repository
```bash
git clone 'https://github.com/liuchengwucn/Safe.git' && cd Safe
```

### Step 2: Verify Docker and NVIDIA Container Toolkit installation
Ensure Docker with NVIDIA Container Toolkit and Docker Compose plugin are properly installed. Reference: [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker info | grep nvidia
```

Successful installation will display `nvidia` in Runtimes:
```
Runtimes: io.containerd.runc.v2 io.containerd.runtime.v1.linux nvidia runc
```

### Step 3: Configure environment variables
Create `.env` file with your preferred editor and set `OPENAI_API_KEY` and `OPENAI_BASE_URL` for GPT-4o/GPT-4o-mini API access.

```bash
OPENAI_API_KEY="sk-123456"
OPENAI_BASE_URL="https://api.openai.com/v1"
```

### Step 4 (Optional): Pre-download model weights
We **strongly recommend** pre-downloading model weights and mounting `~/.cache/huggingface` to the Docker container. Use the following Python code to cache models (Hugging Face CLI login may be required for gated models):

```python
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
# Reasoning Models
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-math-7b-instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Prover Model
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-Prover-V1.5-RL")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Prover-V1.5-RL")

# Reward Models
model = AutoModelForCausalLM.from_pretrained("peiyi9979/math-shepherd-mistral-7b-prm")
tokenizer = AutoTokenizer.from_pretrained("peiyi9979/math-shepherd-mistral-7b-prm")
# model = AutoModelForCausalLM.from_pretrained("RLHFlow/Llama3.1-8B-PRM-Deepseek-Data")
# tokenizer = AutoTokenizer.from_pretrained("RLHFlow/Llama3.1-8B-PRM-Deepseek-Data")
# model = AutoModelForSequenceClassification.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1")
# tokenizer = AutoTokenizer.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1")
# model = AutoModelForSequenceClassification.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
# tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
```

For online model loading (not recommended), modify `compose.yml` to allow downloads:

```yaml
services:
  deepseek-prover:
    ...
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
      # Comment out for online download
      # - HF_HUB_OFFLINE=1
      
  reasoning-model:
    ...
    environment:
      - NVIDIA_VISIBLE_DEVICES=2
      # Comment out for online download
      # - HF_HUB_OFFLINE=1
    ...
```

### Step 5: Download datasets
Required datasets: Math, GSM8K, PRM800K (MATH-500), and CollegeMath. Math/GSM8K/CollegeMath are included; use `download.py` for PRM800K:

```bash
python download.py
```

### Step 6: GPU allocation
Specify GPU devices for each service in `compose.yml`. Empirical requirement: ~40GB VRAM per service (prover & reasoning model).

### Step 7 (Optional): Pre-pull Docker images
Our images include pre-configured Python/Lean environments and cached mathlib. While `docker compose up` auto-pulls these, manual pulling is available:

```bash
docker pull vllm/vllm-openai:v0.8.5.post1
docker pull ghcr.io/liuchengwucn/safe:1.0.0
```

### Step 8: Launch containers & Enter development environment
Mathlib cache will be automatically symlinked to the project directory.

```bash
docker compose up -d
docker compose exec safe bash
```

## Paper Reproduction

To reproduce our experiments, execute these four scripts in order. You can modify the configuration at the beginning of each script, including the reasoning model and prover to be used, as well as the dataset and hyperparameters, among other settings.
1. `collect_trace.py` - Collects reasoning traces (train traces for aggregator training, test traces for evaluation)
2. `aggregator.py` - Trains the LSTM aggregator
3. `benchmark.py` - Evaluates SAFE framework performance using test traces
4. `benchmark_ensemble.py` - Tests hyperparameter alpha's impact on framework performance

## Citation
If you find our work useful, please consider citing our paper.

```
@misc{liu2025safe,
      title={Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification}, 
      author={Chengwu Liu and Ye Yuan and Yichun Yin and Yan Xu and Xin Xu and Zaoyu Chen and Yasheng Wang and Lifeng Shang and Qun Liu and Ming Zhang},
      year={2025},
      eprint={2506.04592},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.04592}, 
}
```

