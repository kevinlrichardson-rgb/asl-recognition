# ASL application development environment

A ready-to-use ASL application development environment for VS Code. Includes **LangChain**, **LlamaIndex**, **Hugging Face Transformers**, **vLLM** (GPU only), and API clients for OpenAI and Anthropic. Available in both GPU and CPU-only configurations.

## What's included

### GPU Environment

| Category | Details |
|----------|---------|
| **Base Image** | NVIDIA PyTorch 24.08 |
| **GPU** | CUDA 12.6, PyTorch 2.4 (GPU-enabled) |
| **Python** | 3.10 |
| **LLM Frameworks** | LangChain, LlamaIndex, Transformers, smolagents, vLLM |
| **API Clients** | OpenAI, Anthropic, Ollama |
| **Vector Store** | ChromaDB, sentence-transformers |
| **Tools** | Gradio, accelerate, datasets, tiktoken |

### CPU Environment

| Category | Details |
|----------|---------|
| **Base Image** | Python 3.11-slim |
| **Python** | 3.11, PyTorch (CPU) |
| **LLM Frameworks** | LangChain, LlamaIndex, Transformers, smolagents |
| **API Clients** | OpenAI, Anthropic, Ollama |
| **Vector Store** | ChromaDB, sentence-transformers |
| **Tools** | Gradio, accelerate, datasets, tiktoken |


## Project structure

```
llms-devcontainer/
├── .devcontainer/
│   ├── gpu/
│   │   └── devcontainer.json   # GPU dev container configuration
│   └── cpu/
│       └── devcontainer.json   # CPU dev container configuration
├── data/                       # Store datasets here
├── logs/                       # Training/experiment logs
├── models/                     # Saved model files
├── src/
│   └── environment_test.py     # Verify your setup
├── .gitignore
├── LICENSE
└── README.md
```

## Requirements

### GPU Environment
- **NVIDIA GPU** (Pascal or newer) with driver ≥545
- **Docker** with GPU support ([Windows](https://docs.docker.com/desktop/setup/install/windows-install) | [Linux](https://docs.docker.com/desktop/setup/install/linux))
- **VS Code** with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

> **Linux users:** Also install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### CPU Environment
- **Docker** ([Windows](https://docs.docker.com/desktop/setup/install/windows-install) | [Linux](https://docs.docker.com/desktop/setup/install/linux))
- **VS Code** with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### GPU compatibility

The GPU environment requires an NVIDIA GPU with **compute capability 6.0+** (Pascal architecture or newer):

| Architecture | Example GPUs | Compute Capability |
|--------------|--------------|-------------------|
| Pascal | GTX 1050–1080, Tesla P100 | 6.0–6.1 |
| Volta | Tesla V100, Titan V | 7.0 |
| Turing | RTX 2060–2080, GTX 1660 | 7.5 |
| Ampere | RTX 3060–3090, A100 | 8.0–8.6 |
| Ada Lovelace | RTX 4060–4090 | 8.9 |
| Hopper | H100, H200 | 9.0 |
| Blackwell | RTX 5070–5090, B100, B200 | 10.0 |

Check your GPU's compute capability: [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)

## Quick start

1. **Fork** this repository (click "Fork" button above)

2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/asl-recognition.git
   ```

3. **Open VS Code**

4. **Open Folder in Container** from the VS Code command palette (Ctrl+Shift+P), start typing `Open Folder in`...
   - Select the **GPU** or **CPU** configuration when prompted

5. **Verify** by running `python src/environment_test.py`

## Using as a template for new projects

You can use your fork as a template to quickly create new LLM application projects:

### One-time setup: Make your fork a template

1. Go to your fork on GitHub
2. Click **Settings** → scroll to **Template repository**
3. Check the box to enable it

### Creating a new project from your template

1. Go to your fork on GitHub
2. Click the green **Use this template** button → **Create a new repository**
3. Enter your new repository name and settings
4. Click **Create repository**
5. **Clone** your new repository:
   ```bash
   git clone https://github.com/<your-username>/my-new-project.git
   ```

Now you have a fresh LLM application project with the dev container configuration ready to go!

## Adding Python packages

### Using pip directly

Install packages in the container terminal:

```bash
pip install <package-name>
```

> **Note:** Packages installed this way will be lost when the container is rebuilt.

### Using requirements.txt (Recommended)

For persistent packages that survive container rebuilds:

1. **Create** a `requirements.txt` file in the repository root:
   ```
   langchain-community
   faiss-cpu
   ```

2. **Update** the appropriate `.devcontainer/*/devcontainer.json` to install packages on container creation by adding a `postCreateCommand`:
   ```json
   "postCreateCommand": "pip install -r requirements.txt"
   ```

3. **Rebuild** the container (`F1` → "Dev Containers: Rebuild Container")

Now your packages will be automatically installed whenever the container is created.

## Gradio Web UI

Both environments include Gradio for building interactive demos. To run a Gradio app:

```python
import gradio as gr

def greet(name):
    return f"Hello, {name}!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch(server_name="0.0.0.0")
```

The default Gradio port (7860) is accessible from your host machine.

## Keeping your fork updated

```bash
# Add upstream (once)
git remote add upstream https://github.com/gperdrizet/llms-devcontainer.git

# Sync
git fetch upstream
git merge upstream/main
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Docker won't start | Enable virtualization in BIOS |
| Permission denied (Linux) | Add user to docker group, then log out/in |
| GPU not detected | Update NVIDIA drivers (≥545), install NVIDIA Container Toolkit |
| Container build fails | Check internet connection |
| vLLM not available | vLLM requires GPU; use the GPU devcontainer |
| Module not found | Rebuild container after adding to requirements.txt |

# Dataset Acknowledgements
@misc{https://www.kaggle.com/grassknoted/aslalphabet_akash nagaraj_2018,
title={ASL Alphabet},
url={https://www.kaggle.com/dsv/29550},
DOI={10.34740/KAGGLE/DSV/29550},

https://data.mendeley.com/datasets/8fmvr9m98w/2

ASL_Dynamic.zip
SignAlphaSet.zip

The files associated with this dataset are licensed under a Creative Commons Attribution 4.0 International licence.
