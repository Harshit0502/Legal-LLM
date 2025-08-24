import os
import sys
import subprocess
import random
import platform

PACKAGES = [
    "transformers>=4.42",
    "datasets",
    "accelerate",
    "peft",
    "bitsandbytes",
    "sentencepiece",
    "evaluate",
    "rouge_score",
    "bert_score",
    "wandb",
    "pandas",
    "scikit-learn",
    "faiss-cpu",
    "nltk",
    "tiktoken",
    "unidecode",
    "spacy",
]

def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + PACKAGES)


def ensure_spacy_model(model: str = "en_core_web_sm"):
    try:
        import spacy
        spacy.load(model)
    except (ImportError, OSError):
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
        import spacy  # noqa: F401
        spacy.load(model)


def set_seed(seed: int = 42):
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_system_info():
    import torch

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print("CUDA is available")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU")
    print(f"CPU: {platform.processor()}")


if __name__ == "__main__":
    install_packages()
    ensure_spacy_model()
    print_system_info()
    set_seed(42)
    print("Random seeds set to 42")
