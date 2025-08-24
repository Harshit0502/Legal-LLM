# Legal-LLM

## Colab Environment Setup

Run the following script in Google Colab or a local notebook to install dependencies,
print hardware information, and initialize random seeds.

```python
!python setup_colab.py
```

The script installs required libraries, downloads the `en_core_web_sm` spaCy model if
necessary, displays CUDA/CPU details, defines a `set_seed` helper, and sets the default
seed to `42` for reproducibility.
