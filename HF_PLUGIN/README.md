# HF Plugin

Reusable Hugging Face helper for this branch.

## What it provides
- Token login (`HFPlugin.login`)
- Model download (`HFPlugin.download_model`)
- Dataset download (`HFPlugin.download_dataset`)

## Minimal usage
```python
from HF_PLUGIN import HFPlugin

hf = HFPlugin()
# hf.login(token="hf_xxx")  # optional if already authenticated
hf.download_model(
    "Qwen/Qwen2.5-3B-Instruct",
    local_dir="./models/Qwen2.5-3B-Instruct",
)
```
