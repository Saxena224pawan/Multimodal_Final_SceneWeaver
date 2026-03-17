# Storytelling Dataset Plan

This branch is organized to collect datasets first, mapped to your case folders.

## Curated set (phase 1)

- `LLM_MODEL`
  - `RUCAIBox/Story-Generation`
  - `shawon/rocstories-combined`
  - `fabraz/writingPromptAug`

- `Caption_Gen`
  - `tonyhong/vwp`

- `Window_Gen`
  - `AlexZigma/msr-vtt`

- `Globa_Local_Emb_Feedback`
  - `friedrichor/ActivityNet_Captions`

- `Feedback_Caption`
  - `HuggingFaceM4/vatex`

- `VIDEO_GENERATIVE_BACKBONE`
  - `ViStoryBench/ViStoryBench`

## How to download

Dry-run selection:
```bash
python3 HF_PLUGIN/download_story_datasets.py --dry_run
```

Download all:
```bash
python3 HF_PLUGIN/download_story_datasets.py
```

Download one module only:
```bash
python3 HF_PLUGIN/download_story_datasets.py --module LLM_MODEL
```

Download specific datasets only:
```bash
python3 HF_PLUGIN/download_story_datasets.py --keys rocstories_combined msrvtt
```

## Notes

- Registry file: `datasets_registry.json`
- Destination root: `datasets/` (auto-created)
- You can edit/add datasets in the registry as we refine quality/size balance.
