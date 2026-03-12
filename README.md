# SceneWeaver Story Pipeline

Active and supported runtime is the story pipeline script stack:
- `run_story_pipeline.sh`
- `scripts/run_story_pipeline.py`
- `director_llm/scene_director.py`
- `video_backbone/wan_backbone.py`
- `memory_module/embedding_memory.py`

## Install
```bash
pip install -r requirements.txt
```

## Quick Start (Cluster)
```bash
sbatch --partition=a40 --gres=gpu:a40:1 \
  --export=ALL,ENV_PATH=sceneweaver311,HF_HOME=$PWD/.hf,DRY_RUN=0,AUTO_FALLBACK_DRY_RUN=0 \
  run_story_pipeline.sh
```

## Architecture (Active Path)
1. Storyline is split into time windows by `SceneDirector`.
2. Each window prompt is refined with continuity context.
3. `WanBackbone` (or another diffusers T2V model id) generates frames.
4. Frames are encoded to per-window MP4 clips.
5. Optional memory embeddings (`clip` or `dinov2`) provide continuity feedback.
6. Optional captioner hook (BLIP-2/LLaVA) captions each generated clip to tighten environment anchors and detect duplicate subjects.

## Captioner Hook (optional)
- Download a captioner checkpoint (example already supported):  
  `huggingface-cli download Salesforce/blip2-flan-t5-xl --local-dir models/blip2-flan-t5-xl --local-dir-use-symlinks False`
- Enable in the pipeline: add `--captioner_model_id models/blip2-flan-t5-xl --captioner_device cpu` to `scripts/run_story_pipeline.py` arguments (or set `CAPTIONER_MODEL_ID` in the bash launcher).
- Captions are logged per window (`captions`, `caption_summary`, `caption_duplicates`) and feed the next window’s environment anchor and duplicate-aware negative prompt.

## Continuity Adapter Fine-Tuning
The project includes supervised continuity-adapter training for DINOv2 embeddings:
- `scripts/finetune_pororo_continuity.py`
- `scripts/finetune_flintstones_continuity.py`

Training objective and model:
- Uses bidirectional InfoNCE between anchor/positive transition frames.
- Backbone: DINOv2 (`AutoModel` + `AutoImageProcessor`).
- Head: MLP projector (`feature_dim -> hidden_dim -> proj_dim`), L2-normalized output embeddings.
- Optimizer: AdamW with separate LR for projector and (optional) backbone.

Dataset pairing:
- PororoSV: transition pair from each cached window (`frames[-2]` -> `frames[-1]`), with `seen`/`unseen` validation split control.
- FlintstonesSV: consecutive shot pairs per episode (`last frame of previous shot` -> `first frame of next shot`).

Run Pororo fine-tuning (cluster wrapper):
```bash
sbatch finetune_continuity.sh
```

Important wrapper defaults in `finetune_continuity.sh`:
- `EPOCHS=1000`, `BATCH_SIZE=64`, `TEMPERATURE=0.07`
- `LR_PROJECTOR=5e-5`, `LR_BACKBONE=1e-6`, `WEIGHT_DECAY=5e-2`
- `UNFREEZE_BACKBONE=0` (projector-only by default; set `1` to full finetuning)
- `VAL_SPLIT=unseen`

Run Flintstones fine-tuning (direct):
```bash
python scripts/finetune_flintstones_continuity.py \
  --dataset_root /home/vault/v123be/v123be36/FlintstonesSV \
  --dino_model_id /home/vault/v123be/v123be36/facebook/dinov2-base \
  --save_path outputs/flintstones_continuity_adapter.pt
```

Outputs:
- Best checkpoint (`*.pt`) is saved by lowest validation loss.
- Per-epoch metrics are written to `*.history.json`.
- Checkpoint contains: `projector_state_dict`, `args`, `dino_model_id`, `feature_dim`, and in-run metric history.

Use a trained adapter in the story pipeline:
- Set `EMBEDDING_ADAPTER_CKPT=/path/to/adapter.pt` in `run_story_pipeline.sh`, or pass `--embedding_adapter_ckpt` to `scripts/run_story_pipeline.py`.
- The launcher fails fast if the adapter checkpoint path is set but missing.

## Note on `src/driftguard`
`src/driftguard` is retained only as archived experimental code and is not a supported runtime path for current jobs.
