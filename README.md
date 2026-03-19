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

## StorySpec Authoring (Characters + Objects + Beats)
Use this path when you want story-specific entities and tighter prompt control:

1) Build StorySpec from a raw storyline (LLM-assisted, heuristic fallback):
```bash
python scripts/02_story/01_storyline_to_story_spec.py \
  --storyline "A courier races across a flooded station to deliver a final letter before the last train departs." \
  --runtime-seconds 48 \
  --output outputs/story/story_spec.generated.json
```

2) Convert StorySpec into window-level prompts:
```bash
python scripts/02_story/00_story_to_scene_plan.py \
  --story-spec outputs/story/story_spec.generated.json \
  --output outputs/story/scene_plan.json \
  --window-seconds 4
```

Or run both steps in one command:
```bash
python scripts/02_story/02_storyline_to_scene_plan.py \
  --storyline "A courier races across a flooded station to deliver a final letter before the last train departs." \
  --runtime-seconds 48 \
  --scene-plan-output outputs/story/scene_plan.generated.json
```

`StorySpec` now supports:
- root `objects`: per-story prop/set definitions with stable IDs
- beat `must_include_objects`: explicit object continuity per beat

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

## VBench Continuity Evaluation
Run continuity-focused VBench metrics on generated videos with:
- `scripts/09_eval_vbench_continuity.py`

Install (example):
```bash
python3 -m pip install vbench
```

Install on cluster with SLURM:
```bash
sbatch --partition=a40 --gres=gpu:a40:1 \
  --export=ALL,ENV_PATH=sceneweaver_runtime,UPGRADE_PIP=1 \
  run_install_vbench.sh
```

Dry run (prints planned commands):
```bash
python3 scripts/09_eval_vbench_continuity.py --dry_run
```

Evaluate latest story run (`outputs/story_run_*/clips`):
```bash
python3 scripts/09_eval_vbench_continuity.py
```
Default is `--sequence_mode concat_windows`, which concatenates `window_*.mp4` into a single sequential video before scoring.

Evaluate a specific folder:
```bash
python3 scripts/09_eval_vbench_continuity.py \
  --videos_path outputs/story_run_20260301_043739/clips \
  --dimensions subject_consistency,background_consistency,motion_smoothness,temporal_flickering
```

Per-clip mode (no concatenation):
```bash
python3 scripts/09_eval_vbench_continuity.py \
  --videos_path outputs/story_run_20260301_043739/clips \
  --sequence_mode per_clip
```

Chunked evaluation (score continuity on 8-second chunks after optional concatenation):
```bash
python3 scripts/09_eval_vbench_continuity.py \
  --videos_path outputs/story_run_20260301_043739/clips \
  --sequence_mode concat_windows \
  --chunk_seconds 8
```

Chunked evaluation by frame count (for example, 96-frame units):
```bash
python3 scripts/09_eval_vbench_continuity.py \
  --videos_path outputs/story_run_20260301_043739/clips \
  --sequence_mode concat_windows \
  --chunk_frames 96
```

Run on cluster with SLURM (copied runtime style from `run_story_pipeline.sh`):
```bash
sbatch --partition=a40 --gres=gpu:a40:1 \
  --export=ALL,ENV_PATH=sceneweaver_runtime,DRY_RUN=0,MODE=custom_input,SEQUENCE_MODE=concat_windows,DIMENSIONS=subject_consistency,background_consistency,motion_smoothness,temporal_flickering \
  run_vbench_continuity.sh
```

To evaluate a specific run, add `VIDEOS_PATH`:
```bash
sbatch --partition=a40 --gres=gpu:a40:1 \
  --export=ALL,ENV_PATH=sceneweaver_runtime,VIDEOS_PATH=$PWD/outputs/story_run_20260301_043739/clips \
  run_vbench_continuity.sh
```

Outputs:
- Command logs and artifacts are written under `outputs/reports/vbench_continuity/<run_name>/`.
- `summary.json` records per-dimension status, command, and log paths.
- `interpretation_report.json` and `interpretation_report.md` provide score summary and interpretation notes.

## Video-Bench Window Prompt Evaluation
Run prompt-conditioned per-window evaluation with:
- `scripts/10_eval_videobench_window_prompt.py`

This benchmark prepares your `window_*.mp4` clips plus the corresponding prompt text from `run_log.jsonl` and runs Video-Bench custom mode one dimension at a time. Prompt source precedence for `--prompt_source auto` is:
- `generation_prompt`
- `refined_prompt`
- `prompt_seed`
- `beat`

Install (example):
```bash
python3 -m pip install videobench
```

Dry run on the latest story run:
```bash
python3 scripts/10_eval_videobench_window_prompt.py --dry_run
```

Evaluate a specific story run:
```bash
python3 scripts/10_eval_videobench_window_prompt.py \
  --videos_path outputs/story_run_stateful_20260313_205126 \
  --config_path /path/to/videobench_config.json
```

Use a narrower prompt source:
```bash
python3 scripts/10_eval_videobench_window_prompt.py \
  --videos_path outputs/story_run_stateful_20260313_205126 \
  --config_path /path/to/videobench_config.json \
  --prompt_source beat \
  --dimensions "video-text consistency,action,scene"
```

Run on cluster with SLURM:
```bash
sbatch --partition=a40 --gres=gpu:a40:1 \
  --export=ALL,ENV_PATH=sceneweaver_runtime,VIDEOBENCH_CONFIG_PATH=/path/to/videobench_config.json \
  run_videobench_window_prompt.sh
```

To evaluate a specific run, add `VIDEOS_PATH`:
```bash
sbatch --partition=a40 --gres=gpu:a40:1 \
  --export=ALL,ENV_PATH=sceneweaver_runtime,VIDEOBENCH_CONFIG_PATH=/path/to/videobench_config.json,VIDEOS_PATH=$PWD/outputs/story_run_stateful_20260313_205126 \
  run_videobench_window_prompt.sh
```

Outputs:
- Prepared Video-Bench input, prompt map, logs, and reports are written under `outputs/reports/videobench_window_prompt/<run_name>/`.
- `summary.json` records prompt-source resolution, clip count, commands, and per-dimension status.
- `interpretation_report.json` and `interpretation_report.md` summarize prompt samples and benchmark execution.

## Note on `src/driftguard`
`src/driftguard` is retained only as archived experimental code and is not a supported runtime path for current jobs.
