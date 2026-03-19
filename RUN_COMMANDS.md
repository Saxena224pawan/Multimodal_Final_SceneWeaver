# Run Commands

This file lists the standard commands used to run the main entrypoints in this repository.

## Environment
Use the shared runtime unless a launcher says otherwise:

```bash
source /apps/python/3.12-conda/etc/profile.d/conda.sh
conda activate sceneweaver_runtime
```

## Standard Patterns
- Python CLIs: `python3 <script>.py ...`
- Local shell launchers: `bash <launcher>.sh`
- SLURM launchers: `sbatch <launcher>.sh`

## Main Story Pipelines

### Simple local story run
```bash
bash run_story_simple.sh
```

### Main thirsty-crow pipeline
Recommended on cluster:
```bash
sbatch run_story_pipeline.sh
```
Dry run locally:
```bash
DRY_RUN=1 bash run_story_pipeline.sh
```

### Stateful pipeline
```bash
sbatch run_story_pipeline_stateful.sh
```
Python entrypoint:
```bash
python3 scripts/run_story_pipeline_stateful.py --storyline "..." --output_dir outputs/story_run_stateful_test
```

### Stateful Wan I2V pipeline
```bash
sbatch run_story_pipeline_stateful_wan_i2v.sh
```
Python entrypoint:
```bash
python3 scripts/run_story_pipeline_stateful_wan_i2v.py --storyline "..." --output_dir outputs/story_run_stateful_wan_i2v_test
```

### Identity-layout pipeline
```bash
sbatch run_story_pipeline_identity_layout.sh
```
Python entrypoint:
```bash
python3 scripts/run_story_pipeline_identity_layout.py --storyline "..." --output_dir outputs/story_run_identity_layout_test
```

### DINOv2 tracking launcher
```bash
bash run_story_pipeline_dinov2_tracking.sh
```

### Storyboard-to-I2V launcher
```bash
bash run_story_i2v_from_storyboard.sh
```

## Core Python Pipeline CLI

### Full pipeline directly
```bash
python3 scripts/run_story_pipeline.py   --storyline "A thirsty crow finds a pot with little water and raises the water level with stones."   --output_dir outputs/story_run_manual   --window_plan_json configs/window_plans/thirsty_crow_story.json   --reference_conditioning   --initial_condition_image thirsty_crow_start_image.png
```

## Model Registry and Setup

### Validate models registry
```bash
python3 scripts/00_pipeline/00_validate_models_registry.py
```

### Link models into outputs/pipeline/model_links.json
```bash
python3 scripts/00_pipeline/01_link_models.py
```

### Show resolved pipeline models
```bash
python3 scripts/00_pipeline/02_show_pipeline_models.py
```

### Validate datasets registry
```bash
python3 scripts/01_datasets/00_validate_registry.py
```

## Story Planning

### Story to scene plan
```bash
python3 scripts/02_story/00_story_to_scene_plan.py   --storyline "A thirsty crow finds a pot with little water and raises the water level with stones."   --output outputs/story/scene_plan.json
```

### Storyline to StorySpec
```bash
python3 scripts/02_story/01_storyline_to_story_spec.py
```

### Storyline to scene-plan windows/prompts
```bash
python3 scripts/02_story/02_storyline_to_scene_plan.py   --storyline "A thirsty crow finds a pot with little water and raises the water level with stones."   --window-seconds 8   --scene-plan-output outputs/story/scene_plan.generated.json
```

## Caption Generation

### Expected captions
```bash
python3 scripts/03_caption/00_build_expected_video_captions.py
```

### Dense expected captions
```bash
python3 scripts/03_caption/01_build_dense_expected_video_captions.py
```

## Video Window Utilities

### Prepare video output layout
```bash
python3 scripts/04_video/00_prepare_video_outputs.py
```

### Generate window placeholders/metadata
```bash
python3 scripts/04_video/01_generate_video_windows.py
```

### Generate a window with Diffusers
```bash
python3 scripts/04_video/02_generate_window_with_diffusers.py
```

### Extract window anchor frames
```bash
python3 scripts/04_video/03_extract_window_anchor_frames.py
```

## Embedding and Repair

### Global/local embedding feedback
```bash
python3 scripts/05_embedding/00_global_local_embedding_feedback.py
```

### Repair failed or weak windows
```bash
python3 scripts/08_repair_windows.py --run_dir outputs/story_run_example
```

## Evaluation

### Install VBench runtime on cluster
```bash
sbatch run_install_vbench.sh
```

### VBench continuity evaluation
Local:
```bash
python3 scripts/09_eval_vbench_continuity.py --dry_run
python3 scripts/09_eval_vbench_continuity.py --videos_path outputs/story_run_example/clips
```
Cluster:
```bash
sbatch run_vbench_continuity.sh
```

### Video-Bench window-prompt evaluation
Local:
```bash
python3 scripts/10_eval_videobench_window_prompt.py --dry_run
python3 scripts/10_eval_videobench_window_prompt.py   --videos_path outputs/story_run_example   --config_path /path/to/videobench_config.json
```
Cluster:
```bash
sbatch run_videobench_window_prompt.sh
```

## Training and Finetuning

### Continuity finetuning shell wrapper
```bash
bash finetune_continuity.sh
```

### SLURM finetuning launcher
```bash
sbatch slurm_finetune_i2v.sh
```

### Python finetuning scripts
```bash
python3 scripts/finetune_i2v_model.py
python3 scripts/finetune_pororo_continuity.py
python3 scripts/finetune_flintstones_continuity.py
```

## Dataset Download

### Download I2V datasets
```bash
python3 scripts/download_i2v_datasets.py
```

### Download story datasets through HF helper
```bash
python3 HF_PLUGIN/download_story_datasets.py
```

## Tests

### Unit tests
```bash
pytest tests
```

### Targeted tests
```bash
python3 test_generation.py
python3 test_noise_conditioning.py
```

## Notes
- Files such as `__init__.py`, module implementations under `director_llm/`, `memory_module/`, `video_backbone/`, and helper utilities are imported by other entrypoints and are not normally run directly.
- For launchers with embedded `#SBATCH` headers, `sbatch <file>.sh` is the standard cluster command.
- If you want, this can be expanded into a stricter per-file matrix with required env vars, inputs, and example outputs.
