# Core vs Agent Benchmark Comparison

## Benchmark Audit

- Scope: 4 core story runs and 4 agent-based story runs, each checked for 5 Video-Bench prompt metrics and 4 VBench continuity metrics.
- Coverage: all 8 runs now have complete prompt-metric and continuity reports. The comparison below uses the latest full combined summaries with both phases enabled; for all four agent stories this means the `parserfix_rerun` outputs from March 22, 2026.
- Output basis: comparison is derived from the latest `benchmark_reports/combined/*/summary.json`, then expanded using the linked per-phase Video-Bench and continuity summaries.

### Aggregate Means

| Metric | Core Mean | Agents Mean | Delta (Agents-Core) |
|---|---:|---:|---:|
| video-text consistency | 1.6667 | 0.3333 | -1.3333 |
| action | 1.3333 | 1.9167 | 0.5833 |
| scene | 1.7083 | 1.3333 | -0.3750 |
| object_class | 1.7500 | 2.0000 | 0.2500 |
| color | 1.3333 | 2.0417 | 0.7083 |
| subject_consistency | 0.6488 | 0.5611 | -0.0878 |
| background_consistency | 0.8385 | 0.7918 | -0.0467 |
| motion_smoothness | 0.9856 | 0.9719 | -0.0136 |
| temporal_flickering | 0.9820 | 0.9602 | -0.0218 |
| Prompt Metrics Average (5 dims) | 1.5583 | 1.5250 | -0.0333 |
| Continuity Average (4 dims) | 0.8637 | 0.8212 | -0.0425 |

### Story-wise Comparison

#### fox_and_grapes

- Core benchmark: `/home/vault/v123be/v123be36/sceneweaver_runs/core_fox_and_grapes_260319_225547/benchmark_reports/combined/combined_benchmark_core_fox_and_grapes_260319_225547/summary.json`
- Agents benchmark: `/home/vault/v123be/v123be36/sceneweaver_runs/agents_fox_grapes_agents_a100_260321_220554/benchmark_reports/combined/combined_benchmark_agents_fox_grapes_agents_a100_260321_220554_parserfix_rerun_20260322/summary.json`
- Prompt average: core `1.1333` vs agents `1.6000`
- Continuity average: core `0.8820` vs agents `0.8223`

| Metric | Core | Agents | Delta |
|---|---:|---:|---:|
| video-text consistency | 1.1667 | 0.0000 | -1.1667 |
| action | 1.3333 | 2.0000 | 0.6667 |
| scene | 0.8333 | 0.5000 | -0.3333 |
| object_class | 1.3333 | 2.6667 | 1.3333 |
| color | 1.0000 | 2.8333 | 1.8333 |
| subject_consistency | 0.7115 | 0.5524 | -0.1591 |
| background_consistency | 0.8617 | 0.7804 | -0.0813 |
| motion_smoothness | 0.9793 | 0.9835 | 0.0042 |
| temporal_flickering | 0.9754 | 0.9729 | -0.0025 |

#### lion_and_mouse

- Core benchmark: `/home/vault/v123be/v123be36/sceneweaver_runs/core_lion_and_mouse_260319_225545/benchmark_reports/combined/combined_benchmark_core_lion_and_mouse_260319_225545/summary.json`
- Agents benchmark: `/home/vault/v123be/v123be36/sceneweaver_runs/agents_lion_mouse_agents_a100_260321_220526/benchmark_reports/combined/combined_benchmark_agents_lion_mouse_agents_a100_260321_220526_parserfix_rerun_20260322/summary.json`
- Prompt average: core `1.7333` vs agents `1.4000`
- Continuity average: core `0.8490` vs agents `0.8079`

| Metric | Core | Agents | Delta |
|---|---:|---:|---:|
| video-text consistency | 1.8333 | 0.3333 | -1.5000 |
| action | 1.5000 | 2.5000 | 1.0000 |
| scene | 1.6667 | 1.5000 | -0.1667 |
| object_class | 1.8333 | 1.5000 | -0.3333 |
| color | 1.8333 | 1.1667 | -0.6667 |
| subject_consistency | 0.5894 | 0.5126 | -0.0768 |
| background_consistency | 0.8368 | 0.7681 | -0.0687 |
| motion_smoothness | 0.9867 | 0.9812 | -0.0054 |
| temporal_flickering | 0.9831 | 0.9695 | -0.0136 |

#### thirsty_crow

- Core benchmark: `/home/vault/v123be/v123be36/sceneweaver_runs/core_thirsty_crow_260319_225546/benchmark_reports/combined/combined_benchmark_core_thirsty_crow_260319_225546/summary.json`
- Agents benchmark: `/home/vault/v123be/v123be36/sceneweaver_runs/agents_thirsty_crow_agents_a100_260321_220515/benchmark_reports/combined/combined_benchmark_agents_thirsty_crow_agents_a100_260321_220515_parserfix_rerun_20260322/summary.json`
- Prompt average: core `1.7667` vs agents `2.0333`
- Continuity average: core `0.9084` vs agents `0.8727`

| Metric | Core | Agents | Delta |
|---|---:|---:|---:|
| video-text consistency | 2.0000 | 0.5000 | -1.5000 |
| action | 1.0000 | 2.6667 | 1.6667 |
| scene | 2.5000 | 2.3333 | -0.1667 |
| object_class | 2.1667 | 2.5000 | 0.3333 |
| color | 1.1667 | 2.1667 | 1.0000 |
| subject_consistency | 0.7840 | 0.6777 | -0.1063 |
| background_consistency | 0.8692 | 0.8451 | -0.0241 |
| motion_smoothness | 0.9908 | 0.9859 | -0.0048 |
| temporal_flickering | 0.9896 | 0.9820 | -0.0076 |

#### tortoise_and_hare

- Core benchmark: `/home/vault/v123be/v123be36/sceneweaver_runs/core_tortoise_and_hare_260319_225652/benchmark_reports/combined/combined_benchmark_core_tortoise_and_hare_260319_225652/summary.json`
- Agents benchmark: `/home/vault/v123be/v123be36/sceneweaver_runs/agents_tortoise_hare_agents_a100_260321_220555/benchmark_reports/combined/combined_benchmark_agents_tortoise_hare_agents_a100_260321_220555_parserfix_rerun2_20260322/summary.json`
- Prompt average: core `1.6000` vs agents `1.0667`
- Continuity average: core `0.8155` vs agents `0.7821`

| Metric | Core | Agents | Delta |
|---|---:|---:|---:|
| video-text consistency | 1.6667 | 0.5000 | -1.1667 |
| action | 1.5000 | 0.5000 | -1.0000 |
| scene | 1.8333 | 1.0000 | -0.8333 |
| object_class | 1.6667 | 1.3333 | -0.3333 |
| color | 1.3333 | 2.0000 | 0.6667 |
| subject_consistency | 0.5105 | 0.5016 | -0.0089 |
| background_consistency | 0.7862 | 0.7735 | -0.0127 |
| motion_smoothness | 0.9855 | 0.9370 | -0.0485 |
| temporal_flickering | 0.9798 | 0.9165 | -0.0634 |


## Agent Workflow Evidence

The agent pipeline is not the video generator itself; it is a refinement loop layered on top of video generation. For each window, the system generates a candidate clip, evaluates it with specialized agents, and retries with a tighter prompt until either the quality threshold is reached or the iteration budget is exhausted. In this codebase, the agent stack corresponds to continuity, story-beat, and physics checks, with prompt optimization used to revise the next attempt based on critique. The relevant implementation lives in [run_story_pipeline_with_agents.py](/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/scripts/run_story_pipeline_with_agents.py), [refinement_engine.py](/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/multi_agent_refinement/refinement_engine.py), [continuity_auditor.py](/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/multi_agent_refinement/agents/continuity_auditor.py), [storybeats_checker.py](/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/multi_agent_refinement/agents/storybeats_checker.py), and [physics_validator.py](/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/multi_agent_refinement/agents/physics_validator.py).

| Story | Quality Threshold | Total Windows | Avg Iterations/Window | Max Iterations Observed | Slurm Runtime | Log |
|---|---:|---:|---:|---:|---:|---|
| fox_and_grapes | 0.90 | 6 | 3.00 | 3 | 04:37:41 | `/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/slurm_logs/sceneweaver_foxgrapes_3486279.out` |
| lion_and_mouse | 0.90 | 6 | 3.00 | 3 | 04:37:17 | `/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/slurm_logs/sceneweaver_lionmouse_3486278.out` |
| thirsty_crow | 0.90 | 6 | 3.00 | 3 | 04:26:33 | `/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/slurm_logs/sceneweaver_crow_hq_3486277.out` |
| tortoise_and_hare | 0.90 | 6 | 3.33 | 4 | 05:08:17 | `/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/slurm_logs/sceneweaver_torthare_3486280.out` |

Observations from the Slurm logs:
- All four agent runs used `--max-iterations 10` and `--quality-threshold 0.90`, but in practice they converged in 3.00 to 3.33 iterations per window.
- Each agent run planned exactly 6 windows for a 48-second story, then generated one accepted clip per window.
- The most difficult agent run was tortoise and hare, which required 4 iterations on two windows and had the longest runtime at 05:08:17.
- The other three agent runs converged uniformly in 3 iterations per window, with total job runtimes between 04:26:33 and 04:37:41.

## Reliability Notes

- VBench continuity scores remain well-formed and stay in the expected 0-1 range for all runs.
- The latest agent reruns removed the previous out-of-range `object_class = 3.3333` anomaly by clamping parser extraction to valid local prompt-score formats.
- Video-Bench prompt scores now all lie in the effective local range 0-3. A score of `0` indicates that the local judge response could not be parsed into a valid prompt rating for that window or averaged metric path.
- The most important remaining local-evaluator weakness is `video-text consistency`: several agent reruns still receive low or zero values because the local VLM does not consistently emit a stable score template even when other prompt metrics succeed.

## Experiments

We compare two variants of the SceneWeaver pipeline on four short fable-style stories: a core pipeline and an agent-based refinement pipeline. Both settings use the same underlying video backbone, the same story window plans, and the same benchmark suite. The benchmark suite consists of five Video-Bench prompt-alignment metrics (`video-text consistency`, `action`, `scene`, `object_class`, and `color`) and four VBench continuity metrics (`subject_consistency`, `background_consistency`, `motion_smoothness`, and `temporal_flickering`). For the core setting, evaluation is performed on the completed runs stored under the `core_*` directories. For the agent setting, evaluation is performed on the latest full reruns produced after the local Video-Bench parser fixes on March 22, 2026.

The agent pipeline augments generation with an iterative critique-and-revision loop. For each story window, the system first generates a candidate clip, then evaluates it using continuity, story-beat, and physics agents. These agent signals are aggregated into a window quality score, and when the score is insufficient the prompt is tightened and generation is retried. In the submitted runs, the maximum budget was set to 10 iterations per window with a target quality threshold of 0.90. In practice, all stories converged much earlier, averaging between 3.00 and 3.33 iterations per window over six windows per story. This shows that the agent loop remained active and selective, but did not require the full iteration budget.

Across the four stories, the aggregate continuity average remains lower for the agent setting than for the core setting (`0.8212` vs `0.8637`). After the latest reruns, the aggregate prompt average is also slightly lower for agents (`1.5250` vs `1.5583`), although that aggregate hides a strong metric split. The agent setting is still better on `action` (`1.9167` vs `1.3333`), `object_class` (`2.0000` vs `1.7500`), and `color` (`2.0417` vs `1.3333`), but it is now clearly worse on `video-text consistency` and moderately worse on `scene`.

The story-wise results are mixed. Fox and grapes and thirsty crow still benefit from the agent controller on prompt-facing metrics, with higher prompt averages than their core counterparts (`1.6000` vs `1.1333` and `2.0333` vs `1.7667`), but both agent runs continue to trail core on continuity. Lion and mouse and tortoise and hare remain weaker in the agent setting on both overall prompt average and continuity average. Tortoise and hare is still the hardest agent story: it has the lowest agent continuity profile and required the highest iteration count during generation.

The rerun results also change the interpretation of reliability. The previous out-of-range `object_class` anomaly is gone, so the agent prompt metrics are now numerically bounded. However, the local Video-Bench evaluator still produces zero-heavy `video-text consistency` outputs when the local judge fails to emit a stable parseable score. This means the updated comparison is cleaner numerically, but the absolute prompt-alignment values, especially for `video-text consistency`, should still be treated as conservative local-evaluator estimates rather than fully equivalent hosted GPT-based Video-Bench scores.

## Conclusion

The updated experiments do not support a general claim that the current agent-based SceneWeaver pipeline outperforms the core pipeline overall. After replacing the stale agent reports with the latest full reruns, the core pipeline remains better on average continuity (`0.8637` vs `0.8212`) and also ends slightly ahead on the overall prompt-metric mean (`1.5583` vs `1.5250`). The most defensible interpretation is therefore selective improvement rather than broad superiority.

The agent controller is still useful. Its strongest gains remain concentrated in metrics that reflect explicit beat visibility and prompt-grounded appearance, especially `action`, `object_class`, and `color`. That pattern is visible most clearly for fox and grapes and thirsty crow, where the agent runs improve semantic prompt-facing metrics even while losing continuity. In other words, the current multi-agent loop behaves more like a semantic refinement mechanism than a continuity-preserving generator controller.

The main practical limitation is evaluation robustness. Although the parser fixes removed the earlier out-of-range anomaly, the local Video-Bench path still under-scores some agent runs in `video-text consistency` because the local judge does not always emit a stable score template. So the current reports are cleaner and more trustworthy than the earlier draft, but trends should still be emphasized over small absolute differences. A sensible next step is to keep the agent loop for selective semantic repair while strengthening the continuity objective and stabilizing the local prompt-evaluation stack.

