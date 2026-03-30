# Full-Story Benchmark Evaluation Report

## Scope and Goal

This report evaluates full-story benchmark results for SceneWeaver runs. The comparison spans three control strategies, `simple`, `core`, and `agentic`, and two conditioning backbones, text-to-video (T2V) and image-to-video (I2V), on four short narrative tasks: Fox and Grapes, Lion and Mouse, Thirsty Crow, and Tortoise and Hare. All scores are reported on the concatenated full-story clip for each run. This version of the report removes every derived overall or composite score and keeps only the original benchmark outputs on their native scales.

The report covers five experiment families: Simple T2V, Core T2V, Agentic T2V, Core I2V, and Agentic I2V. This gives 20 benchmarked runs in total. The agentic benchmark families were rerun end-to-end, and the report below uses those refreshed outputs for every agentic story.

## Experimental Setup

All runs use the same narrative workload: four Aesop-style stories, fixed to eight windows per story, with target resolution `640x384`, `64` frames per window, `20` denoising steps, and `8` fps. The target story duration is therefore `80` seconds, i.e. eight `10`-second windows. The benchmark evaluates the concatenated full-story clip for each run.

The benchmark input is always the full concatenated story clip, not the per-window clips. Both the prompt benchmark and the continuity benchmark read that same story-level video. This ensures that Video-Bench and VBench score the same temporal artifact and that the paper tables reflect story-level, not window-level, behavior.

| Family | Backbone | Pipeline | Memory / carryover | Refinement controller |
| --- | --- | --- | --- | --- |
| Simple T2V | Wan `2.2-T2V-A14B-Diffusers` | standard story pipeline | no embedding memory, no last-frame memory, no reference conditioning | none |
| Core T2V | Wan `2.2-T2V-A14B-Diffusers` | standard story pipeline | DINOv2 memory, continuity adapter, last_frame_memory, continuity_candidates=2, continuity_regen_attempts=2 | none |
| Agentic T2V | Wan `2.2-T2V-A14B-Diffusers` | agent-guided story pipeline | DINOv2 embeddings, no reference conditioning | continuity + storybeats + physics agents, max_iterations=5, quality_threshold=0.76 |
| Core I2V | Wan `2.2-I2V-A14B-Diffusers` | standard story pipeline | DINOv2 memory, continuity adapter, starter image, reference_tail_frames=4, reference_strength=0.70 | none |
| Agentic I2V | Wan `2.2-I2V-A14B-Diffusers` | agent-guided story pipeline | starter image plus previous-window visual carryover with the same I2V reference settings | continuity + storybeats + physics agents, max_iterations=5, quality_threshold=0.76 |


Generation used Wan `2.2` 14B backbones with a Qwen `2.5-3B-Instruct` director model for shot and beat planning. The simple and core T2V results and the I2V results were collected from their respective story-run outputs. The agentic T2V and I2V results use refreshed benchmark outputs from the rerun evaluations.

## Video-Bench Setup Details

The prompt benchmark was executed with the full-story prompt-evaluation pipeline. For the full-story setting, the wrapper builds a temporary one-clip Video-Bench dataset, links the concatenated full-story clip as the single evaluation video, and writes a story-level prompt specification. Prompt sourcing is set to `auto`, which first attempts to recover a story prompt from the run metadata and then falls back to the internal story library if required.

The actual Video-Bench judge is served locally using `Qwen2.5-VL-7B-Instruct` behind an OpenAI-compatible endpoint. A fixed local server configuration was used for every run. Each full-story evaluation runs the five Video-Bench dimensions sequentially: `video-text consistency`, `action`, `scene`, `object_class`, and `color`. The wrapper preserves the original Video-Bench score outputs, per-dimension logs, and a run-level summary for each story.

The agentic reruns therefore validate the prompt benchmark under the exact same local judge setup that is used in the rest of the paper. The full agentic reruns reproduced the same raw Video-Bench prompt scores as the previous corrected report, while the VBench continuity values changed only at floating-point scale in `motion_smoothness`.

## VBench Setup Details

Continuity was evaluated on the same concatenated story video that was passed to Video-Bench. The reported continuity metrics are `subject_consistency`, `background_consistency`, `motion_smoothness`, and `temporal_flickering`, all read directly from the VBench outputs. No rescaling or cross-metric aggregation is applied in this paper report.

## Metric Definitions and Original Scales

| Metric | Benchmark | Native scale | How it is obtained in this report |
| --- | --- | --- | --- |
| `video-text consistency` | Video-Bench | `0-5` | Raw `average_scores` value from the Video-Bench output. No normalization. |
| `action` | Video-Bench | `0-3` | Raw `average_scores` value from Video-Bench. No normalization. |
| `scene` | Video-Bench | `0-3` | Raw `average_scores` value from Video-Bench. No normalization. |
| `object_class` | Video-Bench | `0-3` | Raw `average_scores` value from Video-Bench. No normalization. |
| `color` | Video-Bench | `0-3` | Raw `average_scores` value from Video-Bench. No normalization. |
| `subject_consistency` | VBench | native VBench score, typically `0-1` | Directly read from the VBench outputs. |
| `background_consistency` | VBench | native VBench score, typically `0-1` | Directly read from the VBench outputs. |
| `motion_smoothness` | VBench | native VBench score, typically `0-1` | Directly read from the VBench outputs. |
| `temporal_flickering` | VBench | native VBench score, typically `0-1` | Directly read from the VBench outputs. |

No overall score, prompt composite, or continuity composite is used below. Metrics are only compared within their own original scales. This means, for example, that `video-text consistency` should not be averaged together with `action`, because they come from different native ranges (`0-5` vs `0-3`).

## Aggregate Prompt Metrics

| Variant | Mean VTC (0-5) | Mean Action (0-3) | Mean Scene (0-3) | Mean Object Class (0-3) | Mean Color (0-3) | Mean prompt eval time (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple T2V | 3.5000 | 1.7500 | 1.0000 | 2.0000 | 2.0000 | 8.18 |
| Core T2V | 1.7500 | 2.2500 | 2.2500 | 2.7500 | 2.2500 | 14.16 |
| Agentic T2V | 3.0000 | 2.0000 | 1.2500 | 2.2500 | 2.2500 | 44.76 |
| Core I2V | 2.7500 | 2.0000 | 1.7500 | 2.5000 | 2.7500 | 19.26 |
| Agentic I2V | 3.2500 | 3.0000 | 1.2500 | 2.2500 | 2.0000 | 35.15 |


At the raw prompt-metric level, the leader on mean `video-text consistency` is `Simple T2V` at `3.5000` on the native `0-5` scale. The leader on mean `action` is `Agentic I2V` at `3.0000` on the native `0-3` scale. The leader on mean `scene` is `Core T2V` at `2.2500` on the native `0-3` scale. The leader on mean `color` is `Core I2V` at `2.7500` on the native `0-3` scale.

## Aggregate Continuity Metrics

| Variant | Mean Subject Consistency | Mean Background Consistency | Mean Motion Smoothness | Mean Temporal Flickering | Mean continuity eval time (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple T2V | 0.7696 | 0.9135 | 0.9866 | 0.9832 | 1.11 |
| Core T2V | 0.7693 | 0.9191 | 0.9840 | 0.9801 | 1.05 |
| Agentic T2V | 0.6816 | 0.8595 | 0.9852 | 0.9798 | 0.99 |
| Core I2V | 0.6430 | 0.7915 | 0.9724 | 0.9629 | 1.10 |
| Agentic I2V | 0.6296 | 0.7756 | 0.9651 | 0.9490 | 1.07 |


At the raw continuity level, the leader on `subject_consistency` is `Simple T2V` at `0.7696`. The leader on `background_consistency` is `Core T2V` at `0.9191`. The leader on `motion_smoothness` is `Simple T2V` at `0.9866`, and the leader on `temporal_flickering` is `Simple T2V` at `0.9832`.

## Fox and Grapes

### Video-Bench Raw Metrics

| Variant | VTC (0-5) | Action (0-3) | Scene (0-3) | Object Class (0-3) | Color (0-3) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple T2V | 4.0000 | 3.0000 | 0.0000 | 3.0000 | 3.0000 |
| Core T2V | 0.0000 | 3.0000 | 3.0000 | 3.0000 | 3.0000 |
| Agentic T2V | 5.0000 | 3.0000 | 2.0000 | 3.0000 | 3.0000 |
| Core I2V | 0.0000 | 1.0000 | 2.0000 | 3.0000 | 3.0000 |
| Agentic I2V | 4.0000 | 3.0000 | 2.0000 | 2.0000 | 2.0000 |

### VBench Raw Continuity Metrics

| Variant | Subject Consistency | Background Consistency | Motion Smoothness | Temporal Flickering |
| --- | ---: | ---: | ---: | ---: |
| Simple T2V | 0.6793 | 0.8809 | 0.9875 | 0.9847 |
| Core T2V | 0.6552 | 0.8776 | 0.9842 | 0.9800 |
| Agentic T2V | 0.6293 | 0.8279 | 0.9850 | 0.9799 |
| Core I2V | 0.6411 | 0.7593 | 0.9751 | 0.9634 |
| Agentic I2V | 0.6067 | 0.7656 | 0.9687 | 0.9524 |


## Lion and Mouse

### Video-Bench Raw Metrics

| Variant | VTC (0-5) | Action (0-3) | Scene (0-3) | Object Class (0-3) | Color (0-3) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple T2V | 4.0000 | 2.0000 | 2.0000 | 3.0000 | 3.0000 |
| Core T2V | 0.0000 | 2.0000 | 3.0000 | 3.0000 | 0.0000 |
| Agentic T2V | 0.0000 | 2.0000 | 2.0000 | 0.0000 | 3.0000 |
| Core I2V | 3.0000 | 3.0000 | 0.0000 | 3.0000 | 3.0000 |
| Agentic I2V | 0.0000 | 3.0000 | 0.0000 | 1.0000 | 2.0000 |

### VBench Raw Continuity Metrics

| Variant | Subject Consistency | Background Consistency | Motion Smoothness | Temporal Flickering |
| --- | ---: | ---: | ---: | ---: |
| Simple T2V | 0.7740 | 0.9171 | 0.9905 | 0.9879 |
| Core T2V | 0.7915 | 0.9425 | 0.9908 | 0.9891 |
| Agentic T2V | 0.6652 | 0.8844 | 0.9898 | 0.9864 |
| Core I2V | 0.5898 | 0.7833 | 0.9744 | 0.9667 |
| Agentic I2V | 0.5633 | 0.7443 | 0.9636 | 0.9436 |


## Thirsty Crow

### Video-Bench Raw Metrics

| Variant | VTC (0-5) | Action (0-3) | Scene (0-3) | Object Class (0-3) | Color (0-3) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple T2V | 3.0000 | 0.0000 | 2.0000 | 0.0000 | 2.0000 |
| Core T2V | 3.0000 | 2.0000 | 3.0000 | 2.0000 | 3.0000 |
| Agentic T2V | 3.0000 | 1.0000 | 1.0000 | 3.0000 | 0.0000 |
| Core I2V | 5.0000 | 3.0000 | 3.0000 | 3.0000 | 3.0000 |
| Agentic I2V | 4.0000 | 3.0000 | 3.0000 | 3.0000 | 3.0000 |

### VBench Raw Continuity Metrics

| Variant | Subject Consistency | Background Consistency | Motion Smoothness | Temporal Flickering |
| --- | ---: | ---: | ---: | ---: |
| Simple T2V | 0.8237 | 0.8997 | 0.9851 | 0.9820 |
| Core T2V | 0.8148 | 0.9066 | 0.9873 | 0.9849 |
| Agentic T2V | 0.7244 | 0.8419 | 0.9824 | 0.9751 |
| Core I2V | 0.7766 | 0.8688 | 0.9758 | 0.9697 |
| Agentic I2V | 0.8066 | 0.8446 | 0.9792 | 0.9716 |


## Tortoise and Hare

### Video-Bench Raw Metrics

| Variant | VTC (0-5) | Action (0-3) | Scene (0-3) | Object Class (0-3) | Color (0-3) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple T2V | 3.0000 | 2.0000 | 0.0000 | 2.0000 | 0.0000 |
| Core T2V | 4.0000 | 2.0000 | 0.0000 | 3.0000 | 3.0000 |
| Agentic T2V | 4.0000 | 2.0000 | 0.0000 | 3.0000 | 3.0000 |
| Core I2V | 3.0000 | 1.0000 | 2.0000 | 1.0000 | 2.0000 |
| Agentic I2V | 5.0000 | 3.0000 | 0.0000 | 3.0000 | 1.0000 |

### VBench Raw Continuity Metrics

| Variant | Subject Consistency | Background Consistency | Motion Smoothness | Temporal Flickering |
| --- | ---: | ---: | ---: | ---: |
| Simple T2V | 0.8012 | 0.9562 | 0.9834 | 0.9781 |
| Core T2V | 0.8157 | 0.9498 | 0.9739 | 0.9665 |
| Agentic T2V | 0.7077 | 0.8836 | 0.9836 | 0.9777 |
| Core I2V | 0.5644 | 0.7546 | 0.9642 | 0.9516 |
| Agentic I2V | 0.5420 | 0.7478 | 0.9491 | 0.9286 |


## Metric-Level Observations

The raw-metric view changes the emphasis of the discussion. `Core T2V` remains strong because it is consistently high across the continuity metrics and does not collapse on the object and color prompt dimensions. `Simple T2V` is still the best family on mean raw `video-text consistency`, but that advantage does not transfer into a uniform lead on the other Video-Bench dimensions. `Agentic I2V` remains strong on mean raw `action`, while `Core I2V` remains strong on mean raw `color`, so the prompt-side picture stays distributed across multiple families rather than collapsing to one dominant configuration.

The story-wise tables show why a single scalar can be misleading. On Fox and Grapes, `Agentic T2V` is strong on prompt-side `video-text consistency`, but its `subject_consistency` is not the best. On Thirsty Crow, the I2V variants are especially strong on the prompt side, but the continuity metrics still favor metric-by-metric reading rather than a merged score. Lion and Mouse remains the clearest example of why keeping the original scales matters: `Agentic I2V` has a very strong raw `action` score (`3.0`) and a moderate raw `color` score (`2.0`), but weak raw `video-text consistency` (`0.0`) and `scene` (`0.0`) scores.

Qualitatively, however, `Core I2V` and `Agentic I2V` appear much better at maintaining visual carryover from one window to the next. Across stories, these two I2V variants produce the most consistent cross-window appearance for characters, scene layout, and local visual context, even when that advantage is only partially reflected in the story-level benchmark tables.

## Runtime and Evaluation Cost

| Variant | Mean generation time (min) | Mean prompt eval time (min) | Mean continuity eval time (min) |
| --- | ---: | ---: | ---: |
| Simple T2V | 66.14 | 8.18 | 1.11 |
| Core T2V | 150.05 | 14.16 | 1.05 |
| Agentic T2V | 235.74 | 44.76 | 0.99 |
| Core I2V | 89.42 | 19.26 | 1.10 |
| Agentic I2V | 234.39 | 35.15 | 1.07 |


Measured generation wall times still show a large cost gap between the simple/core families and the agentic families. `Agentic T2V` and `Agentic I2V` remain the most expensive settings by a large margin. On the evaluation side, the reruns show that full-story Video-Bench prompt scoring is much slower and more variable than VBench continuity. The refreshed agentic prompt-evaluation means are now `44.76` minutes for `Agentic T2V` and `35.15` minutes for `Agentic I2V`, while continuity remains near one minute per story.

The rerun logs also show large story-to-story latency spread inside the same benchmark family. For `Agentic I2V`, Fox and Grapes completed prompt evaluation in about `8.31` minutes, while Lion and Mouse and Tortoise and Hare each required roughly `48-49` minutes. That spread is much larger than the change in raw scores, which indicates that the local judge path is the main source of evaluation-time variability rather than a change in the underlying generated video.

The internal optimization traces still show `quality_threshold=0.76`, `average_iterations=5.0`, and `converged_windows=0` for the agentic families. That result remains important even without any composite score: it indicates that the present controller spends extra compute without obtaining a clean, early-converged solution on any story.

## Conclusion

Without introducing any composite score, the raw metrics still support three robust conclusions. First, `Core T2V` is the most reliable balanced family because it maintains strong continuity metrics while staying competitive across the original Video-Bench prompt dimensions. Second, `Simple T2V` has the highest mean raw `video-text consistency`, while `Core T2V`, `Core I2V`, and `Agentic I2V` lead different subsets of the remaining prompt and continuity metrics, showing that no single family dominates every native benchmark axis. Third, the agentic families remain computationally expensive and story-dependent: they can lead on some raw prompt dimensions for some stories, but they do not dominate the raw continuity metrics and they do not converge early in the stored optimization traces.

## Paper-Ready Conclusion Excerpt

Using only the original benchmark outputs and no derived overall score, the results show that `Core T2V` is the most stable family across the raw VBench continuity metrics while remaining competitive on the raw Video-Bench prompt dimensions. `Simple T2V` achieves the highest mean raw `video-text consistency`, while `Core T2V`, `Core I2V`, and `Agentic I2V` lead different subsets of the remaining prompt and continuity metrics. The full agentic reruns confirm the same raw score pattern as the corrected earlier report and therefore support metric-by-metric reporting rather than a merged scalar summary.
