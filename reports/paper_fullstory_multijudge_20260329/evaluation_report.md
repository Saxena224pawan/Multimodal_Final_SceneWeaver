# Multi-Judge Full-Story Benchmark Report

## Scope

This report compares Video-Bench prompt metrics across multiple judge models while reusing the same VBench continuity metrics from the baseline full-story evaluation. The continuity benchmark is judge-independent and is therefore copied from the baseline report for every judge row.

## Judge Coverage

| Judge | Variants with complete prompt results | Variants with failed prompt results |
| --- | ---: | ---: |
| Qwen2.5-VL-7B-Instruct | 5/5 | 0/5 |
| SmolVLM-Instruct | 0/5 | 5/5 |
| LLaVA-1.5-7B-HF | 0/5 | 5/5 |

## Aggregate Prompt Metrics by Judge

### Qwen2.5-VL-7B-Instruct

| Variant | Mean VTC (0-5) | Mean Action (0-3) | Mean Scene (0-3) | Mean Object Class (0-3) | Mean Color (0-3) | Mean prompt eval time (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple T2V | 3.5000 | 1.7500 | 1.0000 | 2.0000 | 2.0000 | 8.18 |
| Core T2V | 1.7500 | 2.2500 | 2.2500 | 2.7500 | 2.2500 | 14.16 |
| Agentic T2V | 3.0000 | 2.0000 | 1.2500 | 2.2500 | 2.2500 | 44.76 |
| Core I2V | 2.7500 | 2.0000 | 1.7500 | 2.5000 | 2.7500 | 19.26 |
| Agentic I2V | 3.2500 | 3.0000 | 1.2500 | 2.2500 | 2.0000 | 35.15 |

### SmolVLM-Instruct

| Variant | Mean VTC (0-5) | Mean Action (0-3) | Mean Scene (0-3) | Mean Object Class (0-3) | Mean Color (0-3) | Mean prompt eval time (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple T2V | failed | failed | failed | failed | failed | failed |
| Core T2V | failed | failed | failed | failed | failed | failed |
| Agentic T2V | failed | failed | failed | failed | failed | failed |
| Core I2V | failed | failed | failed | failed | failed | failed |
| Agentic I2V | failed | failed | failed | failed | failed | failed |

### LLaVA-1.5-7B-HF

| Variant | Mean VTC (0-5) | Mean Action (0-3) | Mean Scene (0-3) | Mean Object Class (0-3) | Mean Color (0-3) | Mean prompt eval time (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple T2V | failed | failed | failed | failed | failed | failed |
| Core T2V | failed | failed | failed | failed | failed | failed |
| Agentic T2V | failed | failed | failed | failed | failed | failed |
| Core I2V | failed | failed | failed | failed | failed | failed |
| Agentic I2V | failed | failed | failed | failed | failed | failed |

## Continuity Metrics

The VBench continuity metrics are unchanged by the Video-Bench judge model. Reuse the baseline continuity tables from the existing report when the new prompt-judge results are merged into the paper text.

## Notes

- Rows marked `pending` indicate that the corresponding prompt-judge benchmark outputs have not been found yet under the expected report root.
- Rows marked `failed` indicate that prompt outputs were found, but the score JSONs contained no usable `average_scores`.
- In the current local reruns, both `SmolVLM-Instruct` and `LLaVA-1.5-7B-HF` produced prompt-evaluation outputs with judge-side failures rather than usable scores. The saved stdout logs show `CUDA out of memory` errors during evaluation.
- The current script expects judge-specific prompt outputs under each run directory, for example `benchmark_reports_fullstory_promptjudge_smolvlm`.
