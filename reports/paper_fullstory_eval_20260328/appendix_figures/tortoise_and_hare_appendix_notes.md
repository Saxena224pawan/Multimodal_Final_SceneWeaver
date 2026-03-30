# Tortoise And Hare Appendix Figures

Story prompt: A hare mocks a slow tortoise, races ahead with confidence, falls asleep on the path, and wakes too late as the tortoise reaches the finish first.

Generated files:
- `tortoise_and_hare_boundary_pairs.png`: full-frame last-5 to first-5 boundary strips.
- `tortoise_and_hare_boundary_pairs_paper.png`: compact NeurIPS-friendly boundary figure with four representative transitions and one shared method-label column.
- `tortoise_and_hare_boundary_pairs_center_crop.png`: center-crop version of the same 5-frame strips.
- `tortoise_and_hare_window_storyboard.png`: middle-frame storyboard across the eight windows.

Suggested appendix caption:
Figure X compares cross-window continuity for the same story across the five generation variants using a compact matrix layout. Rows correspond to methods, with method labels shown once in the shared left column, and columns show four representative transitions (W0->W1, W2->W3, W4->W5, and W6->W7). Within each cell, the top strip shows the last five frames of window k and the bottom strip shows the first five frames of window k+1, making visual carryover across windows directly inspectable in a NeurIPS-friendly figure width.

Resolved clip sources:
- `Simple T2V`: `/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/outputs/story_runs_origin_pavan/simple_tortoise_and_hare_origin_pavan_simple_t2v14b_640x384_f64_s20_w8_concat_a40x2_20260326_232406/clips`
- `Core T2V`: `/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/outputs/story_runs_origin_pavan/core_tortoise_and_hare_origin_pavan_core_t2v14b_640x384_f64_s20_w8_concat_a40x2_20260326_232406/clips`
- `Agentic T2V`: `/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/outputs/story_runs_origin_pavan/agentic_tortoise_and_hare_origin_pavan_agentic_t2v14b_640x384_f64_s20_w8_concat_a40x2_20260326_232406/clips`
- `Core I2V`: `/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/outputs/story_runs_origin_pavan_i2v_runs/core_tortoise_and_hare_origin_pavan_core_i2v_t14b_640x384_f64_s20_w8_a40x2_20260327_113205/clips`
- `Agentic I2V`: `/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/outputs/story_runs_origin_pavan_i2v_runs/agentic_tortoise_and_hare_origin_pavan_agentic_i2v_t14b_640x384_f64_s20_w8_a40x2_20260327_113205/clips`
