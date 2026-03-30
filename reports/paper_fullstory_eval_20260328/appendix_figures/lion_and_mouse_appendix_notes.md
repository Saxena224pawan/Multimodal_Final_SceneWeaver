# Lion And Mouse Appendix Figures

Story prompt: A lion spares a frightened mouse, later gets trapped in a hunter's net, and the mouse returns to gnaw the ropes and set the lion free.

Generated files:
- `lion_and_mouse_boundary_pairs.png`: full-frame last-5 to first-5 boundary strips.
- `lion_and_mouse_boundary_pairs_paper.png`: compact NeurIPS-friendly boundary figure with four representative transitions and one shared method-label column.
- `lion_and_mouse_boundary_pairs_center_crop.png`: center-crop version of the same 5-frame strips.
- `lion_and_mouse_window_storyboard.png`: middle-frame storyboard across the eight windows.

Suggested appendix caption:
Figure X compares cross-window continuity for the same story across the five generation variants using a compact matrix layout. Rows correspond to methods, with method labels shown once in the shared left column, and columns show four representative transitions (W0->W1, W2->W3, W4->W5, and W6->W7). Within each cell, the top strip shows the last five frames of window k and the bottom strip shows the first five frames of window k+1, making visual carryover across windows directly inspectable in a NeurIPS-friendly figure width.

Resolved clip sources:
- `Simple T2V`: `/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/outputs/story_runs_origin_pavan/simple_lion_and_mouse_origin_pavan_simple_t2v14b_640x384_f64_s20_w8_concat_a40x2_20260326_232406/clips`
- `Core T2V`: `/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/outputs/story_runs_origin_pavan/core_lion_and_mouse_origin_pavan_core_t2v14b_640x384_f64_s20_w8_concat_a40x2_20260326_232406/clips`
- `Agentic T2V`: `/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/outputs/story_runs_origin_pavan/agentic_lion_and_mouse_origin_pavan_agentic_t2v14b_640x384_f64_s20_w8_concat_a40x2_20260326_232406/clips`
- `Core I2V`: `/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/outputs/story_runs_origin_pavan_i2v_runs/core_lion_and_mouse_origin_pavan_core_i2v_t14b_640x384_f64_s20_w8_a40x2_20260327_113205/clips`
- `Agentic I2V`: `/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver/outputs/story_runs_origin_pavan_i2v_runs/agentic_lion_and_mouse_origin_pavan_agentic_i2v_t14b_640x384_f64_s20_w8_a40x2_20260327_113205/clips`
