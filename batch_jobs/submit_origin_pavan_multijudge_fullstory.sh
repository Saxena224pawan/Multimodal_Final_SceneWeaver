#!/bin/bash -l
set -euo pipefail

project_root="${PROJECT_ROOT:-/home/hpc/v123be/v123be36/Multimodal_Final_SceneWeaver}"
array_script="${ARRAY_SCRIPT:-${project_root}/batch_jobs/run_vbench_origin_pavan_simple_fullstory_array.sh}"
logs_dir="${project_root}/slurm_logs"
mkdir -p "${logs_dir}"

[ -x "${array_script}" ] || { echo "Missing array launcher: ${array_script}"; exit 1; }

judge_filter_csv="${JUDGE_FILTER:-}"
sbatch_gres="${SBATCH_GRES:-}"
sbatch_cpus_per_task="${SBATCH_CPUS_PER_TASK:-}"
sbatch_partition="${SBATCH_PARTITION_OVERRIDE:-}"
sbatch_time="${SBATCH_TIME_OVERRIDE:-}"

declare -a judges=(
  "smolvlm|${project_root}/configs/videobench_local_smolvlm.json|benchmark_reports_fullstory_promptjudge_smolvlm"
  "llava15_7b|${project_root}/configs/videobench_local_llava15_7b.json|benchmark_reports_fullstory_promptjudge_llava15_7b_sampled3_scored"
)

gpt_config_path="${OPENAI_GPT_CONFIG_PATH:-}"
if [ -n "${gpt_config_path}" ] && [ -f "${gpt_config_path}" ]; then
  judges+=("gpt4o_mini|${gpt_config_path}|benchmark_reports_fullstory_promptjudge_gpt4o_mini")
else
  echo "OPENAI_GPT_CONFIG_PATH not set or missing; GPT judge submission will be skipped."
fi

declare -a variants=(
  "simple|${project_root}/outputs/story_runs_origin_pavan"
  "core|${project_root}/outputs/story_runs_origin_pavan"
  "agentic|${project_root}/outputs/story_runs_origin_pavan"
  "core|${project_root}/outputs/story_runs_origin_pavan_i2v_concat"
  "agentic|${project_root}/outputs/story_runs_origin_pavan_i2v_concat"
)

declare -a labels=(
  "simple_t2v"
  "core_t2v"
  "agentic_t2v"
  "core_i2v"
  "agentic_i2v"
)

dependency_arg=()
submitted_ids=()

for judge_spec in "${judges[@]}"; do
  IFS='|' read -r judge_tag config_path report_root_name <<<"${judge_spec}"
  if [ -n "${judge_filter_csv}" ]; then
    if [[ ",${judge_filter_csv}," != *",${judge_tag},"* ]]; then
      continue
    fi
  fi
  [ -f "${config_path}" ] || { echo "Config missing for ${judge_tag}: ${config_path}"; exit 1; }

  for idx in "${!variants[@]}"; do
    IFS='|' read -r mode runs_root <<<"${variants[$idx]}"
    variant_label="${labels[$idx]}"
    echo "Submitting ${variant_label} with judge ${judge_tag}"

    submit_cmd=(
      sbatch
      --parsable
      --array=0-3%1
      --export=ALL,MODE="${mode}",RUNS_ROOT="${runs_root}",REPORT_ROOT_NAME="${report_root_name}",VIDEOBENCH_CONFIG_PATH="${config_path}",RUN_WINDOW_PROMPT=1,RUN_CONTINUITY=0,REQUIRE_WINDOW_PROMPT=1
    )
    if [ -n "${sbatch_gres}" ]; then
      submit_cmd+=(--gres="${sbatch_gres}")
    fi
    if [ -n "${sbatch_cpus_per_task}" ]; then
      submit_cmd+=(--cpus-per-task="${sbatch_cpus_per_task}")
    fi
    if [ -n "${sbatch_partition}" ]; then
      submit_cmd+=(--partition="${sbatch_partition}")
    fi
    if [ -n "${sbatch_time}" ]; then
      submit_cmd+=(--time="${sbatch_time}")
    fi
    if [ "${#dependency_arg[@]}" -gt 0 ]; then
      submit_cmd+=("${dependency_arg[@]}")
    fi
    submit_cmd+=("${array_script}")

    job_id="$("${submit_cmd[@]}")"
    echo "Submitted ${variant_label} / ${judge_tag}: ${job_id}"
    submitted_ids+=("${judge_tag}:${variant_label}:${job_id}")
    dependency_arg=(--dependency="afterany:${job_id}")
  done
done

printf '%s\n' "${submitted_ids[@]}"
