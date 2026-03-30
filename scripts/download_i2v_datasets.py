#!/usr/bin/env python3
"""
Download and prepare high-quality datasets for I2V fine-tuning.
Focuses on datasets that will improve temporal coherence and video quality.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import datasets
from tqdm import tqdm
import os


def download_i2v_datasets(
    datasets_config: List[Dict],
    output_dir: str = "Datasets/video_finetune",
    max_samples_per_dataset: Optional[int] = None,
):
    """Download and prepare I2V fine-tuning datasets."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for dataset_config in datasets_config:
        dataset_key = dataset_config["key"]
        repo_id = dataset_config["repo_id"]
        local_subdir = dataset_config["local_subdir"]

        print(f"\n=== Downloading {dataset_key} ===")
        print(f"Repository: {repo_id}")
        print(f"Local path: {output_path / local_subdir}")

        try:
            # Load dataset
            load_kwargs = {"path": repo_id}
            if "config" in dataset_config:
                load_kwargs["name"] = dataset_config["config"]

            dataset = datasets.load_dataset(**load_kwargs)

            # Process each split
            for split_name, split_data in dataset.items():
                if split_name not in dataset_config.get("splits", []):
                    continue

                print(f"Processing split: {split_name}")

                # Limit samples if specified
                if max_samples_per_dataset:
                    split_data = split_data.select(range(min(max_samples_per_dataset, len(split_data))))

                # Save to disk
                split_output_dir = output_path / local_subdir / split_name
                split_output_dir.mkdir(parents=True, exist_ok=True)

                # Save as JSON for easier processing
                data_list = []
                for i, sample in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
                    # Convert sample to serializable format
                    processed_sample = {}

                    # Handle different data types
                    for key, value in sample.items():
                        if hasattr(value, 'tolist'):  # numpy arrays
                            processed_sample[key] = value.tolist()
                        elif isinstance(value, (str, int, float, bool)):
                            processed_sample[key] = value
                        elif isinstance(value, list):
                            processed_sample[key] = value
                        else:
                            # Skip complex objects like PIL images for now
                            processed_sample[key] = str(type(value))

                    data_list.append(processed_sample)

                # Save split
                output_file = split_output_dir / "data.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, ensure_ascii=False, indent=2)

                print(f"Saved {len(data_list)} samples to {output_file}")

        except Exception as e:
            print(f"Error downloading {dataset_key}: {e}")
            continue

    print("\n=== Download Complete ===")
    print(f"All datasets saved to: {output_path}")


def main():
    # I2V fine-tuning dataset configurations
    i2v_datasets = [
        {
            "key": "msr_vtt_enhanced",
            "repo_id": "AlexZigma/msr-vtt",
            "local_subdir": "msr_vtt_enhanced",
            "splits": ["train", "validation", "test"],
            "description": "Enhanced MSR-VTT dataset with better captions"
        },
        {
            "key": "video_feedback_annotated",
            "repo_id": "TIGER-Lab/VideoFeedback",
            "local_subdir": "video_feedback_annotated",
            "splits": ["train", "validation"],
            "config": "annotated",
            "description": "Annotated video feedback dataset"
        },
        {
            "key": "llava_video_academic",
            "repo_id": "lmms-lab/LLaVA-Video-178K",
            "local_subdir": "llava_video_academic",
            "splits": ["train"],
            "config": "0_30_s_academic_v0_1",
            "description": "LLaVA video instruction dataset"
        }
    ]

    parser = argparse.ArgumentParser(description="Download I2V fine-tuning datasets")
    parser.add_argument("--output_dir", type=str, default="Datasets/video_finetune",
                       help="Output directory for datasets")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples per dataset (for testing)")
    parser.add_argument("--datasets", type=str, nargs="+",
                       choices=[d["key"] for d in i2v_datasets],
                       help="Specific datasets to download (default: all)")

    args = parser.parse_args()

    # Filter datasets if specified
    if args.datasets:
        selected_datasets = [d for d in i2v_datasets if d["key"] in args.datasets]
    else:
        selected_datasets = i2v_datasets

    download_i2v_datasets(
        datasets_config=selected_datasets,
        output_dir=args.output_dir,
        max_samples_per_dataset=args.max_samples,
    )


if __name__ == "__main__":
    main()