"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import random
import shutil

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "your_hf_username/libero"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def main(data_dir: str, *, push_to_hub: bool = False, val_split: float = 0.1, seed: int = 42):
    # Clean up any existing dataset in the output directory
    train_repo = REPO_NAME + "-train"
    val_repo = REPO_NAME + "-val"
    train_output_path = LEROBOT_HOME / train_repo
    val_output_path = LEROBOT_HOME / val_repo
    for output_path in [train_output_path, val_output_path]:
        if output_path.exists():
            shutil.rmtree(output_path)

    # Create LeRobot datasets for train and val
    features = {
        "image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    }
    train_dataset = LeRobotDataset.create(
        repo_id=train_repo,
        robot_type="panda",
        fps=10,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    val_dataset = LeRobotDataset.create(
        repo_id=val_repo,
        robot_type="panda",
        fps=10,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Gather all episodes from all datasets
    episodes = []
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            steps = list(episode["steps"].as_numpy_iterator())
            language_instruction = steps[-1]["language_instruction"] if "language_instruction" in steps[-1] else b""
            episodes.append((steps, language_instruction))

    # Shuffle and split into train/val
    random.seed(seed)
    random.shuffle(episodes)
    n_val = int(len(episodes) * val_split)
    val_episodes = episodes[:n_val]
    train_episodes = episodes[n_val:]

    # Write train episodes
    for steps, language_instruction in train_episodes:
        for step in steps:
            train_dataset.add_frame(
                {
                    "image": step["observation"]["image"],
                    "wrist_image": step["observation"]["wrist_image"],
                    "state": step["observation"]["state"],
                    "actions": step["action"],
                }
            )
        train_dataset.save_episode(
            task=language_instruction.decode() if isinstance(language_instruction, bytes) else str(language_instruction)
        )

    # Write val episodes
    for steps, language_instruction in val_episodes:
        for step in steps:
            val_dataset.add_frame(
                {
                    "image": step["observation"]["image"],
                    "wrist_image": step["observation"]["wrist_image"],
                    "state": step["observation"]["state"],
                    "actions": step["action"],
                }
            )
        val_dataset.save_episode(
            task=language_instruction.decode() if isinstance(language_instruction, bytes) else str(language_instruction)
        )

    # Consolidate datasets
    train_dataset.consolidate(run_compute_stats=False)
    val_dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        train_dataset.push_to_hub(
            tags=["libero", "panda", "rlds", "train"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        val_dataset.push_to_hub(
            tags=["libero", "panda", "rlds", "val"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
