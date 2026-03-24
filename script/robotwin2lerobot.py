#!/usr/bin/env python3
"""
Convert RoboTwin multiview data (view_*/data/episode*.hdf5) to LeRobot v2 dataset.


Dependencies:
  pip install lerobot h5py opencv-python-headless numpy torch tqdm

Example usage:
  export HF_LEROBOT_HOME=/path/to/lerobot_datasets   # optional, default is documented in lerobot

  python script/robotwin2lerobot.py \\
    --root /path/to/robotwin/data/place_can_basket/demo_multiview \\
    --repo-id place_can_basket_multiview \\
    --task-name place_can_basket
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import tqdm

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


ALOHA_MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]

CAMERA_MAP = [
    ("left_camera", "cam_left_wrist"),
    ("right_camera", "cam_right_wrist"),
    ("observer_camera", "cam_observer"),
]


def _natural_episode_paths(data_dir: Path) -> list[Path]:
    paths = list(data_dir.glob("episode*.hdf5"))

    def key(p: Path):
        m = re.search(r"episode(\d+)", p.stem)
        return int(m.group(1)) if m else 0

    paths.sort(key=key)
    return paths


def _decode_jpeg_ds(ds: h5py.Dataset) -> np.ndarray:
    """(T,) variable-length byte -> (T, H, W, 3) RGB uint8."""
    frames = []
    for i in range(ds.shape[0]):
        row = ds[i]
        buf = np.frombuffer(row.tobytes().rstrip(b"\0"), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode failed for a frame")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def _state_action_from_h5(ep: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(ep["joint_action/vector"][:], dtype=np.float32)
    action = np.empty_like(state)
    if len(state) > 1:
        action[:-1] = state[1:]
        action[-1] = state[-1]
    else:
        action[:] = state
    return state, action


def _task_description_from_task_name(task_name: str) -> str:
    """从 description/task_instruction/<task_name>.json 读取 full_description。"""
    repo_root = Path(__file__).resolve().parent.parent
    task_json = repo_root / "description" / "task_instruction" / f"{task_name}.json"
    if not task_json.is_file():
        raise SystemExit(f"Task instruction file not found: {task_json}")

    with open(task_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    full_description = str(data.get("full_description", "")).strip()
    if not full_description:
        raise SystemExit(f"'full_description' is empty in: {task_json}")
    return full_description


def collect_hdf5_list(root: Path) -> list[tuple[str, Path]]:
    """返回 [(view_name, hdf5_path), ...]，按 view 名与 episode 序号排序。"""
    out: list[tuple[str, Path]] = []
    for view_dir in sorted(root.glob("view_*")):
        if not view_dir.is_dir():
            continue
        data_dir = view_dir / "data"
        if not data_dir.is_dir():
            continue
        for ep in _natural_episode_paths(data_dir):
            out.append((view_dir.name, ep))
    return out


def probe_cameras(ep_path: Path) -> dict[str, tuple[int, int]]:
    shapes: dict[str, tuple[int, int]] = {}
    with h5py.File(ep_path, "r") as ep:
        for src, dst in CAMERA_MAP:
            key = f"observation/{src}/rgb"
            if key not in ep:
                continue
            rgb = _decode_jpeg_ds(ep[key])
            h, w = rgb.shape[1], rgb.shape[2]
            shapes[dst] = (h, w)
    if not shapes:
        raise SystemExit(f"No RGB cameras found under observation/*/rgb in {ep_path}")
    return shapes


def build_features(cam_shapes: dict[str, tuple[int, int]], mode: str) -> dict:
    feats = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(ALOHA_MOTORS),),
            "names": [ALOHA_MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ALOHA_MOTORS),),
            "names": [ALOHA_MOTORS],
        },
    }
    for name, (h, w) in cam_shapes.items():
        feats[f"observation.images.{name}"] = {
            "dtype": mode,
            "shape": (3, h, w),
            "names": ["channels", "height", "width"],
        }
    return feats


def convert(
    root: Path,
    repo_id: str,
    *,
    task: str,
    robot_type: str = "aloha",
    fps: int = 15,
    mode: str = "image",
    use_videos: bool = True,
    image_writer_processes: int = 4,
    image_writer_threads: int = 4,
) -> None:
    root = root.resolve()
    pairs = collect_hdf5_list(root)
    if not pairs:
        raise SystemExit(f"No view_*/data/episode*.hdf5 under {root}")

    cam_shapes = probe_cameras(pairs[0][1])
    features = build_features(cam_shapes, mode)

    out_dir = HF_LEROBOT_HOME / repo_id
    if out_dir.exists():
        shutil.rmtree(out_dir)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

    for ep_path in tqdm.tqdm(pairs, desc="episodes"):
        with h5py.File(ep_path, "r") as ep:
            state, action = _state_action_from_h5(ep)
            imgs: dict[str, np.ndarray] = {}
            for src, dst in CAMERA_MAP:
                key = f"observation/{src}/rgb"
                if key in ep:
                    imgs[dst] = _decode_jpeg_ds(ep[key])

            n = state.shape[0]
            for cam, arr in imgs.items():
                if arr.shape[0] != n:
                    raise ValueError(
                        f"{ep_path}: length mismatch {cam} frames={arr.shape[0]} state={n}"
                    )

            for i in range(n):
                frame = {
                    "observation.state": torch.from_numpy(state[i]),
                    "action": torch.from_numpy(action[i]),
                    "task": task,
                }
                for cam, arr in imgs.items():
                    frame[f"observation.images.{cam}"] = arr[i]
                dataset.add_frame(frame)
            dataset.save_episode()

    print(f"Done. LeRobot dataset at: {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="The root directory of the multiview data, e.g. .../data/place_can_basket/demo_multiview",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="The id of the output dataset, e.g. place_can_basket_multiview",
    )
    p.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="The name of the task, e.g. place_can_basket; will read the full_description from description/task_instruction/<task_name>.json",
    )
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--robot-type", type=str, default="aloha")
    p.add_argument("--mode", choices=("image", "video"), default="video")
    p.add_argument("--image-writer-processes", type=int, default=32)
    p.add_argument("--image-writer-threads", type=int, default=1)
    args = p.parse_args()

    task = _task_description_from_task_name(args.task_name)
    convert(
        args.root,
        args.repo_id,
        task=task,
        robot_type=args.robot_type,
        fps=args.fps,
        mode=args.mode,
        use_videos=True,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
