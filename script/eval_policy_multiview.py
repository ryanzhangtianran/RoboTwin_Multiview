"""
OOD multi-viewpoint evaluation for RoboTwin.

Randomly samples N observer camera configs from the pool (independent of
training viewpoints), then evaluates the policy on each viewpoint for
`episodes_per_view` episodes.  Each viewpoint runs in a fresh subprocess
to avoid SAPIEN multi-instance crashes.

Usage:
  python script/eval_policy_multiview.py place_can_basket_single demo_clean_side \\
      --policy_name your_policy \\
      --ckpt_setting latest \\
      --observer_configs observer_configs.json \\
      --n_views 10 \\
      --episodes_per_view 10
"""

import sys

sys.path.append("./")

import argparse
import json
import os
import random

import torch.multiprocessing as mp
import yaml

from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.task_loader import get_env_class


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _eval_worker(task_name: str, task_config_name: str, base_args: dict,
                 cam_cfg: dict, view_idx: int, result_queue):
    """Run eval_policy.main() in a fresh subprocess with a specific observer camera."""
    sys.path.append("./")
    sys.path.append("./policy")
    sys.path.append("./description/utils")

    import copy
    from script.eval_policy import main as eval_main

    args = copy.deepcopy(base_args)
    args["observer_camera"] = cam_cfg
    args["task_name"]       = task_name
    args["task_config"]     = task_config_name

    try:
        eval_main(args)
        result_queue.put((view_idx, "ok"))
    except Exception as e:
        result_queue.put((view_idx, f"error: {e}"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_base_args(task_config: str, policy_name: str, ckpt_setting: str,
                    instruction_type: str, seed: int, episodes_per_view: int) -> dict:
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_file(etype):
        return _embodiment_types[etype]["file_path"]

    def get_embodiment_config(robot_file):
        with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment_type = args["embodiment"]
    if len(embodiment_type) == 1:
        args["left_robot_file"]  = get_file(embodiment_type[0])
        args["right_robot_file"] = get_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"]  = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"]  = get_file(embodiment_type[0])
        args["right_robot_file"] = get_file(embodiment_type[1])
        args["embodiment_dis"]   = embodiment_type[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"]  = f"{embodiment_type[0]}+{embodiment_type[1]}"
    else:
        raise ValueError("embodiment should have 1 or 3 entries")

    args["left_embodiment_config"]  = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    args["policy_name"]      = policy_name
    args["ckpt_setting"]     = ckpt_setting
    args["instruction_type"] = instruction_type
    args["seed"]             = seed
    args["test_num"]         = episodes_per_view

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a policy on randomly sampled OOD observer viewpoints."
    )
    parser.add_argument("task_name",   type=str)
    parser.add_argument("task_config", type=str, help="Config name (without .yml)")
    parser.add_argument("--policy_name",       type=str, required=True)
    parser.add_argument("--ckpt_setting",      type=str, required=True)
    parser.add_argument("--instruction_type",  type=str, default="default")
    parser.add_argument("--observer_configs",  type=str, default="observer_configs.json",
                        help="Path to observer_configs.json")
    parser.add_argument("--n_views",           type=int, default=10,
                        help="Number of OOD viewpoints to sample")
    parser.add_argument("--episodes_per_view", type=int, default=10,
                        help="Episodes to run per viewpoint")
    parser.add_argument("--seed",              type=int, default=0,
                        help="Random seed for both viewpoint sampling and eval episodes")
    parsed = parser.parse_args()

    # ------------------------------------------------------------------
    # Sample OOD viewpoints (fully random, no relation to training views)
    # ------------------------------------------------------------------
    with open(parsed.observer_configs, "r") as f:
        all_cam_configs = json.load(f)

    rng = random.Random(parsed.seed)
    selected_cams = rng.sample(all_cam_configs, parsed.n_views)
    print(f"Sampled {parsed.n_views} OOD viewpoints from {len(all_cam_configs)} candidates.")

    # ------------------------------------------------------------------
    # Build base args
    # ------------------------------------------------------------------
    base_args = build_base_args(
        task_config       = parsed.task_config,
        policy_name       = parsed.policy_name,
        ckpt_setting      = parsed.ckpt_setting,
        instruction_type  = parsed.instruction_type,
        seed              = parsed.seed,
        episodes_per_view = parsed.episodes_per_view,
    )
    # Resolve task env from single_arm_tasks_multiview or envs (for multiview eval)
    base_args["env_class"] = get_env_class(parsed.task_name)

    # ------------------------------------------------------------------
    # Evaluate each viewpoint in a fresh subprocess
    # ------------------------------------------------------------------
    ctx = mp.get_context("spawn")
    summary = {}

    for view_idx, cam_cfg in enumerate(selected_cams):
        print(f"\n{'='*60}")
        print(f"OOD view {view_idx+1}/{parsed.n_views}  position={cam_cfg['position']}")
        print(f"{'='*60}")

        result_queue = ctx.Queue()
        p = ctx.Process(
            target=_eval_worker,
            args=(parsed.task_name, parsed.task_config,
                  base_args, cam_cfg, view_idx, result_queue)
        )
        p.start()
        p.join()

        if p.exitcode == 0 and not result_queue.empty():
            status = result_queue.get()[1]
        elif p.exitcode != 0:
            status = f"subprocess crashed (exitcode={p.exitcode})"
        else:
            status = "ok"

        summary[f"ood_view_{view_idx:03d}"] = status
        print(f"ood_view_{view_idx:03d}: {status}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("OOD Evaluation complete.")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved under: eval_result/{parsed.task_name}/{parsed.policy_name}/")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    from test_render import Sapien_TEST
    Sapien_TEST()
    main()
