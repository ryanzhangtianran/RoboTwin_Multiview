"""
Multi-viewpoint data collection for RoboTwin.

Strategy (two phases):
  Phase 1 - Seed collection (planning):
    Run the task once with observer camera disabled, collect `episodes_per_view`
    successful trajectories.  Saves seed.txt + _traj_data/ to a shared base directory.

  Phase 2 - Multi-view replay:
    For each of the N randomly sampled observer cameras, symlink seed.txt and
    _traj_data/ from the base directory into a per-view sub-directory, then replay
    the pre-planned trajectories while only the observer camera differs.
    This avoids re-planning for every viewpoint.

Directory layout produced:
  data/{task_name}/{task_config}/
    base/              <- seed.txt + _traj_data/ (shared)
    view_000/          <- episode*.hdf5 + video/ for viewpoint 0
    view_001/
    ...
    view_029/

Usage:
  python script/collect_multiview.py place_can_basket demo_multiview \\
      --num_views 30 --episodes_per_view 10 \\
      --observer_configs observer_configs.json \\
      --gpu 0
"""

import sys

sys.path.append("./")

import argparse
import copy
import importlib
import json
import os
import random

import torch.multiprocessing as mp
import yaml

from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from script.collect_data import get_embodiment_config, run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _worker(task_name: str, args: dict):
    """
    Run inside a fresh spawned subprocess to avoid SAPIEN multi-instance crashes.
    Each call creates its own renderer/physics context from scratch.
    """
    sys.path.append("./")

    try:
        mod = importlib.import_module(f"envs.{task_name}")
        env_class = getattr(mod, task_name)
        task_env = env_class()
    except (ModuleNotFoundError, AttributeError) as e:
        raise SystemExit(f"No such task: {task_name} ({e})")
    run(task_env, args)


def run_in_subprocess(task_name: str, args: dict):
    """Spawn a child process to run one collection phase and wait for it."""
    p = mp.Process(target=_worker, args=(task_name, args))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(
            f"Collection subprocess exited with code {p.exitcode} "
            f"(task={task_name}, save_path={args.get('save_path')})"
        )


def build_base_args(task_name: str, task_config: str) -> dict:
    """
    Load task config yml and resolve embodiment files.
    Embodiment configs are loaded here so the dict is fully serialisable
    and can be passed safely to spawned subprocesses.
    """
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config

    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_file(etype):
        return _embodiment_types[etype]["file_path"]

    embodiment_type = args["embodiment"]
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_file(embodiment_type[0])
        args["right_robot_file"] = get_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_file(embodiment_type[0])
        args["right_robot_file"] = get_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment_type[0]}+{embodiment_type[1]}"
    else:
        raise ValueError("embodiment should have 1 or 3 entries")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def symlink_shared_data(base_dir: str, view_dir: str):
    """Symlink seed.txt and _traj_data/ from base_dir into view_dir."""
    os.makedirs(view_dir, exist_ok=True)

    for name in ("seed.txt", "_traj_data"):
        src = os.path.abspath(os.path.join(base_dir, name))
        dst = os.path.join(view_dir, name)
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst) if os.path.isfile(dst) else None
            if os.path.islink(dst):
                os.unlink(dst)
        if os.path.exists(src):
            os.symlink(src, dst)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect multi-viewpoint observer-camera data for RoboTwin."
    )
    parser.add_argument("task_name",   type=str, help="Task class name, e.g. place_can_basket_single")
    parser.add_argument("task_config", type=str, help="Config name (without .yml), e.g. demo_clean_side")
    parser.add_argument("--num_views", type=int, default=30,
                        help="Number of observer viewpoints to sample")
    parser.add_argument("--episodes_per_view", type=int, default=10,
                        help="Successful episodes to collect per viewpoint")
    parser.add_argument("--observer_configs",  type=str, default="observer_configs.json",
                        help="Path to the observer configs JSON generated by gen_observer_configs.py")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for viewpoint sampling")
    parser.add_argument("--save_all", action="store_true",
                        help="In Phase 1, also save wrist and observer camera videos when a seed run fails")
    parsed = parser.parse_args()

    # ------------------------------------------------------------------
    # Load and sample observer configs
    # ------------------------------------------------------------------
    with open(parsed.observer_configs, "r") as f:
        all_cam_configs = json.load(f)

    rng = random.Random(parsed.seed)
    selected_cams = rng.sample(all_cam_configs, parsed.num_views)
    print(f"Sampled {parsed.num_views} viewpoints from {len(all_cam_configs)} candidates.")

    # ------------------------------------------------------------------
    # Build shared base args
    # ------------------------------------------------------------------
    base_args = build_base_args(parsed.task_name, parsed.task_config)
    root_save = os.path.join(
        base_args["save_path"], parsed.task_name, parsed.task_config
    )
    base_dir = os.path.join(root_save, "base")

    # ------------------------------------------------------------------
    # Phase 1: Seed / trajectory collection (observer camera OFF)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 1: Seed collection (planning trajectories once)")
    print(f"{'='*60}")

    seed_args = copy.deepcopy(base_args)
    seed_args["episode_num"] = parsed.episodes_per_view
    seed_args["save_path"] = base_dir
    seed_args["use_seed"] = False
    seed_args["collect_data"] = False      # only plan, don't render/save hdf5
    seed_args["data_type"]["observer_camera"] = False if not parsed.save_all else True  # enable observer when save_all to record video on fail
    seed_args["save_all"] = parsed.save_all

    run_in_subprocess(parsed.task_name, seed_args)

    # ------------------------------------------------------------------
    # Phase 2: Multi-view replay
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 2: Replaying trajectories with each observer viewpoint")
    print(f"{'='*60}")

    for view_idx, cam_cfg in enumerate(selected_cams):
        view_dir = os.path.join(root_save, f"view_{view_idx:03d}")
        print(f"\n--- View {view_idx+1}/{parsed.num_views}  "
              f"position={cam_cfg['position']} ---")

        symlink_shared_data(base_dir, view_dir)

        view_args = copy.deepcopy(base_args)
        view_args["episode_num"]              = parsed.episodes_per_view
        view_args["save_path"]                = view_dir
        view_args["use_seed"]                 = True       # replay pre-planned seeds
        view_args["collect_data"]             = True
        view_args["observer_camera"]          = cam_cfg
        view_args["data_type"]["observer_camera"] = True    # enable observer camera video

        run_in_subprocess(parsed.task_name, view_args)

    print(f"\nDone. Data saved under: {root_save}/view_*/")


if __name__ == "__main__":
    # spawn is required so each view gets a fresh SAPIEN renderer context
    mp.set_start_method("spawn", force=True)
    from test_render import Sapien_TEST
    Sapien_TEST()
    main()
