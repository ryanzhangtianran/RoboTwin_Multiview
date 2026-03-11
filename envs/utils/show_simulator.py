# python envs/utils/show_simulator.py --num_views n

import sys
import os
import argparse
import json
import yaml
import numpy as np
import sapien.core as sapien
from PIL import Image

# Append project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.task_loader import get_env_class
from script.collect_data import get_embodiment_config

TABLE_XY_BIAS_DEFAULT = [0, 0]           # _init_task_env_(table_xy_bias=...)
TABLE_HEIGHT_BIAS_DEFAULT = 0             # _init_task_env_(table_height_bias=...)


def get_default_domain_randomization():
    """与 _init_task_env_ 中 random_setting 的默认键一致。"""
    return {
        "random_background": False,
        "fixed_background_id": None,
        "cluttered_table": False,
        "cluttered_numbers": 10,
        "clean_background_rate": 1,
        "random_head_camera_dis": 0,
        "random_table_height": 0,
        "random_light": False,
        "crazy_random_light_rate": 0,
        "random_embodiment": False,
    }


def build_simulator_init_args(
    task_name: str,
    task_config: str,
    seed: int = 0,
    render_freq: int = 0,
    table_xy_bias=None,
    table_height_bias: float = TABLE_HEIGHT_BIAS_DEFAULT,
) -> dict:
    """
    构建与 _multiview_task._init_task_env_(**kwags) 一致的初始化参数：
    setup_scene -> create_table_and_wall -> load_robot -> load_camera -> load_actors 等所需字段均包含。
    可视化模式：need_plan=False, collect_data=False，从而跳过稳定性检查。
    render_freq=0 时不创建 Viewer（无头/SSH 可用）；render_freq>0 时创建 Viewer（需有显示器，材质与光追更完整）。
    table_xy_bias / table_height_bias 与 _multiview_task 缺省一致。
    """
    if table_xy_bias is None:
        table_xy_bias = list(TABLE_XY_BIAS_DEFAULT)

    # 1) 从 task_config yml 加载并补全 embodiment / robot / camera 相关
    args = build_base_args(task_name, task_config)

    # 2) _init_task_env_ 显式使用的字段（与 _multiview_task.py 42-110 行对应）
    defaults = {
        "seed": seed,
        "task_name": task_name,
        "save_path": args.get("save_path", "data"),
        "now_ep_num": 0,
        "render_freq": render_freq,
        "data_type": args.get("data_type") or {"observer_camera": True},
        "save_data": False,
        "dual_arm": True,
        "eval_mode": False,
        "need_plan": False,
        "collect_data": False,
        "save_freq": args.get("save_freq", 15),
        "eval_video_save_dir": None,
        "left_joint_path": [],
        "right_joint_path": [],
        "table_xy_bias": table_xy_bias,
        "table_height_bias": table_height_bias,
    }
    dr_default = get_default_domain_randomization()
    defaults["domain_randomization"] = {**dr_default, **args.get("domain_randomization", {})}

    # 3) 合并：defaults 打底，args（yml + embodiment）覆盖
    for k, v in defaults.items():
        if k not in args:
            args[k] = v
        elif k == "domain_randomization":
            args[k] = {**dr_default, **v}
    args["need_plan"] = False
    args["collect_data"] = False
    args["render_freq"] = render_freq
    if "data_type" not in args:
        args["data_type"] = {}
    args["data_type"]["observer_camera"] = True
    return args


def build_base_args(task_name: str, task_config: str) -> dict:
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

def main():
    parser = argparse.ArgumentParser(description="Visualize simulator and save images from multi-view configs without action planning.")
    parser.add_argument("--task_name", type=str, default="place_can_basket")
    parser.add_argument("--task_config", type=str, default="demo_multiview")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for env init (same as _init_task_env_)")
    parser.add_argument("--observer_configs", type=str, default="observer_configs.json")
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--num_views", type=int, default=10, help="Number of views to render")
    parser.add_argument("--run_demo", action="store_true", help="Run play_once() so arms move, then capture multi-view (saved images show post-demo state)")
    parser.add_argument("--render_freq", type=int, default=0, help="If >0, create Viewer (requires display); 0=headless, no window (default)")
    parser.add_argument("--table_xy_bias", type=str, default=",".join(map(str, TABLE_XY_BIAS_DEFAULT)), help="Table xy bias 'x,y'")
    parser.add_argument("--table_height_bias", type=float, default=TABLE_HEIGHT_BIAS_DEFAULT, help="Table height bias (m)")
    parsed = parser.parse_args()

    # 保证工作目录为项目根，否则 ./assets/ 下的材质路径无法加载
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    os.chdir(project_root)

    os.makedirs(parsed.output_dir, exist_ok=True)

    with open(parsed.observer_configs, "r") as f:
        all_cam_configs = json.load(f)

    print(f"Loaded {len(all_cam_configs)} camera configs.")

    # 解析桌面位置（与 _multiview_task 一致）
    table_xy_bias = [float(x.strip()) for x in parsed.table_xy_bias.split(",")]
    if len(table_xy_bias) != 2:
        raise ValueError("--table_xy_bias must be two numbers, e.g. '0,0' or '0.3,0'")

    # 与 _init_task_env_ 一致的初始化参数（默认 render_freq=0 无头，无显示器时避免 Create window failed）
    args = build_simulator_init_args(
        parsed.task_name,
        parsed.task_config,
        seed=parsed.seed,
        render_freq=parsed.render_freq,
        table_xy_bias=table_xy_bias,
        table_height_bias=parsed.table_height_bias,
    )

    # Load task class (single-arm multiview from envs.single_arm_tasks_multiview, else envs.{task_name})
    env_class = get_env_class(parsed.task_name)
    env = env_class()
    # 部分任务会显式传 table_xy_bias（如 put_bottles_dustbin_single），若 args 里也有会报 multiple values
    args_for_setup = {k: v for k, v in args.items() if k not in ("table_xy_bias", "table_height_bias")}
    env.setup_demo(**args_for_setup)

    if parsed.run_demo:
        print("Running demo (play_once)...")
        env.play_once()
        env.scene.step()
        env._update_render()
        print("Demo finished, capturing multi-view.")

    # Get the observer camera added in Camera.load_camera()
    observer_camera = env.cameras.observer_camera

    rendered_count = 0
    for i, cam_cfg in enumerate(all_cam_configs[:parsed.num_views]):
        # Construct camera pose from JSON
        cam_pos = np.array(cam_cfg["position"])
        cam_forward = np.array(cam_cfg["forward"])
        cam_left = np.array(cam_cfg["left"])
        
        cam_forward = cam_forward / np.linalg.norm(cam_forward)
        cam_left = cam_left / np.linalg.norm(cam_left)
        
        up = np.cross(cam_forward, cam_left)
        
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([cam_forward, cam_left, up], axis=1)
        mat44[:3, 3] = cam_pos
        
        observer_camera.entity.set_pose(sapien.Pose(mat44))
        
        # Step the scene and update render buffer
        env.scene.step()
        env._update_render()
        
        # Render the image from observer_camera
        rgb = env.cameras.get_observer_rgb()
        
        # Save image
        img = Image.fromarray(rgb)
        save_path = os.path.join(parsed.output_dir, f"view_{i:03d}.png")
        img.save(save_path)
        print(f"Saved {save_path}")
        rendered_count += 1
        
    print(f"Successfully visualized and saved {rendered_count} images in {parsed.output_dir}/")

if __name__ == "__main__":
    # Required to avoid SAPIEN crash if multiple SAPIEN instances try to access OpenGL context
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
