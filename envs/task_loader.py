"""
Resolve task_name to env class. Single-arm multiview tasks live in
envs.single_arm_tasks_multiview; other tasks live in envs.
No need for per-task stub files in envs/.
"""
import importlib


def get_env_class(task_name: str):
    """
    Return the env class for task_name.
    Tries envs.single_arm_tasks_multiview.{task_name} first, then envs.{task_name}.
    """
    # Single-arm multiview tasks (e.g. place_can_basket_single) are in single_arm_tasks_multiview
    try:
        mod = importlib.import_module(f"envs.single_arm_tasks_multiview.{task_name}")
        if hasattr(mod, task_name):
            return getattr(mod, task_name)
    except ModuleNotFoundError:
        pass
    # Otherwise envs.{task_name} (e.g. place_can_basket, click_bell)
    mod = importlib.import_module(f"envs.{task_name}")
    if not hasattr(mod, task_name):
        raise AttributeError(f"No class '{task_name}' in module envs.{task_name}")
    return getattr(mod, task_name)
