"""
Generate N valid third-person (observer) camera configurations for RoboTwin.

Visibility guarantee:
  - The robot base region is visible: key point at ROBOT_BASE_KP
  - The table workspace is visible: key point at TABLE_KP

Usage:
  python script/gen_observer_configs.py --n 50000 --out observer_configs.json
  python script/gen_observer_configs.py --n 50000 --out observer_configs.json --visualize
"""

import numpy as np
import json
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Scene constants — multiview layout (_init_multiview_task_env_)
# Table center at (0,0), robot base pose at (0,-0.70,0,0.707,0,0,0.707)
# ---------------------------------------------------------------------------
TABLE_HEIGHT = 0.74
TABLE_LENGTH_X = 1.20
TABLE_WIDTH_Y = 0.70

# Sphere shell center (table center height)
SHELL_CENTER = np.array([0.0, 0.0, 1.0])

# Robot base pose from embodiment config (x, y, z, qw, qx, qy, qz)
ROBOT_BASE_POSE = np.array([0.0, -0.70, 0.0, 0.707, 0.0, 0.0, 0.707])
ARM_XY = ROBOT_BASE_POSE[:2]

# XY footprint used for top-down base rectangle visualisation
ROBOT_BASE_LENGTH_X = 0.6
ROBOT_BASE_WIDTH_Y = 0.7

# Key point near robot base (used for visibility check)
ROBOT_BASE_KP = np.array([ARM_XY[0], ARM_XY[1], 0.25])
# Key point at the table workspace (center of table in xy, slightly above table top)
TABLE_KP = np.array([0.0, 0.3, 0.8])

# Camera image size and FOV (matching default observer camera)
IMG_W, IMG_H = 320, 240
FOVY_DEG = 93.0

# ---------------------------------------------------------------------------
# Spherical shell: camera positions on shell around (0, 0, 0.74)
# ---------------------------------------------------------------------------
SHELL_RADIUS_MIN = 0.3   # m
SHELL_RADIUS_MAX = 0.7   # m
AZIMUTH_HALF_DEG = 90.0
CAM_Y_MIN = 0.15
CAM_Z_MIN = 0.90
CAM_Z_MAX = 1.44

# How many degrees inside the FOV boundary each key point must fall
VISIBILITY_MARGIN_DEG = 8.0

def _look_at(cam_pos: np.ndarray, target: np.ndarray):
    """
    Compute (forward, left) unit vectors for a camera placed at cam_pos
    looking toward target, with world-up = [0, 0, 1].
    """
    forward = target - cam_pos
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        return None, None
    forward = forward / norm

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])

    left = np.cross(world_up, forward)
    left_norm = np.linalg.norm(left)
    if left_norm < 1e-6:
        return None, None
    left = left / left_norm
    return forward, left


def _angle_from_axis(cam_pos: np.ndarray, forward: np.ndarray,
                     point: np.ndarray) -> float:
    """Angular distance (radians) between camera forward axis and direction to point."""
    to_pt = point - cam_pos
    d = np.linalg.norm(to_pt)
    if d < 1e-6:
        return np.pi
    to_pt = to_pt / d
    cos_a = np.clip(np.dot(forward, to_pt), -1.0, 1.0)
    return np.arccos(cos_a)


def generate_configs(n: int, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)

    fovy_rad = np.deg2rad(FOVY_DEG)
    aspect = IMG_W / IMG_H
    fovx_rad = 2.0 * np.arctan(np.tan(fovy_rad / 2.0) * aspect)
    
    margin_rad = np.deg2rad(VISIBILITY_MARGIN_DEG)
    half_fovy = fovy_rad / 2.0 - margin_rad
    half_fovx = fovx_rad / 2.0 - margin_rad

    look_at_base = 0.45 * ROBOT_BASE_KP + 0.55 * TABLE_KP

    configs = []
    attempts = 0
    max_attempts = n * 30

    # Spherical shell: azimuth ±65° (130° total), polar [0, pi], r in [0.5, 0.8]
    # Azimuth 0 = +Y; x = r*sin(phi)*sin(az), y = r*sin(phi)*cos(az), z = 0.74 + r*cos(phi)
    azimuth_half_rad = np.deg2rad(AZIMUTH_HALF_DEG)

    print(f"Sampling camera configs on spherical shell (center={SHELL_CENTER}, r=[{SHELL_RADIUS_MIN},{SHELL_RADIUS_MAX}]m, "
          f"azimuth ±{AZIMUTH_HALF_DEG}°, y>{CAM_Y_MIN}, {CAM_Z_MIN}<z<={CAM_Z_MAX}, target={n}) ...")

    while len(configs) < n and attempts < max_attempts:
        attempts += 1

        # --- Sample on spherical shell around (0, 0, 0.74) ---
        r = rng.uniform(SHELL_RADIUS_MIN, SHELL_RADIUS_MAX)
        # Azimuth: x-direction angle within 130° (e.g. from +Y: -65° to +65°)
        az = rng.uniform(-azimuth_half_rad, azimuth_half_rad)
        # Polar: 0 = +Z (above center), pi = -Z (below). Use [0, pi] for full sphere
        phi = rng.uniform(0.0, np.pi)

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        # az=0 => +Y; x = r*sin(phi)*sin(az), y = r*sin(phi)*cos(az)
        dx = r * sin_phi * np.sin(az)
        dy = r * sin_phi * np.cos(az)
        dz = r * cos_phi

        cam_pos = SHELL_CENTER + np.array([dx, dy, dz])

        # Keep only y > -0.15, z in (0.8, 1.7]
        if cam_pos[1] <= CAM_Y_MIN:
            continue
        if cam_pos[2] <= CAM_Z_MIN or cam_pos[2] > CAM_Z_MAX:
            continue
        # --- Compute look-at with small random jitter on the target ---
        jitter = rng.normal(0, 0.06, size=3)
        target = look_at_base + jitter

        forward, left = _look_at(cam_pos, target)
        if forward is None:
            continue

        # --- Visibility check for both key points ---
        def visible(kp):
            to_kp = kp - cam_pos
            to_kp_norm = np.linalg.norm(to_kp)
            if to_kp_norm < 1e-6:
                return False
            to_kp_u = to_kp / to_kp_norm

            total_angle = _angle_from_axis(cam_pos, forward, kp)
            if total_angle >= max(half_fovy, half_fovx):
                return False

            up = np.cross(forward, left)
            vert_angle = abs(np.arcsin(np.clip(np.dot(to_kp_u, up), -1, 1)))
            horiz_angle = abs(np.arcsin(np.clip(np.dot(to_kp_u, -left), -1, 1)))
            return vert_angle < half_fovy and horiz_angle < half_fovx

        if not visible(ROBOT_BASE_KP) or not visible(TABLE_KP):
            continue

        configs.append({
            "position": cam_pos.round(4).tolist(),
            "forward":  forward.round(4).tolist(),
            "left":     left.round(4).tolist(),
            "fovy":     FOVY_DEG,
            "width":    IMG_W,
            "height":   IMG_H,
        })

        if len(configs) % 5000 == 0:
            print(f"  ... {len(configs)}/{n} configs generated "
                  f"(acceptance rate: {len(configs)/attempts*100:.1f}%)")

    print(f"Done: {len(configs)} configs generated in {attempts} attempts "
          f"(acceptance rate: {len(configs)/attempts*100:.1f}%)")
    return configs


def visualize_topdown(configs: list, out_path: str = "observer_configs_preview.png"):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Wedge, Rectangle
    except ImportError:
        print("matplotlib not available, skipping visualisation.")
        return

    all_pos = np.array([c["position"] for c in configs])

    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_aspect("equal")

    # ---------- table rectangle ----------
    table_rect = Rectangle(
        (-TABLE_LENGTH_X / 2.0, -TABLE_WIDTH_Y / 2.0),
        TABLE_LENGTH_X,
        TABLE_WIDTH_Y,
        linewidth=2,
        edgecolor="#8B6914",
        facecolor="#F5DEB3",
        alpha=0.45,
        label="Table",
        zorder=1,
    )
    ax.add_patch(table_rect)

    # ---------- all camera positions ----------
    if len(all_pos) > 0:
        sc = ax.scatter(
            all_pos[:, 0], all_pos[:, 1],
            c=all_pos[:, 2], cmap="plasma",
            s=6, alpha=0.35, zorder=3, label=f"Camera ({len(all_pos)} pts)",
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Camera height Z (m)", fontsize=10)

    # ---------- Spherical shell sector (top-down: circle sector, r 0.5~0.8, azimuth ±65°) ----------
    fo = (SHELL_CENTER[0], SHELL_CENTER[1])
    fan_rad = np.deg2rad(AZIMUTH_HALF_DEG)
    for sign in (+1, -1):
        angle = np.pi * 0.5 + sign * fan_rad
        dx_in = SHELL_RADIUS_MIN * np.cos(angle)
        dy_in = SHELL_RADIUS_MIN * np.sin(angle)
        dx_out = SHELL_RADIUS_MAX * np.cos(angle)
        dy_out = SHELL_RADIUS_MAX * np.sin(angle)
        ax.plot(
            [fo[0] + dx_in, fo[0] + dx_out],
            [fo[1] + dy_in, fo[1] + dy_out],
            color="orange", linewidth=1.5, linestyle="--", zorder=4,
        )

    wedge = Wedge(
        center=(fo[0], fo[1]),
        r=SHELL_RADIUS_MAX,
        theta1=90 - AZIMUTH_HALF_DEG,
        theta2=90 + AZIMUTH_HALF_DEG,
        width=SHELL_RADIUS_MAX - SHELL_RADIUS_MIN,
        facecolor="orange", alpha=0.08, zorder=2,
    )
    ax.add_patch(wedge)

    ax.text(fo[0], fo[1] + SHELL_RADIUS_MAX + 0.1,
            f"Shell R:{SHELL_RADIUS_MIN}-{SHELL_RADIUS_MAX}m\n±{AZIMUTH_HALF_DEG:.0f}° (y>{CAM_Y_MIN}, {CAM_Z_MIN}<z<={CAM_Z_MAX})",
            ha="center", va="bottom", fontsize=9, color="darkorange")

    # ---------- robot base rectangle ----------
    base_rect = Rectangle(
        (ARM_XY[0] - ROBOT_BASE_LENGTH_X / 2.0, ARM_XY[1] - ROBOT_BASE_WIDTH_Y / 2.0),
        ROBOT_BASE_LENGTH_X,
        ROBOT_BASE_WIDTH_Y,
        linewidth=2,
        edgecolor="#8B0000",
        facecolor="#FF6B6B",
        alpha=0.30,
        label="Robot base",
        zorder=5,
    )
    ax.add_patch(base_rect)
    ax.scatter(*ARM_XY, s=80, c="red", marker="o", zorder=6)
    ax.annotate("Arm base (0, -0.65)", ARM_XY, textcoords="offset points",
                xytext=(8, 6), fontsize=9, color="red")

    # ---------- axes & decorations ----------
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_title(
        f"Observer camera positions — top-down view (spherical shell)\n"
        f"N={len(all_pos)},  center (0,0,0.74),  R=[{SHELL_RADIUS_MIN},{SHELL_RADIUS_MAX}]m,  "
        f"azimuth ±{AZIMUTH_HALF_DEG:.0f}°,  y>{CAM_Y_MIN},  {CAM_Z_MIN}<z<={CAM_Z_MAX}",
        fontsize=12,
    )
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.2, 2.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Top-down preview saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int,  default=50000,
                        help="Number of configs to generate")
    parser.add_argument("--out",        type=str,  default="observer_configs.json",
                        help="Output JSON file path")
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--visualize",  action="store_true",
                        help="Show 3-D scatter of camera positions")
    args = parser.parse_args()

    configs = generate_configs(args.n, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(configs, f)
    print(f"Saved {len(configs)} configs → {out_path}")

    if args.visualize:
        visualize_topdown(configs, out_path="observer_configs_preview.png")
        