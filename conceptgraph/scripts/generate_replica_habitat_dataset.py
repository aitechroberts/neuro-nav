#!/usr/bin/env python3
"""
generate_replica_habitat_dataset.py

Render RGB-D + poses from Replica scenes using Habitat-Sim into the
format expected by ReplicaDataset in datasets_common.py:

  <output_root>/<scene_name>/
    results/
      frame0000.jpg
      frame0001.jpg
      ...
      depth0000.png
      depth0001.png
      ...
    traj.txt   # one 4x4 c2w per line (row-major)

You probably want to match (H, W, fx, fy, cx, cy) to your existing
Replica config_dict["camera_params"]. This script exposes H/W and the
sensor HFOV; from those you can compute intrinsics if needed.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import imageio.v2 as imageio

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

import quaternion  # numpy-quaternion


def make_simulator(scene_mesh_path: str, height: int, width: int, hfov_deg: float):
    """
    Build a Habitat-Sim Simulator with RGB + depth sensors attached to one agent.
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_mesh_path  # can be .ply
    sim_cfg.enable_physics = False

    # Sensor specs
    sensors = {}

    # RGB
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [height, width]
    rgb_spec.position = [0.0, 1.5, 0.0]  # camera height above agent position
    rgb_spec.orientation = [0.0, 0.0, 0.0]  # pitch, yaw, roll in *radians*
    rgb_spec.hfov = hfov_deg
    sensors[rgb_spec.uuid] = rgb_spec

    # Depth
    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [height, width]
    depth_spec.position = [0.0, 1.5, 0.0]
    depth_spec.orientation = [0.0, 0.0, 0.0]
    depth_spec.hfov = hfov_deg
    sensors[depth_spec.uuid] = depth_spec

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = list(sensors.values())

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    return sim


def quaternion_to_matrix(q):
    """
    Convert np.quaternion -> 3x3 rotation matrix.
    """
    if not isinstance(q, quaternion.quaternion):
        q = quaternion.quaternion(q[0], q[1], q[2], q[3])
    R = quaternion.as_rotation_matrix(q)
    return R


# ---------------------------------------------------------------------
# NEW: continuous random-walk trajectory helpers
# ---------------------------------------------------------------------
def init_random_agent_pose(sim: habitat_sim.Simulator, agent_id: int = 0):
    """
    Place the agent at a random navigable location with a random yaw and
    small random pitch. This is only called ONCE; subsequent frames use
    step_random_walk_pose to move smoothly.
    """
    pathfinder = sim.pathfinder
    pt = pathfinder.get_random_navigable_point()

    agent = sim.get_agent(agent_id)
    state = agent.get_state()

    # Agent base position is on the navmesh; sensors are offset by +1.5 m in Y.
    state.position = np.array([pt[0], pt[1], pt[2]], dtype=np.float32)

    # Random heading (yaw) around vertical axis, small pitch
    yaw_deg = np.random.uniform(-180.0, 180.0)
    pitch_deg = np.random.uniform(-10.0, 10.0)

    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    q_yaw = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    q_pitch = quat_from_angle_axis(pitch, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    state.rotation = q_yaw * q_pitch

    agent.set_state(state)
    return agent.get_state()


def step_random_walk_pose(
    sim: habitat_sim.Simulator,
    prev_state,
    step_size: float = 0.05,
    yaw_std_deg: float = 10.0,
    max_retries: int = 10,
    agent_id: int = 0,
):
    """
    Take a small forward step along the current heading on the navmesh,
    plus a small random yaw perturbation. If the forward step is not
    navigable after several retries, we respawn the agent at a new
    random navigable point.

    Returns the updated AgentState.
    """
    pathfinder = sim.pathfinder
    agent = sim.get_agent(agent_id)
    state = agent.get_state()

    # Start from the previous state's position/rotation
    state.position = np.array(prev_state.position, dtype=np.float32)
    state.rotation = prev_state.rotation

    for _ in range(max_retries):
        # Compute forward direction in world coordinates.
        # In Habitat, the camera/agent looks along -Z in its local frame,
        # so forward = -R[:, 2].
        R = quaternion_to_matrix(state.rotation)
        forward = -R[:, 2]  # shape (3,)

        target = state.position + forward * step_size

        # If forward step is navigable, move there and apply a small yaw change.
        if pathfinder.is_navigable(target):
            target = pathfinder.snap_point(target)
            state.position = target

            yaw_delta = np.deg2rad(np.random.normal(0.0, yaw_std_deg))
            q_delta = quat_from_angle_axis(
                yaw_delta, np.array([0.0, 1.0, 0.0], dtype=np.float32)
            )
            state.rotation = q_delta * state.rotation

            agent.set_state(state)
            return agent.get_state()

        # Otherwise, try a different yaw and retry
        yaw_delta = np.deg2rad(np.random.uniform(-45.0, 45.0))
        q_delta = quat_from_angle_axis(
            yaw_delta, np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )
        state.rotation = q_delta * state.rotation

    # If we failed too many times (e.g., stuck in a corner), respawn
    # somewhere else and continue.
    return init_random_agent_pose(sim, agent_id=agent_id)


# ---------------------------------------------------------------------


def get_color_sensor_pose_c2w(agent_state):
    """
    Get camera-to-world transform for the color sensor as a 4x4 matrix.
    Uses the color sensor's SixDOFPose under agent_state.sensor_states.
    """
    sensor_state = agent_state.sensor_states["color_sensor"]
    pos = np.array(sensor_state.position, dtype=np.float32)
    q = sensor_state.rotation  # np.quaternion

    R = quaternion_to_matrix(q)  # 3x3
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def render_replica_scene(
    replica_root: Path,
    scene_name: str,
    output_root: Path,
    num_frames: int = 500,
    height: int = 480,
    width: int = 640,
    hfov_deg: float = 90.0,
    depth_scale: float = 1000.0,
    seed: int = 0,
):
    """
    replica_root: path to the folder that directly contains scene dirs,
                  e.g. ~/data/Replica_raw (with apartment_1/, hotel_0/, ...)
    scene_name: e.g. "apartment_1", "hotel_0"
    output_root: base folder for ConceptGraphs-style output
    """

    np.random.seed(seed)

    scene_dir = replica_root / scene_name
    hab_dir = scene_dir / "habitat"
    scene_mesh_path = hab_dir / "mesh_semantic.ply"
    navmesh_path = hab_dir / "mesh_semantic.navmesh"

    if not scene_mesh_path.is_file():
        raise FileNotFoundError(f"Scene mesh not found: {scene_mesh_path}")

    if not navmesh_path.is_file():
        raise FileNotFoundError(f"Navmesh not found: {navmesh_path}")

    # Create simulator
    sim = make_simulator(
        str(scene_mesh_path), height=height, width=width, hfov_deg=hfov_deg
    )

    # Load navmesh for random navigable sampling
    if not sim.pathfinder.load_nav_mesh(str(navmesh_path)):
        raise RuntimeError(f"Failed to load navmesh: {navmesh_path}")

    # Prepare output dirs
    scene_out_dir = output_root / scene_name / "results"
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Trajectory generation: initialize once, then random-walk
    # -----------------------------------------------------------------
    poses = []

    agent_state = init_random_agent_pose(sim, agent_id=0)

    for idx in range(num_frames):
        # Get c2w for the color sensor at the *current* pose
        c2w = get_color_sensor_pose_c2w(agent_state)
        poses.append(c2w)

        # Render sensors at this pose
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"]  # HxWx4 uint8
        depth = obs["depth_sensor"]  # HxW float32 (meters)

        # RGB: drop alpha if present
        rgb_img = rgb[..., :3].astype(np.uint8)
        rgb_pil = Image.fromarray(rgb_img)

        # Depth: meters -> uint16 via depth_scale (e.g., 1000 for mm)
        depth_m = depth.astype(np.float32)
        depth_mm = np.clip(
            depth_m * depth_scale, 0, np.iinfo(np.uint16).max
        )
        depth_u16 = depth_mm.astype(np.uint16)

        rgb_path = scene_out_dir / f"frame{idx:04d}.jpg"
        depth_path = scene_out_dir / f"depth{idx:04d}.png"

        rgb_pil.save(rgb_path, quality=95)
        imageio.imwrite(depth_path, depth_u16)

        if (idx + 1) % 50 == 0:
            print(f"[{scene_name}] Rendered {idx+1}/{num_frames} frames")

        # Step to the next pose along a small random-walk on the navmesh
        agent_state = step_random_walk_pose(
            sim,
            agent_state,
            step_size=0.25,
            yaw_std_deg=10.0,
            max_retries=10,
            agent_id=0,
        )

    # Write traj.txt with one 4x4 matrix (row-major) per line
    traj_path = output_root / scene_name / "traj.txt"
    with open(traj_path, "w") as f:
        for T in poses:
            flat = T.reshape(-1)
            f.write(" ".join(f"{x:.6f}" for x in flat) + "\n")

    sim.close()
    print(f"[{scene_name}] Done. Wrote {num_frames} frames to {scene_out_dir}")
    print(f"[{scene_name}] Trajectory saved to {traj_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replica_root",
        type=str,
        required=True,
        help="Folder containing Replica scenes (apartment_1/, hotel_0/, ...).",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene name, e.g. apartment_1 or hotel_0.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Base output directory for ConceptGraphs-style RGB-D dataset.",
    )
    parser.add_argument("--num_frames", type=int, default=500)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--hfov_deg",
        type=float,
        default=90.0,
        help="Horizontal field-of-view for the camera.",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="Meters -> depth PNG scaling factor.",
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    replica_root = Path(args.replica_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    render_replica_scene(
        replica_root=replica_root,
        scene_name=args.scene,
        output_root=output_root,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        hfov_deg=args.hfov_deg,
        depth_scale=args.depth_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
