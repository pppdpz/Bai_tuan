# 4_motion_planning_baituan.py (精简后)
"""UR5e + Robotiq 2F-85 motion planning demo with inline MuJoCo scene."""

from __future__ import annotations

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import os
from typing import Literal

import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.scene import SceneCfg
from metasim.utils.configclass import configclass
from metasim.utils.obs_utils import ObsSaver
from metasim.utils.setup_util import get_handler

# Import and install the adapter BEFORE get_handler()
from inline_scene_adapter import install_inline_scene_patches


@configclass
class BaituanSceneCfg(SceneCfg):
    def __post_init__(self):
        if self.mjx_mjcf_path is None:
            self.mjx_mjcf_path = self.mjcf_path


@configclass
class Args:
    robot: str = "ur5e_2f85"
    sim: Literal["isaacsim", "isaacgym", "genesis", "pybullet",
                 "sapien2", "sapien3", "mujoco", "mjx"] = "mujoco"
    num_envs: int = 1
    headless: bool = True
    solver: Literal["curobo", "pyroki"] = "pyroki"

    def __post_init__(self):
        log.info(f"Args: {self}")


args = tyro.cli(Args)
log.info(f"Using IK solver: {args.solver}")

# Install patches once
install_inline_scene_patches()

scenario = ScenarioCfg(
    scene=BaituanSceneCfg(
        name="baituan_satellite",
        mjcf_path="asset_baituan/example_scene_y.xml",
    ),
    robots=["ur5e_2f85"],
    objects=[],
    simulator=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
    gravity=(0.0, 0.0, 0.0),
)

scenario.cameras = [PinholeCameraCfg(
    width=1024, height=1024,
    pos=(4.0, -4.0, 4.0),
    look_at=(0.0, 0.0, 0.0),
)]

handler = get_handler(scenario)

# UR5e initial joint config
robot_dict = {
    "ur5e_2f85": {
        "pos": torch.tensor([0.0, 0.0, 0.0]),
        "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "dof_pos": {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "elbow_joint": 1.5708,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 4.7124,
            "wrist_3_joint": -1.5708,
            "right_driver_joint": 0.0, "right_coupler_joint": 0.0,
            "right_spring_link_joint": 0.0, "right_follower_joint": 0.0,
            "left_driver_joint": 0.0, "left_coupler_joint": 0.0,
            "left_spring_link_joint": 0.0, "left_follower_joint": 0.0,
        },
    }
}

init_states = [{"objects": {}, "robots": robot_dict} for _ in range(args.num_envs)]

robot = scenario.robots[0]
from metasim.utils.ik_solver import process_gripper_command, setup_ik_solver
ik_solver = setup_ik_solver(robot, args.solver)

handler.set_states(init_states)
obs = handler.get_states(mode="tensor")
os.makedirs("get_started/output", exist_ok=True)

obs_saver = ObsSaver(video_path=f"get_started/output/4_motion_planning_{args.sim}.mp4")
obs_saver.add(obs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for step in range(200):
    log.debug(f"Step {step}")
    states = handler.get_states()

    t = step / 100.0
    ee_pos_target = torch.tensor([[0.3 + 0.1 * t, 0.0, 0.5]], device=device).repeat(args.num_envs, 1)
    ee_quat_target = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device).repeat(args.num_envs, 1)

    reorder_idx = handler.get_joint_reindex(robot.name)
    inv_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
    curr_q = obs.robots[robot.name].joint_pos[:, inv_idx]

    q_sol, _ = ik_solver.solve_ik_batch(ee_pos_target, ee_quat_target, curr_q)

    gripper_widths = process_gripper_command(
        torch.ones(args.num_envs, device=device), robot, device
    )

    actions = ik_solver.compose_joint_action(q_sol, gripper_widths, curr_q, return_dict=True)

    handler.set_dof_targets(actions)
    handler.simulate()
    obs = handler.get_states(mode="tensor")
    obs_saver.add(obs)

obs_saver.save()