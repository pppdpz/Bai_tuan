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

import datetime as _dt

_log_dir = "get_started/output"
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(
    _log_dir,
    f"4_motion_planning_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)
log.add(_log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="DEBUG")



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
    # gravity=(0.0, 0.0, -9.81),
)

scenario.cameras = [PinholeCameraCfg(
    width=1024, height=1024,
    # pos=(4.0, -4.0, 4.0),
    # look_at=(0.0, 0.0, 0.0),

    # # 保持类似角度但拉近，看到整星 + 机械臂
    # pos=(3.0, -2.5, 2.5),
    # look_at=(1.0, 0.0, 0.5),

    # pos=(-4.0, -4.0, 4.0),
    pos=(4.0, 4.0, 3.0),
    look_at=(0.0, 0.0, 0.0),


    # # 从前侧偏上方看，聚焦机械臂末端和卫星碎片
    # pos=(2.5, -1.5, 2.0),
    # look_at=(1.0, 0.3, 0.7),

    # # 从 Y 负方向平视
    # pos=(1.0, -2.5, 1.2),
    # look_at=(1.0, 0.0, 0.7),

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


# === 折叠小卫星太阳翼 ===
import mujoco as _mj

_model = handler.physics.model.ptr  # 原生 MjModel
_data = handler.physics.data.ptr    # 原生 MjData

_fold_angles = {
    "hinge_1": -1.5708, "hinge_2": 3.1416,
    "hinge_3": -3.1416, "hinge_4": 1.5708,
    "hinge_1_L": -1.5708, "hinge_2_L": 3.1416,
    "hinge_3_L": -3.1416, "hinge_4_L": 1.5708,
}

for _jname, _angle in _fold_angles.items():
    _jid = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_JOINT, _jname)
    if _jid >= 0:
        _data.qpos[_model.jnt_qposadr[_jid]] = _angle

_mj.mj_forward(_model, _data)

obs = handler.get_states(mode="tensor")
os.makedirs("get_started/output", exist_ok=True)

obs_saver = ObsSaver(video_path=f"get_started/output/4_motion_planning_{args.sim}.mp4")
obs_saver.add(obs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for step in range(200):
#     log.debug(f"Step {step}")
#     states = handler.get_states()

#     t = step / 100.0
#     ee_pos_target = torch.tensor([[0.3 + 0.1 * t, 0.0, 0.5]], device=device).repeat(args.num_envs, 1)
#     ee_quat_target = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device).repeat(args.num_envs, 1)

#     reorder_idx = handler.get_joint_reindex(robot.name)
#     inv_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]
#     curr_q = obs.robots[robot.name].joint_pos[:, inv_idx]

#     q_sol, _ = ik_solver.solve_ik_batch(ee_pos_target, ee_quat_target, curr_q)

#     gripper_widths = process_gripper_command(
#         torch.ones(args.num_envs, device=device), robot, device
#     )

#     actions = ik_solver.compose_joint_action(q_sol, gripper_widths, curr_q, return_dict=True)

#     handler.set_dof_targets(actions)
#     handler.simulate()
#     obs = handler.get_states(mode="tensor")
#     obs_saver.add(obs)

# obs_saver.save()

# ============================================================
# 坐标系定义
# ============================================================
# # 机械臂基座世界坐标（carrier pos + mount pos）
# BASE_WORLD = torch.tensor([1.0, 0.0, 0.895], device=device)

# # 帆板 site 世界坐标
# SITE_FOLDED_W   = torch.tensor([1.5, 0.24, 1.38], device=device)   # 折叠后
# SITE_UNFOLDED_W = torch.tensor([1.5, -0.30, 1.20], device=device)  # 完全展开

# # 转换为基座局部坐标（IK 求解器使用）
# SITE_FOLDED   = SITE_FOLDED_W - BASE_WORLD     # (-0.5, 0.24, 0.485)
# SITE_UNFOLDED = SITE_UNFOLDED_W - BASE_WORLD   # (-0.5, -0.30, 0.305)

# ============================================================
# 帆板选择：左右独立参数
# ============================================================
SIDE = "left"   # "right" 或 "left"
log.info(f"======== 帆板选择: {SIDE} ========")

# 1) 读取机械臂基座世界坐标
_base_bid = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_BODY, "ur5e_base")
assert _base_bid >= 0, "找不到 body 'ur5e_base'"
BASE_WORLD = torch.tensor(_data.xpos[_base_bid].copy(), device=device, dtype=torch.float32)

# 2) 根据 SIDE 选择对应的 site 和参数
if SIDE == "right":
    _tip_sid = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_SITE, "panel4_tip")
    assert _tip_sid >= 0, "找不到 site 'panel4_tip'"
    # 右侧：夹爪朝下抓取
    EE_QUAT = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
    PRE_GRASP_OFFSET = torch.tensor([0.0, 0.08, 0.0], device=device)   # Y+ 方向偏移
    WAYPOINT = torch.tensor([0.4, 0.15, 0.45], device=device)
    _unfold_joints = {k: v for k, v in _fold_angles.items() if not k.endswith("_L")}
else:  # left
    _tip_sid = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_SITE, "panel4_tip_L")
    assert _tip_sid >= 0, "找不到 site 'panel4_tip_L'"
    # 左侧：夹爪水平朝 -Y 方向抓取，从 +Y 侧接近（绕X轴旋转90°）
    EE_QUAT = torch.tensor([[0.707, 0.707, 0.0, 0.0]], device=device)
    PRE_GRASP_OFFSET = torch.tensor([0.0, 0.08, 0.0], device=device)   # Y+ 偏移（从 +Y 侧接近）
    WAYPOINT = torch.tensor([0.4, 0.2, 0.45], device=device)
    _unfold_joints = {k: v for k, v in _fold_angles.items() if k.endswith("_L")}

# 3) 读取折叠后的世界坐标（当前 qpos 已经是折叠状态）
SITE_FOLDED_W = torch.tensor(_data.site_xpos[_tip_sid].copy(), device=device, dtype=torch.float32)

# 4) 临时将帆板关节归零 → 读取展开后的世界坐标 → 恢复折叠状态
_saved_qpos = _data.qpos.copy()

for _jname in _fold_angles:
    _jid = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_JOINT, _jname)
    if _jid >= 0:
        _data.qpos[_model.jnt_qposadr[_jid]] = 0.0
_mj.mj_forward(_model, _data)

SITE_UNFOLDED_W = torch.tensor(_data.site_xpos[_tip_sid].copy(), device=device, dtype=torch.float32)

_data.qpos[:] = _saved_qpos
_mj.mj_forward(_model, _data)

# 5) 验证：读取 tcp_link body 交叉检查基座坐标转换
_tcp_body_id = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_BODY, "tcp_link")
if _tcp_body_id >= 0:
    _tcp_world = _data.xpos[_tcp_body_id]
    log.info(f"[验证] TCP (tcp_link) 世界坐标: [{_tcp_world[0]:.4f}, {_tcp_world[1]:.4f}, {_tcp_world[2]:.4f}]")
    log.info(f"[验证] TCP (tcp_link) 基座坐标: [{_tcp_world[0]-BASE_WORLD[0].item():.4f}, "
             f"{_tcp_world[1]-BASE_WORLD[1].item():.4f}, {_tcp_world[2]-BASE_WORLD[2].item():.4f}]")

# 转换为基座局部坐标（IK 求解器使用）
SITE_FOLDED   = SITE_FOLDED_W - BASE_WORLD
SITE_UNFOLDED = SITE_UNFOLDED_W - BASE_WORLD

# 预接近点：根据 SIDE 选择的偏移方向
PRE_GRASP = SITE_FOLDED.clone()
PRE_GRASP += PRE_GRASP_OFFSET

# 初始位置（实际初始 TCP 相对于基座的坐标，保证平滑启动）
_tcp_body_init = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_BODY, "tcp_link")
_init_tcp_w = torch.tensor(_data.xpos[_tcp_body_init].copy(), device=device, dtype=torch.float32)
HOME_POS = _init_tcp_w - BASE_WORLD

_init_quat_w = torch.tensor(_data.xquat[_tcp_body_init].copy(), device=device, dtype=torch.float32)
HOME_QUAT = _init_quat_w.unsqueeze(0)  # [1, 4]

# 计算 wrist 到 tcp 的 offset 和相对旋转，以便在 IK 时将目标 tcp 转换为 wrist 目标
from metasim.utils.math import matrix_from_quat
import math

_wrist_body_init = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
wrist_pos_t = torch.tensor(_data.xpos[_wrist_body_init].copy(), device=device, dtype=torch.float32)
tcp_pos_t = torch.tensor(_data.xpos[_tcp_body_init].copy(), device=device, dtype=torch.float32)
wrist_mat_t = torch.tensor(_data.xmat[_wrist_body_init].copy().reshape(3, 3), device=device, dtype=torch.float32)
tcp_mat_t = torch.tensor(_data.xmat[_tcp_body_init].copy().reshape(3, 3), device=device, dtype=torch.float32)

p_tcplocal = wrist_mat_t.T @ (tcp_pos_t - wrist_pos_t)
R_tcplocal = wrist_mat_t.T @ tcp_mat_t
log.info(f"p_tcplocal: {p_tcplocal.tolist()}")

def safe_quat_from_matrix(R):
    B = R.size(0)
    q = torch.zeros((B, 4), device=R.device, dtype=R.dtype)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for i in range(B):
        if tr[i] > 0:
            S = torch.sqrt(tr[i] + 1.0) * 2
            q[i, 0] = 0.25 * S
            q[i, 1] = (R[i, 2, 1] - R[i, 1, 2]) / S
            q[i, 2] = (R[i, 0, 2] - R[i, 2, 0]) / S
            q[i, 3] = (R[i, 1, 0] - R[i, 0, 1]) / S
        elif (R[i, 0, 0] > R[i, 1, 1]) and (R[i, 0, 0] > R[i, 2, 2]):
            S = torch.sqrt(1.0 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2]) * 2
            q[i, 0] = (R[i, 2, 1] - R[i, 1, 2]) / S
            q[i, 1] = 0.25 * S
            q[i, 2] = (R[i, 0, 1] + R[i, 1, 0]) / S
            q[i, 3] = (R[i, 0, 2] + R[i, 2, 0]) / S
        elif R[i, 1, 1] > R[i, 2, 2]:
            S = torch.sqrt(1.0 + R[i, 1, 1] - R[i, 0, 0] - R[i, 2, 2]) * 2
            q[i, 0] = (R[i, 0, 2] - R[i, 2, 0]) / S
            q[i, 1] = (R[i, 0, 1] + R[i, 1, 0]) / S
            q[i, 2] = 0.25 * S
            q[i, 3] = (R[i, 1, 2] + R[i, 2, 1]) / S
        else:
            S = torch.sqrt(1.0 + R[i, 2, 2] - R[i, 0, 0] - R[i, 1, 1]) * 2
            q[i, 0] = (R[i, 1, 0] - R[i, 0, 1]) / S
            q[i, 1] = (R[i, 0, 2] + R[i, 2, 0]) / S
            q[i, 2] = (R[i, 1, 2] + R[i, 2, 1]) / S
            q[i, 3] = 0.25 * S
    return q

def tcp_to_wrist(tcp_pos_b: torch.Tensor, tcp_quat_b: torch.Tensor):
    B = tcp_pos_b.size(0)
    R_tcp = matrix_from_quat(tcp_quat_b)
    R_tcplocal_t_batch = R_tcplocal.T.unsqueeze(0).expand(B, 3, 3)
    R_wrist = torch.bmm(R_tcp, R_tcplocal_t_batch)
    p_local_batch = p_tcplocal.unsqueeze(0).unsqueeze(2).expand(B, 3, 1)
    offset_world = torch.bmm(R_wrist, p_local_batch).squeeze(-1)
    wrist_pos_b = tcp_pos_b - offset_world
    wrist_quat_b = safe_quat_from_matrix(R_wrist)
    return wrist_pos_b, wrist_quat_b

# 确保四元数插值走最短路径
if (HOME_QUAT * EE_QUAT).sum() < 0:
    EE_QUAT = -EE_QUAT

# 记录初始关节角 (转换为 IK solver 期待的 dict 模型顺序)
_joint_names_dict_order = list(ik_solver.robot_cfg.actuators.keys())
_joint_names_alpha_order = sorted(_joint_names_dict_order)
_alpha_to_dict_idx = [_joint_names_alpha_order.index(name) for name in _joint_names_dict_order]
INIT_JOINT_POS = obs.robots[robot.name].joint_pos[:, _alpha_to_dict_idx].clone()

# ============================================================
# 阶段步数
# ============================================================
N_APPROACH  = 100   # HOME → WAYPOINT → PRE_GRASP
N_FINAL_APP = 40    # PRE_GRASP → SITE_FOLDED
N_CLOSE     = 30    # 闭合夹爪
N_UNFOLD    = 200   # 拉开帆板
N_RELEASE   = 20    # 松开夹爪
N_RETURN    = 100   # 回到初始位姿
SIM_SUBSTEPS = 10   # 每个控制步的仿真子步数（让 PD 伺服收敛）
TOTAL = N_APPROACH + N_FINAL_APP + N_CLOSE + N_UNFOLD + N_RELEASE + N_RETURN

# 预计算展开过程中帆板尖端的弧线轨迹（基座坐标系）
_saved_qpos2 = _data.qpos.copy()
UNFOLD_TIP_TRAJ = []
for _i in range(N_UNFOLD + 1):
    _alpha = _i / N_UNFOLD
    for _jname, _fangle in _unfold_joints.items():
        _jid = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_JOINT, _jname)
        if _jid >= 0:
            _data.qpos[_model.jnt_qposadr[_jid]] = _fangle * (1.0 - _alpha)
    _mj.mj_forward(_model, _data)
    _tip_w = torch.tensor(_data.site_xpos[_tip_sid].copy(), device=device, dtype=torch.float32)
    UNFOLD_TIP_TRAJ.append(_tip_w - BASE_WORLD)
_data.qpos[:] = _saved_qpos2
_mj.mj_forward(_model, _data)
log.info(f"帆板轨迹: 起点={UNFOLD_TIP_TRAJ[0].tolist()}, 终点={UNFOLD_TIP_TRAJ[-1].tolist()}")

log.info(f"总步数: {TOTAL}")
log.info(f"基座世界坐标: {BASE_WORLD.tolist()}")
log.info(f"Site 折叠(基座): {SITE_FOLDED.tolist()}")
log.info(f"Site 展开(基座): {SITE_UNFOLDED.tolist()}")

for step in range(TOTAL):
    # 获取最新关节状态作为 IK seed，并严格按照 actuators_dict 顺序排列
    obs = handler.get_states(mode="tensor")
    curr_q = obs.robots[robot.name].joint_pos[:, _alpha_to_dict_idx]


    # ----------------------------------------------------------
    # 阶段 1：HOME → WAYPOINT → PRE_GRASP
    # ----------------------------------------------------------
    if step < N_APPROACH:
        alpha = step / N_APPROACH
        if alpha < 0.5:
            t = alpha * 2.0
            ee_pos = (HOME_POS + t * (WAYPOINT - HOME_POS)).unsqueeze(0)
            # 平滑插值姿态
            q_interp = (1.0 - t) * HOME_QUAT + t * EE_QUAT
            q_interp = q_interp / torch.linalg.norm(q_interp, dim=1, keepdim=True)
            ee_quat_target = q_interp.repeat(args.num_envs, 1)
        else:
            t = (alpha - 0.5) * 2.0
            ee_pos = (WAYPOINT + t * (PRE_GRASP - WAYPOINT)).unsqueeze(0)
            ee_quat_target = EE_QUAT.repeat(args.num_envs, 1)

        ee_pos_target = ee_pos.repeat(args.num_envs, 1)
        gripper_cmd = torch.ones(args.num_envs, device=device)

    # ----------------------------------------------------------
    # 阶段 2：PRE_GRASP → SITE_FOLDED
    # ----------------------------------------------------------
    elif step < N_APPROACH + N_FINAL_APP:
        local_step = step - N_APPROACH
        alpha = local_step / N_FINAL_APP
        ee_pos = (PRE_GRASP + alpha * (SITE_FOLDED - PRE_GRASP)).unsqueeze(0)
        ee_pos_target = ee_pos.repeat(args.num_envs, 1)
        ee_quat_target = EE_QUAT.repeat(args.num_envs, 1)
        gripper_cmd = torch.ones(args.num_envs, device=device)

    # ----------------------------------------------------------
    # 阶段 3：闭合夹爪
    # ----------------------------------------------------------
    elif step < N_APPROACH + N_FINAL_APP + N_CLOSE:
        ee_pos_target = SITE_FOLDED.unsqueeze(0).repeat(args.num_envs, 1)
        ee_quat_target = EE_QUAT.repeat(args.num_envs, 1)
        gripper_cmd = torch.zeros(args.num_envs, device=device)

    # ----------------------------------------------------------
    # 阶段 4：拉开帆板
    # ----------------------------------------------------------
    elif step < N_APPROACH + N_FINAL_APP + N_CLOSE + N_UNFOLD:
        local_step = step - (N_APPROACH + N_FINAL_APP + N_CLOSE)
        alpha = local_step / N_UNFOLD
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3

        # 使用预计算的帆板尖端弧线轨迹（沿铰链实际运动路径）
        traj_idx = min(int(alpha_smooth * N_UNFOLD), N_UNFOLD)
        ee_pos = UNFOLD_TIP_TRAJ[traj_idx].unsqueeze(0)
        ee_pos_target = ee_pos.repeat(args.num_envs, 1)
        ee_quat_target = EE_QUAT.repeat(args.num_envs, 1)
        gripper_cmd = torch.zeros(args.num_envs, device=device)

    # ----------------------------------------------------------
    # 阶段 5：松开夹爪
    # ----------------------------------------------------------
    elif step < N_APPROACH + N_FINAL_APP + N_CLOSE + N_UNFOLD + N_RELEASE:
        ee_pos_target = SITE_UNFOLDED.unsqueeze(0).repeat(args.num_envs, 1)
        ee_quat_target = EE_QUAT.repeat(args.num_envs, 1)
        gripper_cmd = torch.ones(args.num_envs, device=device)

    # ----------------------------------------------------------
    # 阶段 6：回到初始位姿
    # ----------------------------------------------------------
    else:
        local_step = step - (N_APPROACH + N_FINAL_APP + N_CLOSE + N_UNFOLD + N_RELEASE)
        alpha = local_step / N_RETURN
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3

        if alpha_smooth < 0.5:
            t = alpha_smooth * 2.0
            ee_pos = (SITE_UNFOLDED + t * (WAYPOINT - SITE_UNFOLDED)).unsqueeze(0)
            ee_quat_target = EE_QUAT.repeat(args.num_envs, 1)
        else:
            t = (alpha_smooth - 0.5) * 2.0
            ee_pos = (WAYPOINT + t * (HOME_POS - WAYPOINT)).unsqueeze(0)
            q_interp = (1.0 - t) * EE_QUAT + t * HOME_QUAT
            q_interp = q_interp / torch.linalg.norm(q_interp, dim=1, keepdim=True)
            ee_quat_target = q_interp.repeat(args.num_envs, 1)

        ee_pos_target = ee_pos.repeat(args.num_envs, 1)
        gripper_cmd = torch.ones(args.num_envs, device=device)

    # ----------------------------------------------------------
    # 公共部分：IK 求解 → 执行 → 录制
    # ----------------------------------------------------------
    ee_pos_target_wrist, ee_quat_target_wrist = tcp_to_wrist(ee_pos_target, ee_quat_target)
    if step < 5:
        log.info(f"Step {step}: Target TCP(Base) = {ee_pos_target[0].tolist()}, Target WRIST(Base) = {ee_pos_target_wrist[0].tolist()}")
    q_sol, succ = ik_solver.solve_ik_batch(ee_pos_target_wrist, ee_quat_target_wrist, curr_q)
    if step < 5:
        log.info(f"curr_q: {curr_q[0, :6].tolist()}")
        log.info(f"q_sol: {q_sol[0].tolist()}")
    if not succ[0]:
        log.warning(f"Step {step}: IK 求解失败!")

    gripper_widths = process_gripper_command(gripper_cmd, robot, device)
    actions = ik_solver.compose_joint_action(q_sol, gripper_widths, curr_q, return_dict=True)

    log.info(f"actions keys: {actions[0]['ur5e_2f85']['dof_pos_target'].keys()}")


    handler.set_dof_targets(actions)

    log.info(f"ctrl after set_dof_targets: {handler.physics.data.ctrl}")

    # 多次仿真子步让 PD 位置伺服收敛到目标关节角度
    for _sub in range(SIM_SUBSTEPS):
        handler.simulate()

    # 展开阶段：同步驱动帆板关节到目标位置
    _unfold_start = N_APPROACH + N_FINAL_APP + N_CLOSE
    _unfold_end = _unfold_start + N_UNFOLD
    if _unfold_start <= step < _unfold_end:
        _local = step - _unfold_start
        _a = _local / N_UNFOLD
        _a_s = 3 * _a**2 - 2 * _a**3
        for _jname, _fangle in _unfold_joints.items():
            _jid = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_JOINT, _jname)
            if _jid >= 0:
                _data.qpos[_model.jnt_qposadr[_jid]] = _fangle * (1.0 - _a_s)
                _data.qvel[_model.jnt_dofadr[_jid]] = 0.0
        _mj.mj_forward(_model, _data)

    obs = handler.get_states(mode="tensor")

    # 验证实际末端位置
    tcp_body_id = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_BODY, "tcp_link")
    if tcp_body_id >= 0:
        ee_actual = _data.xpos[tcp_body_id]
        log.info(f"Step {step}: 目标(基座)={ee_pos_target[0].tolist()}, 实际TCP(世界)=[{ee_actual[0]:.4f}, {ee_actual[1]:.4f}, {ee_actual[2]:.4f}]")

    ee_target_world = ee_pos_target[0] + BASE_WORLD
    log.info(f"Step {step}: 目标(世界)=[{ee_target_world[0]:.4f}, {ee_target_world[1]:.4f}, {ee_target_world[2]:.4f}], 实际TCP(世界)=[{ee_actual[0]:.4f}, {ee_actual[1]:.4f}, {ee_actual[2]:.4f}]")


    obs_saver.add(obs)

    # 阶段切换日志
    boundaries = [
        N_APPROACH,
        N_APPROACH + N_FINAL_APP,
        N_APPROACH + N_FINAL_APP + N_CLOSE,
        N_APPROACH + N_FINAL_APP + N_CLOSE + N_UNFOLD,
        N_APPROACH + N_FINAL_APP + N_CLOSE + N_UNFOLD + N_RELEASE,
    ]
    phase_names = ["接近完成", "对准完成", "夹爪闭合完成", "帆板展开完成", "夹爪松开完成"]
    for i, b in enumerate(boundaries):
        if step == b:
            log.info(f"Step {step}: {phase_names[i]}")
            if i == 3:  # 展开完成时验证
                sid = _mj.mj_name2id(_model, _mj.mjtObj.mjOBJ_SITE, "panel4_tip_L")
                if sid >= 0:
                    actual = _data.site_xpos[sid]
                    log.info(f"panel4_tip_L 实际位置: x={actual[0]:.4f}, y={actual[1]:.4f}, z={actual[2]:.4f}")

obs_saver.save()
log.info("完成，视频已保存。")
