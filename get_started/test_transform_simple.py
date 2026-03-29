"""
简单测试：验证坐标转换在不同情况下的正确性

测试场景：
1. 初始姿态，不同位置
2. 改变姿态，固定位置
3. 同时改变位置和姿态
"""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from inline_scene_adapter import install_inline_scene_patches
from roboverse_pack.robots.ur5e_2f85_cfg import Ur5E2F85Cfg


CFG_REL_IS_TCP_TO_EE = True
W3_TO_EE_BRIDGE_POS = np.array([0.1, 0.1, 0.0], dtype=np.float64)
W3_TO_EE_BRIDGE_RPY = np.array([np.pi / 2.0, -np.pi / 2.0, 0.0], dtype=np.float64)


def T_from_pos_quat_xyzw(pos3, quat_xyzw):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(pos3, dtype=np.float64).reshape(3)
    return T


def T_from_pos_rpy_xyz(pos3, rpy_xyz):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", np.asarray(rpy_xyz, dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(pos3, dtype=np.float64).reshape(3)
    return T


def cfg_T_ee_tcp(robot_cfg):
    T_cfg = T_from_pos_rpy_xyz(robot_cfg.curobo_tcp_rel_pos, robot_cfg.curobo_tcp_rel_rot)
    return np.linalg.inv(T_cfg) if CFG_REL_IS_TCP_TO_EE else T_cfg


def bridge_T_w3_ee():
    return T_from_pos_rpy_xyz(W3_TO_EE_BRIDGE_POS, W3_TO_EE_BRIDGE_RPY)


def pred_T_w3_tcp(robot_cfg):
    return bridge_T_w3_ee() @ cfg_T_ee_tcp(robot_cfg)


def quat_distance_deg(q1_xyzw, q2_xyzw):
    # 以最短弧角度比较姿态误差，自动处理 q/-q 等价
    dot = abs(float(np.dot(q1_xyzw, q2_xyzw)))
    dot = min(1.0, max(-1.0, dot))
    return np.degrees(2.0 * np.arccos(dot))


def set_ur5e_arm_qpos_by_name(robot, q6):
    arm_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    for jname, q in zip(arm_joint_names, q6):
        jid = robot.model.joint(jname).id
        adr = robot.model.jnt_qposadr[jid]
        robot.data.qpos[adr] = float(q)


def get_current_poses(robot):
    """获取当前 wrist3 和 TCP 的实际位姿"""
    wrist3_body_id = robot.model.body('wrist_3_link').id
    wrist3_pos = robot.data.xpos[wrist3_body_id].copy()
    wrist3_mat = robot.data.xmat[wrist3_body_id].reshape(3, 3).copy()
    wrist3_quat = R.from_matrix(wrist3_mat).as_quat()
    
    pinch_site_id = robot.model.site('pinch').id
    tcp_pos = robot.data.site_xpos[pinch_site_id].copy()
    tcp_mat = robot.data.site_xmat[pinch_site_id].reshape(3, 3).copy()
    tcp_quat = R.from_matrix(tcp_mat).as_quat()
    
    return wrist3_pos, wrist3_quat, tcp_pos, tcp_quat


def compute_offset(robot):
    """计算固定偏移"""
    wrist3_pos, wrist3_quat, tcp_pos, tcp_quat = get_current_poses(robot)
    
    T_world_to_wrist3 = np.eye(4)
    T_world_to_wrist3[:3, :3] = R.from_quat(wrist3_quat).as_matrix()
    T_world_to_wrist3[:3, 3] = wrist3_pos
    
    T_world_to_tcp = np.eye(4)
    T_world_to_tcp[:3, :3] = R.from_quat(tcp_quat).as_matrix()
    T_world_to_tcp[:3, 3] = tcp_pos
    
    T_wrist3_to_tcp = np.linalg.inv(T_world_to_wrist3) @ T_world_to_tcp
    
    offset_pos = T_wrist3_to_tcp[:3, 3]
    offset_quat = R.from_matrix(T_wrist3_to_tcp[:3, :3]).as_quat()
    
    return offset_pos, offset_quat


def wrist3_to_tcp(wrist3_pos, wrist3_quat, offset_pos, offset_quat):
    """wrist3 -> TCP 转换"""
    T_world_to_wrist3 = np.eye(4)
    T_world_to_wrist3[:3, :3] = R.from_quat(wrist3_quat).as_matrix()
    T_world_to_wrist3[:3, 3] = wrist3_pos
    
    T_wrist3_to_tcp = np.eye(4)
    T_wrist3_to_tcp[:3, :3] = R.from_quat(offset_quat).as_matrix()
    T_wrist3_to_tcp[:3, 3] = offset_pos
    
    T_world_to_tcp = T_world_to_wrist3 @ T_wrist3_to_tcp
    
    tcp_pos = T_world_to_tcp[:3, 3]
    tcp_quat = R.from_matrix(T_world_to_tcp[:3, :3]).as_quat()
    
    return tcp_pos, tcp_quat


def tcp_to_wrist3(tcp_pos, tcp_quat, offset_pos, offset_quat):
    """TCP -> wrist3 转换"""
    T_world_to_tcp = np.eye(4)
    T_world_to_tcp[:3, :3] = R.from_quat(tcp_quat).as_matrix()
    T_world_to_tcp[:3, 3] = tcp_pos
    
    T_wrist3_to_tcp = np.eye(4)
    T_wrist3_to_tcp[:3, :3] = R.from_quat(offset_quat).as_matrix()
    T_wrist3_to_tcp[:3, 3] = offset_pos
    
    T_world_to_wrist3 = T_world_to_tcp @ np.linalg.inv(T_wrist3_to_tcp)
    
    wrist3_pos = T_world_to_wrist3[:3, 3]
    wrist3_quat = R.from_matrix(T_world_to_wrist3[:3, :3]).as_quat()
    
    return wrist3_pos, wrist3_quat


def test_scenario(robot, robot_cfg, offset_pos, offset_quat, q6, scenario_name):
    """测试单个场景"""
    print(f"\n{'='*60}")
    print(f"测试场景: {scenario_name}")
    print(f"{'='*60}")
    
    # 按关节名设置 UR5e 6个手臂关节，避免 qpos 前缀写入导致测试姿态不变化
    set_ur5e_arm_qpos_by_name(robot, q6)
    mujoco.mj_forward(robot.model, robot.data)
    
    # 获取实际位姿
    wrist3_actual, wrist3_quat_actual, tcp_actual, tcp_quat_actual = get_current_poses(robot)
    
    print(f"实际位姿:")
    print(f"  wrist3: pos={wrist3_actual}, quat={wrist3_quat_actual}")
    print(f"  TCP:    pos={tcp_actual}, quat={tcp_quat_actual}")
    
    # 测试 wrist3 -> TCP 转换
    tcp_computed, tcp_quat_computed = wrist3_to_tcp(
        wrist3_actual, wrist3_quat_actual, offset_pos, offset_quat
    )
    
    pos_error_forward = np.linalg.norm(tcp_computed - tcp_actual)
    
    # 测试 TCP -> wrist3 转换
    wrist3_computed, wrist3_quat_computed = tcp_to_wrist3(
        tcp_actual, tcp_quat_actual, offset_pos, offset_quat
    )
    
    pos_error_backward = np.linalg.norm(wrist3_computed - wrist3_actual)
    
    print(f"\n转换结果:")
    print(f"  wrist3->TCP 位置误差: {pos_error_forward*1000:.6f} mm")
    print(f"  TCP->wrist3 位置误差: {pos_error_backward*1000:.6f} mm")
    
    # 检查姿态误差（四元数距离）
    quat_error_forward = 1 - abs(np.dot(tcp_quat_computed, tcp_quat_actual))
    quat_error_backward = 1 - abs(np.dot(wrist3_quat_computed, wrist3_quat_actual))
    
    print(f"  wrist3->TCP 姿态误差: {quat_error_forward:.6f}")
    print(f"  TCP->wrist3 姿态误差: {quat_error_backward:.6f}")
    
    # 新增：配置链预测 vs 实测链对齐误差
    T_w3_tcp_meas = T_from_pos_quat_xyzw(offset_pos, offset_quat)
    T_w3_tcp_pred = pred_T_w3_tcp(robot_cfg)
    T_err = np.linalg.inv(T_w3_tcp_pred) @ T_w3_tcp_meas
    cfg_pos_err_mm = np.linalg.norm(T_err[:3, 3]) * 1000.0
    cfg_rot_err_deg = np.degrees(np.linalg.norm(R.from_matrix(T_err[:3, :3]).as_rotvec()))

    print(f"  [cfg对齐] 位置误差: {cfg_pos_err_mm:.6f} mm")
    print(f"  [cfg对齐] 姿态误差: {cfg_rot_err_deg:.6f} deg")

    success = (
        pos_error_forward < 1e-6
        and pos_error_backward < 1e-6
        and cfg_pos_err_mm < 1.0
        and cfg_rot_err_deg < 0.5
    )
    print(f"\n结果: {'✓ 通过' if success else '✗ 失败'}")
    
    return success


def main():
    install_inline_scene_patches()
    
    xml_path = Path(__file__).parent.parent / "asset_baituan/example_scene_y.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    class Robot:
        def __init__(self, model, data):
            self.model = model
            self.data = data
    
    robot = Robot(model, data)
    robot_cfg = Ur5E2F85Cfg()
    mujoco.mj_forward(model, data)
    
    print("="*60)
    print("坐标转换简化测试")
    print("="*60)
    
    # 在初始姿态下计算偏移
    print("\n计算固定偏移（初始姿态）...")
    offset_pos, offset_quat = compute_offset(robot)
    print(f"偏移: pos={offset_pos}, quat={offset_quat}")
    
    # 测试场景
    test_cases = [
        {
            "name": "场景1: 初始姿态",
            "q6": np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
        },
        {
            "name": "场景2: 改变第1个关节 (+45度)",
            "q6": np.array([np.pi/4, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
        },
        {
            "name": "场景3: 改变第2个关节 (+30度)",
            "q6": np.array([0, -np.pi/2 + np.pi/6, np.pi/2, -np.pi/2, -np.pi/2, 0])
        },
        {
            "name": "场景4: 改变多个关节",
            "q6": np.array([np.pi/6, -np.pi/3, np.pi/3, -np.pi/4, -np.pi/2, np.pi/6])
        },
        {
            "name": "场景5: 随机姿态",
            "q6": np.random.uniform(-np.pi, np.pi, 6)
        }
    ]
    
    results = []
    for test_case in test_cases:
        success = test_scenario(
            robot,
            robot_cfg,
            offset_pos,
            offset_quat,
            test_case["q6"],
            test_case["name"],
        )
        results.append(success)
    
    # 总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    print(f"通过: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✓ 所有测试通过！坐标转换函数是正确的。")
        print("  转换函数已经正确处理了旋转变换。")
    else:
        print("\n✗ 部分测试失败，需要检查转换逻辑。")


if __name__ == "__main__":
    main()
