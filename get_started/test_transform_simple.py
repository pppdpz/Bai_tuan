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


def build_transform(pos, rot):
    """构建 4x4 变换矩阵"""
    T = np.eye(4, dtype=np.float64)
    rot = np.asarray(rot, dtype=np.float64)
    
    if rot.shape == (4,):  # quaternion [x,y,z,w]
        T[:3, :3] = R.from_quat(rot).as_matrix()
    elif rot.shape == (3,):  # euler angles [rx,ry,rz]
        T[:3, :3] = R.from_euler("xyz", rot).as_matrix()
    elif rot.shape == (3, 3):  # rotation matrix
        T[:3, :3] = rot
    else:
        raise ValueError(f"Unsupported rotation format: shape={rot.shape}")
    
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def cfg_T_ee_tcp(robot_cfg):
    """从配置获取 EE 到 TCP 的变换"""
    T_cfg = build_transform(robot_cfg.curobo_tcp_rel_pos, robot_cfg.curobo_tcp_rel_rot)
    return np.linalg.inv(T_cfg) if CFG_REL_IS_TCP_TO_EE else T_cfg


def bridge_T_w3_ee():
    """桥接变换：wrist3 到 EE"""
    return build_transform(W3_TO_EE_BRIDGE_POS, W3_TO_EE_BRIDGE_RPY)


def pred_T_w3_tcp(robot_cfg):
    """预测的 wrist3 到 TCP 变换"""
    return bridge_T_w3_ee() @ cfg_T_ee_tcp(robot_cfg)


def set_ur5e_arm_qpos(robot, q6):
    """设置 UR5e 手臂关节角度"""
    arm_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    for jname, q in zip(arm_joints, q6):
        adr = robot.model.jnt_qposadr[robot.model.joint(jname).id]
        robot.data.qpos[adr] = float(q)


def get_pose(robot, body_name=None, site_name=None):
    """获取 body 或 site 的位姿"""
    if body_name:
        body_id = robot.model.body(body_name).id
        pos = robot.data.xpos[body_id].copy()
        mat = robot.data.xmat[body_id].reshape(3, 3).copy()
    else:
        site_id = robot.model.site(site_name).id
        pos = robot.data.site_xpos[site_id].copy()
        mat = robot.data.site_xmat[site_id].reshape(3, 3).copy()
    return pos, R.from_matrix(mat).as_quat()


def compute_offset(robot):
    """计算 wrist3 到 TCP 的固定偏移"""
    wrist3_pos, wrist3_quat = get_pose(robot, body_name='wrist_3_link')
    tcp_pos, tcp_quat = get_pose(robot, site_name='pinch')
    
    T_w3_tcp = np.linalg.inv(build_transform(wrist3_pos, wrist3_quat)) @ build_transform(tcp_pos, tcp_quat)
    
    return T_w3_tcp[:3, 3], R.from_matrix(T_w3_tcp[:3, :3]).as_quat()


def transform_pose(pos, quat, offset_pos, offset_quat, inverse=False):
    """通用位姿变换：wrist3 <-> TCP"""
    T_input = build_transform(pos, quat)
    T_offset = build_transform(offset_pos, offset_quat)
    
    T_output = T_input @ np.linalg.inv(T_offset) if inverse else T_input @ T_offset
    
    return T_output[:3, 3], R.from_matrix(T_output[:3, :3]).as_quat()


def test_scenario(robot, robot_cfg, offset_pos, offset_quat, q6, scenario_name):
    """测试单个场景"""
    print(f"\n{'='*60}\n测试场景: {scenario_name}\n{'='*60}")
    
    set_ur5e_arm_qpos(robot, q6)
    mujoco.mj_forward(robot.model, robot.data)
    
    # 获取实际位姿
    wrist3_pos, wrist3_quat = get_pose(robot, body_name='wrist_3_link')
    tcp_pos, tcp_quat = get_pose(robot, site_name='pinch')
    
    print(f"实际位姿:")
    print(f"  wrist3: pos={wrist3_pos}, quat={wrist3_quat}")
    print(f"  TCP:    pos={tcp_pos}, quat={tcp_quat}")
    
    # 测试双向转换
    tcp_computed, tcp_quat_computed = transform_pose(wrist3_pos, wrist3_quat, offset_pos, offset_quat)
    wrist3_computed, wrist3_quat_computed = transform_pose(tcp_pos, tcp_quat, offset_pos, offset_quat, inverse=True)
    
    pos_err_fwd = np.linalg.norm(tcp_computed - tcp_pos) * 1000
    pos_err_bwd = np.linalg.norm(wrist3_computed - wrist3_pos) * 1000
    quat_err_fwd = 1 - abs(np.dot(tcp_quat_computed, tcp_quat))
    quat_err_bwd = 1 - abs(np.dot(wrist3_quat_computed, wrist3_quat))
    
    print(f"\n转换结果:")
    print(f"  wrist3->TCP 位置误差: {pos_err_fwd:.6f} mm")
    print(f"  TCP->wrist3 位置误差: {pos_err_bwd:.6f} mm")
    print(f"  wrist3->TCP 姿态误差: {quat_err_fwd:.6f}")
    print(f"  TCP->wrist3 姿态误差: {quat_err_bwd:.6f}")
    
    # 配置链对齐误差
    T_w3_tcp_meas = build_transform(offset_pos, offset_quat)
    T_w3_tcp_pred = pred_T_w3_tcp(robot_cfg)
    T_err = np.linalg.inv(T_w3_tcp_pred) @ T_w3_tcp_meas
    
    cfg_pos_err = np.linalg.norm(T_err[:3, 3]) * 1000
    cfg_rot_err = np.degrees(np.linalg.norm(R.from_matrix(T_err[:3, :3]).as_rotvec()))
    
    print(f"  [cfg对齐] 位置误差: {cfg_pos_err:.6f} mm")
    print(f"  [cfg对齐] 姿态误差: {cfg_rot_err:.6f} deg")
    
    success = pos_err_fwd < 0.001 and pos_err_bwd < 0.001 and cfg_pos_err < 1.0 and cfg_rot_err < 0.5
    print(f"\n结果: {'✓ 通过' if success else '✗ 失败'}")
    
    return success


def main():
    install_inline_scene_patches()
    
    xml_path = Path(__file__).parent.parent / "asset_baituan/example_scene_y.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    class Robot:
        def __init__(self, model, data):
            self.model, self.data = model, data
    
    robot = Robot(model, data)
    robot_cfg = Ur5E2F85Cfg()
    mujoco.mj_forward(model, data)
    
    print(f"{'='*60}\n坐标转换简化测试\n{'='*60}")
    
    # 计算固定偏移
    print("\n计算固定偏移（初始姿态）...")
    offset_pos, offset_quat = compute_offset(robot)
    print(f"偏移: pos={offset_pos}, quat={offset_quat}")
    
    # 测试场景
    test_cases = [
        ("场景1: 初始姿态", np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])),
        ("场景2: 改变第1个关节 (+45度)", np.array([np.pi/4, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])),
        ("场景3: 改变第2个关节 (+30度)", np.array([0, -np.pi/2 + np.pi/6, np.pi/2, -np.pi/2, -np.pi/2, 0])),
        ("场景4: 改变多个关节", np.array([np.pi/6, -np.pi/3, np.pi/3, -np.pi/4, -np.pi/2, np.pi/6])),
        ("场景5: 随机姿态", np.random.uniform(-np.pi, np.pi, 6))
    ]
    
    results = [test_scenario(robot, robot_cfg, offset_pos, offset_quat, q6, name) 
               for name, q6 in test_cases]
    
    # 总结
    print(f"\n{'='*60}\n测试总结\n{'='*60}")
    print(f"通过: {sum(results)}/{len(results)}")
    
    if all(results):
        print("\n✓ 所有测试通过！坐标转换函数正确。")
    else:
        print("\n✗ 部分测试失败，需要检查转换逻辑。")


if __name__ == "__main__":
    main()
