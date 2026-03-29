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


def test_scenario(robot, offset_pos, offset_quat, qpos, scenario_name):
    """测试单个场景"""
    print(f"\n{'='*60}")
    print(f"测试场景: {scenario_name}")
    print(f"{'='*60}")
    
    # 设置关节角度
    robot.data.qpos[:len(qpos)] = qpos
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
    
    success = pos_error_forward < 1e-6 and pos_error_backward < 1e-6
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
            "qpos": np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0, 0, 0])
        },
        {
            "name": "场景2: 改变第1个关节 (+45度)",
            "qpos": np.array([np.pi/4, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0, 0, 0])
        },
        {
            "name": "场景3: 改变第2个关节 (+30度)",
            "qpos": np.array([0, -np.pi/2 + np.pi/6, np.pi/2, -np.pi/2, -np.pi/2, 0, 0, 0])
        },
        {
            "name": "场景4: 改变多个关节",
            "qpos": np.array([np.pi/6, -np.pi/3, np.pi/3, -np.pi/4, -np.pi/2, np.pi/6, 0, 0])
        },
        {
            "name": "场景5: 随机姿态",
            "qpos": np.concatenate([np.random.uniform(-np.pi, np.pi, 6), [0, 0]])
        }
    ]
    
    results = []
    for test_case in test_cases:
        success = test_scenario(robot, offset_pos, offset_quat, 
                               test_case["qpos"], test_case["name"])
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
