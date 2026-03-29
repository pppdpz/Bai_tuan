"""
测试 TCP (pinch) 到 wrist_3_link 的坐标转换

功能：
1. 测量 TCP 到 wrist_3_link 的固定几何偏移
2. 验证坐标转换的正确性
3. 在多个机器人姿态下验证偏移的恒定性

使用方法：
    python test_tcp_transform.py
"""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from inline_scene_adapter import install_inline_scene_patches


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
    
    quat = R.from_matrix(mat).as_quat()
    return pos, mat, quat


def build_transform_matrix(pos, mat):
    """构建 4x4 变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = mat
    T[:3, 3] = pos
    return T


def compute_tcp_to_wrist3_transform(robot):
    """
    测量 TCP (pinch) 到 wrist_3_link 的固定变换
    
    Returns:
        offset_pos: TCP 相对 wrist3 的位置偏移 [3]
        offset_quat: TCP 相对 wrist3 的姿态偏移 [4] (x,y,z,w)
        T_tcp_to_wrist3: 完整的 4x4 变换矩阵
    """
    wrist3_pos, wrist3_mat, wrist3_quat = get_pose(robot, body_name='wrist_3_link')
    tcp_pos, tcp_mat, tcp_quat = get_pose(robot, site_name='pinch')
    
    T_world_to_wrist3 = build_transform_matrix(wrist3_pos, wrist3_mat)
    T_world_to_tcp = build_transform_matrix(tcp_pos, tcp_mat)
    
    # 计算相对变换: T_tcp_to_wrist3 = inv(T_world_to_wrist3) @ T_world_to_tcp
    T_tcp_to_wrist3 = np.linalg.inv(T_world_to_wrist3) @ T_world_to_tcp
    
    offset_pos = T_tcp_to_wrist3[:3, 3]
    offset_rot = R.from_matrix(T_tcp_to_wrist3[:3, :3])
    offset_quat = offset_rot.as_quat()
    
    print(f"\n{'='*60}")
    print("TCP 到 wrist_3_link 固定变换测量")
    print(f"{'='*60}")
    print("当前世界坐标系中：")
    print(f"  wrist3 位置: {wrist3_pos}")
    print(f"  wrist3 姿态 (quat): {wrist3_quat}")
    print(f"  TCP 位置: {tcp_pos}")
    print(f"  TCP 姿态 (quat): {tcp_quat}")
    print("\nwrist3 坐标系中的相对变换：")
    print(f"  位置偏移: {offset_pos}")
    print(f"  姿态偏移 (quat x,y,z,w): {offset_quat}")
    print(f"  欧拉角 (xyz, degrees): {offset_rot.as_euler('xyz', degrees=True)}")
    print(f"{'='*60}\n")
    
    return offset_pos, offset_quat, T_tcp_to_wrist3


def transform_pose(pos, quat, offset_pos, offset_quat, inverse=False):
    """
    通用位姿变换函数
    
    Args:
        pos: 输入位置 [3]
        quat: 输入姿态 [4] (x,y,z,w)
        offset_pos: 偏移位置 [3]
        offset_quat: 偏移姿态 [4] (x,y,z,w)
        inverse: True 表示 TCP->wrist3, False 表示 wrist3->TCP
    
    Returns:
        output_pos: 输出位置 [3]
        output_quat: 输出姿态 [4] (x,y,z,w)
    """
    T_input = build_transform_matrix(pos, R.from_quat(quat).as_matrix())
    T_offset = build_transform_matrix(offset_pos, R.from_quat(offset_quat).as_matrix())
    
    T_output = T_input @ np.linalg.inv(T_offset) if inverse else T_input @ T_offset
    
    output_pos = T_output[:3, 3]
    output_quat = R.from_matrix(T_output[:3, :3]).as_quat()
    
    return output_pos, output_quat


def tcp_pose_to_wrist3_pose(tcp_pos, tcp_quat, offset_pos, offset_quat):
    """将 TCP 位姿转换为 wrist_3_link 位姿"""
    return transform_pose(tcp_pos, tcp_quat, offset_pos, offset_quat, inverse=True)


def wrist3_pose_to_tcp_pose(wrist3_pos, wrist3_quat, offset_pos, offset_quat):
    """将 wrist_3_link 位姿转换为 TCP 位姿"""
    return transform_pose(wrist3_pos, wrist3_quat, offset_pos, offset_quat, inverse=False)


def verify_transform_consistency(robot, offset_pos, offset_quat, num_tests=5):
    """
    在多个随机机器人姿态下验证变换的一致性
    
    验证：
    1. 偏移量在不同姿态下是否恒定
    2. 正向和逆向转换是否一致
    """
    print(f"\n{'='*60}")
    print(f"验证变换一致性（测试 {num_tests} 个随机姿态）")
    print(f"{'='*60}\n")
    
    offsets_pos, offsets_quat = [], []
    
    for i in range(num_tests):
        # 随机设置关节角度
        robot.data.qpos[:6] = np.random.uniform(-np.pi, np.pi, 6)
        mujoco.mj_forward(robot.model, robot.data)
        
        # 测量偏移
        offset_pos_test, offset_quat_test, _ = compute_tcp_to_wrist3_transform(robot)
        offsets_pos.append(offset_pos_test)
        offsets_quat.append(offset_quat_test)
        
        # 获取实际位姿
        wrist3_pos_actual, _, wrist3_quat_actual = get_pose(robot, body_name='wrist_3_link')
        tcp_pos_actual, _, tcp_quat_actual = get_pose(robot, site_name='pinch')
        
        # 正向和逆向转换
        wrist3_pos_computed, _ = tcp_pose_to_wrist3_pose(tcp_pos_actual, tcp_quat_actual, offset_pos, offset_quat)
        tcp_pos_computed, _ = wrist3_pose_to_tcp_pose(wrist3_pos_actual, wrist3_quat_actual, offset_pos, offset_quat)
        
        # 计算误差
        pos_error_forward = np.linalg.norm(wrist3_pos_computed - wrist3_pos_actual)
        pos_error_backward = np.linalg.norm(tcp_pos_computed - tcp_pos_actual)
        
        print(f"测试 {i+1}:")
        print(f"  正向转换位置误差: {pos_error_forward*1000:.4f} mm")
        print(f"  逆向转换位置误差: {pos_error_backward*1000:.4f} mm")
    
    # 统计偏移量的变化
    pos_std = np.std(offsets_pos, axis=0)
    quat_std = np.std(offsets_quat, axis=0)
    
    print(f"\n偏移量统计：")
    print(f"  位置偏移标准差: {pos_std} (应接近 0)")
    print(f"  姿态偏移标准差: {quat_std} (应接近 0)")
    print(f"{'='*60}\n")


def main():
    """主测试函数"""
    install_inline_scene_patches()
    
    xml_path = Path(__file__).parent.parent / "asset_baituan/example_scene_y.xml"
    if not xml_path.exists():
        print(f"错误：找不到场景文件 {xml_path}")
        return
    
    print(f"加载场景: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    class Robot:
        def __init__(self, model, data):
            self.model, self.data = model, data
    
    robot = Robot(model, data)
    
    print(f"\n{'='*60}\nTCP 到 wrist_3_link 坐标转换测试\n{'='*60}")
    
    print("\n步骤 1: 测量初始姿态下的固定偏移")
    offset_pos, offset_quat, T_tcp_to_wrist3 = compute_tcp_to_wrist3_transform(robot)
    
    print("\n步骤 2: 验证不同姿态下的一致性")
    verify_transform_consistency(robot, offset_pos, offset_quat, num_tests=5)
    
    print(f"\n{'='*60}\n可用于代码的常量定义：\n{'='*60}")
    print(f"""
# TCP 到 wrist_3_link 的固定偏移（在 wrist3 坐标系中）
TCP_TO_WRIST3_OFFSET_POS = np.array({offset_pos.tolist()})
TCP_TO_WRIST3_OFFSET_QUAT = np.array({offset_quat.tolist()})  # [x,y,z,w]

# 或者直接使用变换矩阵
TCP_TO_WRIST3_TRANSFORM = np.array({T_tcp_to_wrist3.tolist()})
""")
    
    print("测试完成！\n\n使用建议：")
    print("1. 将上述常量复制到主程序中")
    print("2. 使用 tcp_pose_to_wrist3_pose() 转换坐标")
    print("3. 位置误差 < 1mm 说明转换准确")


if __name__ == "__main__":
    main()
