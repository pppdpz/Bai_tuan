"""
测试 TCP (pinch) 到 wrist_3_link 的坐标转换

这个脚本用于：
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

# 添加父目录到路径，以便导入 inline_scene_adapter
sys.path.insert(0, str(Path(__file__).parent.parent))
from inline_scene_adapter import install_inline_scene_patches

# import mujoco
# from scipy.spatial.transform import Rotation as R
# from pathlib import Path
# import sys

# 添加这两行
sys.path.insert(0, str(Path(__file__).parent.parent))
from inline_scene_adapter import install_inline_scene_patches


def compute_tcp_to_wrist3_transform(robot):
    """
    测量 TCP (pinch) 到 wrist_3_link 的固定变换
    
    这个变换是固定的，因为它们都是刚性连接的
    
    Returns:
        offset_pos: TCP 相对 wrist3 的位置偏移 [3]
        offset_quat: TCP 相对 wrist3 的姿态偏移 [4] (x,y,z,w)
        T_tcp_to_wrist3: 完整的 4x4 变换矩阵
    """
    # 获取 wrist_3_link 的位姿
    wrist3_body_id = robot.model.body('wrist_3_link').id
    wrist3_pos = robot.data.xpos[wrist3_body_id].copy()
    wrist3_mat = robot.data.xmat[wrist3_body_id].reshape(3, 3).copy()
    wrist3_quat = R.from_matrix(wrist3_mat).as_quat()  # [x,y,z,w]
    
    # 获取 TCP (pinch site) 的位姿
    pinch_site_id = robot.model.site('pinch').id
    tcp_pos = robot.data.site_xpos[pinch_site_id].copy()
    tcp_mat = robot.data.site_xmat[pinch_site_id].reshape(3, 3).copy()
    tcp_quat = R.from_matrix(tcp_mat).as_quat()  # [x,y,z,w]
    
    # 构建变换矩阵
    T_world_to_wrist3 = np.eye(4)
    T_world_to_wrist3[:3, :3] = wrist3_mat
    T_world_to_wrist3[:3, 3] = wrist3_pos
    
    T_world_to_tcp = np.eye(4)
    T_world_to_tcp[:3, :3] = tcp_mat
    T_world_to_tcp[:3, 3] = tcp_pos
    
    # 计算相对变换: T_tcp_to_wrist3 = T_wrist3_to_world × T_world_to_tcp
    # 这表示：从 wrist3 坐标系看，TCP 在哪里
    T_tcp_to_wrist3 = np.linalg.inv(T_world_to_wrist3) @ T_world_to_tcp
    
    # 提取位置和旋转
    offset_pos = T_tcp_to_wrist3[:3, 3]
    offset_rot = R.from_matrix(T_tcp_to_wrist3[:3, :3])
    offset_quat = offset_rot.as_quat()  # [x,y,z,w]
    
    print(f"\n{'='*60}")
    print(f"TCP 到 wrist_3_link 固定变换测量")
    print(f"{'='*60}")
    print(f"当前世界坐标系中：")
    print(f"  wrist3 位置: {wrist3_pos}")
    print(f"  wrist3 姿态 (quat): {wrist3_quat}")
    print(f"  TCP 位置: {tcp_pos}")
    print(f"  TCP 姿态 (quat): {tcp_quat}")
    print(f"\nwrist3 坐标系中的相对变换：")
    print(f"  位置偏移: {offset_pos}")
    print(f"  姿态偏移 (quat x,y,z,w): {offset_quat}")
    print(f"  欧拉角 (xyz, degrees): {offset_rot.as_euler('xyz', degrees=True)}")
    print(f"{'='*60}\n")
    
    return offset_pos, offset_quat, T_tcp_to_wrist3


def tcp_pose_to_wrist3_pose(tcp_pos, tcp_quat, offset_pos, offset_quat):
    """
    将 TCP 位姿转换为 wrist_3_link 位姿
    
    逻辑：
        已知：TCP 在世界坐标系中的目标位姿
        已知：TCP 相对 wrist3 的固定偏移 (T_tcp_to_wrist3)
        求：wrist3 在世界坐标系中应该在哪里
        
        关系：T_world_to_tcp = T_world_to_wrist3 @ T_wrist3_to_tcp
        因此：T_world_to_wrist3 = T_world_to_tcp @ inv(T_wrist3_to_tcp)
                                = T_world_to_tcp @ T_tcp_to_wrist3
        
        注意：T_tcp_to_wrist3 = inv(T_wrist3_to_tcp)
    
    Args:
        tcp_pos: TCP 目标位置 [3]
        tcp_quat: TCP 目标姿态四元数 [4] (x,y,z,w)
        offset_pos: TCP 相对 wrist3 的位置偏移 [3] (在 wrist3 坐标系中)
        offset_quat: TCP 相对 wrist3 的姿态偏移 [4] (x,y,z,w)
    
    Returns:
        wrist3_pos: wrist_3_link 位置 [3]
        wrist3_quat: wrist_3_link 姿态 [4] (x,y,z,w)
    """
    # 构建 TCP 变换矩阵（世界坐标系到 TCP）
    T_world_to_tcp = np.eye(4)
    T_world_to_tcp[:3, :3] = R.from_quat(tcp_quat).as_matrix()
    T_world_to_tcp[:3, 3] = tcp_pos
    
    # 构建偏移变换矩阵 (wrist3 到 TCP 的相对变换)
    T_wrist3_to_tcp = np.eye(4)
    T_wrist3_to_tcp[:3, :3] = R.from_quat(offset_quat).as_matrix()
    T_wrist3_to_tcp[:3, 3] = offset_pos
    
    # 计算 wrist3 在世界坐标系中的位姿
    # T_world_to_wrist3 = T_world_to_tcp @ inv(T_wrist3_to_tcp)
    T_world_to_wrist3 = T_world_to_tcp @ np.linalg.inv(T_wrist3_to_tcp)
    
    # 提取位置和姿态
    wrist3_pos = T_world_to_wrist3[:3, 3]
    wrist3_quat = R.from_matrix(T_world_to_wrist3[:3, :3]).as_quat()
    
    return wrist3_pos, wrist3_quat


def wrist3_pose_to_tcp_pose(wrist3_pos, wrist3_quat, offset_pos, offset_quat):
    """
    将 wrist_3_link 位姿转换为 TCP 位姿（逆变换，用于验证）
    
    关系：T_world_to_tcp = T_world_to_wrist3 @ T_wrist3_to_tcp
    """
    # 构建 wrist3 变换矩阵
    T_world_to_wrist3 = np.eye(4)
    T_world_to_wrist3[:3, :3] = R.from_quat(wrist3_quat).as_matrix()
    T_world_to_wrist3[:3, 3] = wrist3_pos
    
    # 构建偏移变换矩阵 (wrist3 到 TCP)
    T_wrist3_to_tcp = np.eye(4)
    T_wrist3_to_tcp[:3, :3] = R.from_quat(offset_quat).as_matrix()
    T_wrist3_to_tcp[:3, 3] = offset_pos
    
    # 计算 TCP 在世界坐标系中的位姿
    T_world_to_tcp = T_world_to_wrist3 @ T_wrist3_to_tcp
    
    # 提取位置和姿态
    tcp_pos = T_world_to_tcp[:3, 3]
    tcp_quat = R.from_matrix(T_world_to_tcp[:3, :3]).as_quat()
    
    return tcp_pos, tcp_quat


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
    
    offsets_pos = []
    offsets_quat = []
    
    for i in range(num_tests):
        # 随机设置关节角度
        qpos = robot.data.qpos.copy()
        qpos[:6] = np.random.uniform(-np.pi, np.pi, 6)  # UR5e 的 6 个关节
        robot.data.qpos[:] = qpos
        mujoco.mj_forward(robot.model, robot.data)
        
        # 测量偏移
        offset_pos_test, offset_quat_test, _ = compute_tcp_to_wrist3_transform(robot)
        offsets_pos.append(offset_pos_test)
        offsets_quat.append(offset_quat_test)
        
        # 验证正向和逆向转换
        wrist3_body_id = robot.model.body('wrist_3_link').id
        wrist3_pos_actual = robot.data.xpos[wrist3_body_id].copy()
        wrist3_mat_actual = robot.data.xmat[wrist3_body_id].reshape(3, 3).copy()
        wrist3_quat_actual = R.from_matrix(wrist3_mat_actual).as_quat()
        
        pinch_site_id = robot.model.site('pinch').id
        tcp_pos_actual = robot.data.site_xpos[pinch_site_id].copy()
        tcp_mat_actual = robot.data.site_xmat[pinch_site_id].reshape(3, 3).copy()
        tcp_quat_actual = R.from_matrix(tcp_mat_actual).as_quat()
        
        # 正向转换：TCP -> wrist3
        wrist3_pos_computed, wrist3_quat_computed = tcp_pose_to_wrist3_pose(
            tcp_pos_actual, tcp_quat_actual, offset_pos, offset_quat
        )
        
        # 逆向转换：wrist3 -> TCP
        tcp_pos_computed, tcp_quat_computed = wrist3_pose_to_tcp_pose(
            wrist3_pos_actual, wrist3_quat_actual, offset_pos, offset_quat
        )
        
        # 计算误差
        pos_error_forward = np.linalg.norm(wrist3_pos_computed - wrist3_pos_actual)
        pos_error_backward = np.linalg.norm(tcp_pos_computed - tcp_pos_actual)
        
        print(f"测试 {i+1}:")
        print(f"  正向转换位置误差: {pos_error_forward*1000:.4f} mm")
        print(f"  逆向转换位置误差: {pos_error_backward*1000:.4f} mm")
    
    # 统计偏移量的变化
    offsets_pos = np.array(offsets_pos)
    offsets_quat = np.array(offsets_quat)
    
    pos_std = np.std(offsets_pos, axis=0)
    quat_std = np.std(offsets_quat, axis=0)
    
    print(f"\n偏移量统计：")
    print(f"  位置偏移标准差: {pos_std} (应该接近 0)")
    print(f"  姿态偏移标准差: {quat_std} (应该接近 0)")
    print(f"{'='*60}\n")


def main():
    """主测试函数"""
    # 安装 inline scene patches（处理 mujocoinclude 标签）
    install_inline_scene_patches()
    
    # 加载完整场景（包含机器人和卫星）
    xml_path = Path(__file__).parent.parent / "asset_baituan/example_scene_y.xml"
    
    if not xml_path.exists():
        print(f"错误：找不到场景文件 {xml_path}")
        print("请确保在 RoboVerse 目录下运行此脚本")
        return
    
    print(f"加载场景: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    
    # 创建简单的 robot 对象
    class Robot:
        def __init__(self, model, data):
            self.model = model
            self.data = data
    
    robot = Robot(model, data)
    
    # 初始化仿真
    mujoco.mj_forward(model, data)
    
    print("\n" + "="*60)
    print("TCP 到 wrist_3_link 坐标转换测试")
    print("="*60)
    
    # 1. 测量初始姿态下的偏移
    print("\n步骤 1: 测量初始姿态下的固定偏移")
    offset_pos, offset_quat, T_tcp_to_wrist3 = compute_tcp_to_wrist3_transform(robot)
    
    # 2. 验证多个姿态下的一致性
    print("\n步骤 2: 验证不同姿态下的一致性")
    verify_transform_consistency(robot, offset_pos, offset_quat, num_tests=5)
    
    # 3. 输出可用于代码的常量
    print("\n" + "="*60)
    print("可用于代码的常量定义：")
    print("="*60)
    print(f"""
# TCP 到 wrist_3_link 的固定偏移（在 wrist3 坐标系中）
TCP_TO_WRIST3_OFFSET_POS = np.array({offset_pos.tolist()})
TCP_TO_WRIST3_OFFSET_QUAT = np.array({offset_quat.tolist()})  # [x,y,z,w]

# 或者直接使用变换矩阵
TCP_TO_WRIST3_TRANSFORM = np.array({T_tcp_to_wrist3.tolist()})
""")
    
    print("\n测试完成！")
    print("\n使用建议：")
    print("1. 将上述常量复制到你的主程序中")
    print("2. 在 get_trajectory_targets() 中使用 tcp_pose_to_wrist3_pose() 转换坐标")
    print("3. 如果位置误差 < 1mm，说明转换是准确的")


if __name__ == "__main__":
    main()
