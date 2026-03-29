"""验证框架：检查配置与实测 TCP 偏置的几何一致性"""

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

# 配置文件参数 (提取自 ur5e_2f85_cfg.py)
pos_cfg = np.array([0.0, 0.1, -0.1558])
rot_cfg_rpy = np.array([np.pi, 0.0, 0.0])

# 构建 T_ee_tcp_cfg 齐次位姿矩阵
T_ee_tcp_cfg = np.eye(4)
T_ee_tcp_cfg[:3, :3] = R.from_euler('xyz', rot_cfg_rpy).as_matrix()
T_ee_tcp_cfg[:3, 3] = pos_cfg

print("=" * 50)
print("[阶段1] 配置文件 T_ee_tcp_cfg (ee_link -> TCP):")
print(np.round(T_ee_tcp_cfg, 4))
print("=" * 50 + "\n")

# 初始化 MuJoCo 场景
model = mj.MjModel.from_xml_path("/home/e0/RoboVerse/asset_baituan/example_scene_y.xml")
data = mj.MjData(model)
wrist_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
pinch_sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "pinch")

# 测试姿态配置
test_configs = {
    "全零初始": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "肩抬45度": [0.0, -0.785, -1.57, 0.785, 0.0, 0.0],
    "截图姿态": [0.0628, -0.754, -1.29, 0.503, -0.0628, 0.0]
}
joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

first_bridge = None
print("=" * 50)
print("[阶段2] 动态对齐验证 (循环多组位姿)")
print("=" * 50)

def get_transform(pos, mat):
    """构建齐次变换矩阵"""
    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = mat.reshape(3, 3)
    return T

for config_name, angles in test_configs.items():
    # 设置关节角度并前向运动学
    data.qpos[:] = 0.0
    for jname, angle in zip(joint_names, angles):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = angle
    mj.mj_forward(model, data)

    # 提取变换矩阵
    T_world_w3 = get_transform(data.xpos[wrist_bid], data.xmat[wrist_bid])
    T_world_tcp = get_transform(data.site_xpos[pinch_sid], data.site_xmat[pinch_sid])

    # 计算实测偏置与桥接矩阵
    T_w3_tcp_meas = np.linalg.inv(T_world_w3) @ T_world_tcp
    T_w3_ee_bridge = T_w3_tcp_meas @ np.linalg.inv(T_ee_tcp_cfg)

    # 多姿态一致性分析
    if first_bridge is None:
        first_bridge = T_w3_ee_bridge
        print(f"[{config_name}] (首次计算) T_w3_ee_bridge:")
        print(np.round(T_w3_ee_bridge, 4))
        print(f"  位置偏移 (x,y,z) = {np.round(T_w3_ee_bridge[:3, 3], 4)}")
        print(f"  旋转偏移 (r,p,y) = {np.round(R.from_matrix(T_w3_ee_bridge[:3, :3]).as_euler('xyz'), 4)} rad")
    else:
        E_i = first_bridge @ np.linalg.inv(T_w3_ee_bridge)
        pos_err = np.linalg.norm(E_i[:3, 3])
        rot_err = np.linalg.norm(R.from_matrix(E_i[:3, :3]).as_rotvec())
        status = "✅ 完美一致" if pos_err < 1e-4 and rot_err < 1e-4 else "❌ 不一致"
        
        print(f"[{config_name}] 一致性检验:")
        print(f"  实测位置: {np.round(T_w3_tcp_meas[:3, 3], 4)}")
        print(f"  位置误差: {pos_err:.2e} m | 旋转误差: {rot_err:.2e} rad")
        print(f"  结果: {status}")
    print("-" * 50)

print("\n[结论分析]")
print("如果全部为'✅ 完美一致'，说明:")
print("1. 实测偏置与配置文件偏置【并不矛盾】，描述同一夹爪物理信息")
print("2. 配置原点是 ee_link，实测原点是 wrist_3_link")
print("3. 二者相差固定常数 T_w3_ee_bridge")
print("4. 后续可用 T_w3_tcp = T_w3_ee_bridge @ T_ee_tcp_cfg 替代硬编码偏移")
