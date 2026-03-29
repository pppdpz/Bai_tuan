"""UR5e 工作空间诊断：验证目标点可达性并生成轨迹路径点"""

import mujoco as mj
import numpy as np

model = mj.MjModel.from_xml_path("/home/e0/RoboVerse/asset_baituan/example_scene_y.xml")
data = mj.MjData(model)
robot_base = np.array([1.0, 0.0, 0.895])

# 获取关键 ID
get_id = lambda obj_type, name: mj.mj_name2id(model, obj_type, name)
wrist_bid = get_id(mj.mjtObj.mjOBJ_BODY, "wrist_3_link")
pinch_sid = get_id(mj.mjtObj.mjOBJ_SITE, "pinch")
panel_sid = get_id(mj.mjtObj.mjOBJ_SITE, "panel4_tip_L")
panel_r_sid = get_id(mj.mjtObj.mjOBJ_SITE, "panel4_tip")

def set_joints(joint_angles):
    """设置关节角度并执行正向运动学"""
    data.qpos[:] = 0.0
    for jname, angle in joint_angles.items():
        jid = get_id(mj.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = angle
    mj.mj_forward(model, data)

# 测试配置
test_configs = {
    "截图工作姿态": {"shoulder_pan_joint": 0.0628, "shoulder_lift_joint": -0.754, 
                  "elbow_joint": -1.29, "wrist_1_joint": 0.503, 
                  "wrist_2_joint": -0.0628, "wrist_3_joint": 0.0},
    "全零初始": dict.fromkeys(["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], 0.0),
    "建议初始(紧凑)": {"shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.0,
                    "elbow_joint": -1.5, "wrist_1_joint": 0.5,
                    "wrist_2_joint": 0.0, "wrist_3_joint": 0.0},
}

print("=" * 70)
print(f"UR5e 工作空间诊断 - 机器人底座: {robot_base}")
print("=" * 70)

for name, joints in test_configs.items():
    set_joints(joints)
    pinch_pos = data.site_xpos[pinch_sid].copy()
    pinch_base = pinch_pos - robot_base
    
    print(f"\n--- {name} ---")
    print(f"  关节: sh_lift={joints['shoulder_lift_joint']:.3f}, elbow={joints['elbow_joint']:.3f}")
    print(f"  pinch 世界: [{pinch_pos[0]:.4f}, {pinch_pos[1]:.4f}, {pinch_pos[2]:.4f}]")
    print(f"  pinch base: [{pinch_base[0]:.4f}, {pinch_base[1]:.4f}, {pinch_base[2]:.4f}] (距离: {np.linalg.norm(pinch_base):.4f}m)")

# 太阳能板位置
print("\n" + "=" * 70)
print("太阳能板关键位置")
print("=" * 70)

# 展开状态
set_joints({})
unfolded_pos = data.site_xpos[panel_sid].copy()
print(f"展开 panel4_tip_L: {unfolded_pos} (距base: {np.linalg.norm(unfolded_pos-robot_base):.4f}m)")

# 折叠状态
fold_angles = {"hinge_1": -1.5708, "hinge_2": 3.1416, "hinge_3": -3.1416, "hinge_4": 1.5708,
               "hinge_1_L": -1.5708, "hinge_2_L": 3.1416, "hinge_3_L": -3.1416, "hinge_4_L": 1.5708}
set_joints(fold_angles)
folded_pos = data.site_xpos[panel_sid].copy()
print(f"折叠 panel4_tip_L: {folded_pos} (距base: {np.linalg.norm(folded_pos-robot_base):.4f}m)")

# 生成轨迹路径点
set_joints(test_configs["截图工作姿态"])
screenshot_pinch = data.site_xpos[pinch_sid].copy()

print("\n" + "=" * 70)
print("推荐轨迹路径点 (pinch 坐标)")
print("=" * 70)

waypoints = {
    "initial_pos_pinch": screenshot_pinch,
    "waypoint_pos_pinch": [folded_pos[0], (screenshot_pinch[1] + folded_pos[1]) / 2, 
                           max(folded_pos[2], screenshot_pinch[2]) + 0.10],
    "approach_pos_pinch": folded_pos + [0, 0, 0.15],
    "grasp_pos_pinch": folded_pos,
    "deploy_pos_pinch": unfolded_pos,
    "retract_pos_pinch": unfolded_pos + [0, 0, 0.20],
}

for name, pos in waypoints.items():
    print(f"  {name:20s} = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

print("\n诊断完成!")
