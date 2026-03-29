from __future__ import annotations

from metasim.scenario.robot import BaseActuatorCfg, RobotCfg
from metasim.utils import configclass


@configclass
class Ur5E2F85Cfg(RobotCfg):
    name: str = "ur5e_2f85"
    num_joints: int = 12
    usd_path: str = "data_isaaclab/robots/UniversalRobots/ur5e/ur5e_2f85_fix.usd"
    urdf_path: str = "roboverse_data/robots/universal_robots_ur5e/urdf/ur5e_2f85_joint_limited_robot.urdf"
    mjcf_path: str = "roboverse_data/robots/universal_robots_ur5e/ur5e.xml"
    enabled_gravity: bool = False
    enabled_self_collisions: bool = False
    fix_base_link: bool = True

    actuators: dict[str, BaseActuatorCfg] = {
        "shoulder_pan_joint": BaseActuatorCfg(velocity_limit=2.175),
        "shoulder_lift_joint": BaseActuatorCfg(velocity_limit=2.175),
        "elbow_joint": BaseActuatorCfg(velocity_limit=2.175),
        "wrist_1_joint": BaseActuatorCfg(velocity_limit=2.175),
        "wrist_2_joint": BaseActuatorCfg(velocity_limit=2.61),
        "wrist_3_joint": BaseActuatorCfg(velocity_limit=2.61),
        # "finger_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        # "left_inner_finger_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        # "left_inner_knuckle_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        # "right_inner_finger_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        # "right_inner_knuckle_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        # "right_outer_knuckle_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        
        # actuators 中：
        "right_driver_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        "left_driver_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        "right_coupler_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        "left_coupler_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        "right_spring_link_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        "left_spring_link_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        "right_follower_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),
        "left_follower_joint": BaseActuatorCfg(velocity_limit=2.61, is_ee=True),

    }
    joint_limits: dict[str, tuple[float, float]] = {
        "shoulder_pan_joint": (-6.28319, 6.28319),
        "shoulder_lift_joint": (-6.28319, 6.28319),
        "elbow_joint": (-3.14159, 3.14159),
        "wrist_1_joint": (-6.28319, 6.28319),
        "wrist_2_joint": (-6.28319, 6.28319),
        "wrist_3_joint": (-6.28319, 6.28319),
        # "finger_joint": (0.0, 0.785398),
        # "left_inner_finger_joint": (0.0, 0.785398),
        # "left_inner_knuckle_joint": (0.0, 0.785398),
        # "right_inner_finger_joint": (0.0, 0.785398),
        # "right_inner_knuckle_joint": (0.0, 0.785398),
        # "right_outer_knuckle_joint": (0.0, 0.785398),

        # joint_limits 中：对应 MJCF 中各 class 的 range
        "right_driver_joint": (0.0, 0.8),           # class="driver" → range="0 0.8"
        "left_driver_joint": (0.0, 0.8),            # class="driver" → range="0 0.8"
        "right_coupler_joint": (-1.57, 0.0),        # class="coupler" → range="-1.57 0"
        "left_coupler_joint": (-1.57, 0.0),         # class="coupler" → range="-1.57 0"
        "right_spring_link_joint": (-0.29670597283, 0.8),  # class="spring_link" → range="-0.29670597283 0.8"
        "left_spring_link_joint": (-0.29670597283, 0.8),   # class="spring_link"
        "right_follower_joint": (-0.872664, 0.872664),     # class="follower" → range="-0.872664 0.872664"
        "left_follower_joint": (-0.872664, 0.872664),      # class="follower"

    }
    ee_body_name: str = "wrist_3_link"
    # ee_body_name: str = "tcp_link"

    default_joint_positions: dict[str, float] = {
        # UR5e 6个手臂关节
        "shoulder_pan_joint": 0.0,
        # "shoulder_lift_joint": -1.5708,      # -π/2，让上臂竖起来
        "shoulder_lift_joint": 0.0,      
        "elbow_joint": 0.0,
        # "wrist_1_joint": -1.5708,            # -π/2，让前臂自然下垂
        "wrist_1_joint": 0.0,            
        "wrist_2_joint": 0.0,
        "wrist_3_joint": 0.0,
        # Robotiq 2F-85 夹爪 8 个关节（全 0 = 张开）
        "right_driver_joint": 0.0,
        "right_coupler_joint": 0.0,
        "right_spring_link_joint": 0.0,
        "right_follower_joint": 0.0,
        "left_driver_joint": 0.0,
        "left_coupler_joint": 0.0,
        "left_spring_link_joint": 0.0,
        "left_follower_joint": 0.0,
    }


    # gripper_open_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # gripper_close_q = [0.785398, 0.785398, 0.785398, 0.785398, 0.785398, 0.785398]
    
    # 改为（8 个，对应 MJCF 中 8 个夹爪关节）：
    # 顺序对应 actuators 中的: right_driver, left_driver, right_coupler, left_coupler,
    #                        right_spring_link, left_spring_link, right_follower, left_follower
    gripper_open_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    gripper_close_q = [0.8, 0.8, -1.57, -1.57, 0.8, 0.8, 0.872664, 0.872664]
   
    curobo_ref_cfg_name: str = "ur5e_robotiq_2f_140.yml"
    curobo_tcp_rel_pos: tuple[float, float, float] = [0.0, 0.0, -0.0635]
    curobo_tcp_rel_rot: tuple[float, float, float] = [0.0, 0.0, 0.0]
