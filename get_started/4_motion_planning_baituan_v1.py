"""This script is used to test the static scene."""

from __future__ import annotations                      # 启用 PEP 604 风格的延迟类型注解求值（如 X | Y），让类型提示在运行时不被立即解析，避免前向引用问题

try:
    import isaacgym                                     # noqa: F401
except ImportError:
    pass

import math                                             # 标准库导入：数学运算、操作系统接口、以及 Literal 类型（用于限定变量只能取特定值）
import os
from typing import Literal

import rootutils                                        # 用于自动定位项目根目录并设置 Python 路径
import torch                                            # PyTorch，深度学习框架
import tyro                                             # 基于类型注解自动生成 CLI 参数解析器的库
from loguru import logger as log                        # 更好用的日志库，这里别名为 log
from rich.logging import RichHandler                    # 来自 rich 库，提供带颜色和格式化的终端日志输出

rootutils.setup_root(__file__, pythonpath=True)         # 以当前文件为起点，向上查找项目根目录（通常通过 .git、pyproject.toml 等标记文件判断），然后把根目录加入 sys.path，这样后面就能用绝对路径导入项目内的模块了
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])        # 配置 loguru 使用 Rich 的 handler 输出日志，格式只保留消息本身（Rich 会自动加上时间、级别等装饰）


from metasim.constants import PhysicStateType           # 导入物理状态类型枚举，用于区分不同的物理状态（如静态、动态等）
from metasim.scenario.cameras import PinholeCameraCfg   # 导入针孔相机配置类，用于在仿真场景中设置相机参数（焦距、分辨率等）
from metasim.scenario.objects import (                  
    ArticulationObjCfg,                                 # 关节体（如机械臂，有多个可动关节）
    PrimitiveCubeCfg,                                   # 基础立方体
    PrimitiveSphereCfg,                                 # 基础球体
    RigidObjCfg,                                        # 刚体物体（不可变形，无关节）
)
from metasim.scenario.scenario import ScenarioCfg       # 导入场景配置类，用于定义整个仿真场景的参数
from metasim.utils import configclass                   # 导入 configclass 装饰器，类似 dataclass，用于定义可序列化的配置类
from metasim.utils.obs_utils import ObsSaver            # 导入观测数据保存工具，用于保存仿真过程中的观测信息（图像、状态等）
from metasim.utils.setup_util import get_handler        # 导入 get_handler，用于根据配置获取对应的仿真后端 handler（如 Isaac Sim、MuJoCo 等）


@configclass
class Args:
    """Arguments for the static scene."""

    # robot: str = "franka"                               # 机器人类型，默认是 "franka"（Franka Emika Panda，一款常用的 7 自由度协作机械臂）
    robot: str = "ur5e_2f85"                            # 从 "franka" 改为 "ur5e_2f85"

    ## Handlers
    sim: Literal[                                       # 仿真后端选择，用 Literal 限定只能从这 8 个仿真器中选一个。默认是 "mujoco"。这体现了 MetaSim 框架的核心设计——同一套代码可以跑在不同的物理仿真引擎上
        "isaacsim",
        "isaacgym",
        "genesis",
        "pybullet",
        "sapien2",
        "sapien3",
        "mujoco",
        "mjx",
    ] = "mujoco"

    ## Others
    num_envs: int = 1                                   # 并行仿真环境数量，默认 1 个。GPU 仿真器（如 Isaac Gym）支持同时跑成百上千个环境来加速数据收集
    headless: bool = True                               # 是否无头模式运行（不显示 GUI 窗口），默认 True。服务器上跑仿真通常不需要渲染窗口
    solver: Literal["curobo", "pyroki"] = "pyroki"      # "curobo"：NVIDIA 的 cuRobo，基于 GPU 加速的运动规划库；"pyroki"：一个轻量级的 Python IK 求解器

    def __post_init__(self):
        """Post-initialization configuration."""        # __post_init__ 是 dataclass 的钩子方法，在 __init__ 完成后自动调用。这里用来打印所有参数，方便调试确认配置是否正确
        log.info(f"Args: {self}")


args = tyro.cli(Args)                                   # 这是 tyro 的核心用法——把 Args 类自动转换成命令行参数解析器。运行脚本时可以通过命令行覆盖默认值

# IK solver imports are now handled in the unified solver
log.info(f"Using IK solver: {args.solver}")             # 打印当前使用的 IK 求解器

# initialize scenario
scenario = ScenarioCfg(
    robots=[args.robot],                                # 场景中的机器人列表（传入的是字符串，框架内部会解析成对应的机器人配置）
    simulator=args.sim,                                 # 使用哪个仿真后端
    headless=args.headless,                             # 是否无头运行
    num_envs=args.num_envs,                             # 并行环境数
)

# add cameras
# 给场景添加一个针孔相机：width=1024, height=1024：渲染分辨率 1024×1024；pos=(1.5, -1.5, 1.5)：相机在世界坐标系中的位置（x=1.5, y=-1.5, z=1.5），位于场景的右上方斜后方
# look_at=(0.0, 0.0, 0.0)：相机朝向原点，也就是场景中心，这是一个俯视角度的观察相机，用于捕获场景的 RGB 图像或深度图
scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]

# add objects
scenario.objects = [
    PrimitiveCubeCfg(                                   
        name="cube",                                    # name="cube"：物体名称
        size=(0.1, 0.1, 0.1),                           # size=(0.1, 0.1, 0.1)：边长 10cm 的正方体
        color=[1.0, 0.0, 0.0],                          # color=[1.0, 0.0, 0.0]：RGB 颜色，纯红色
        physics=PhysicStateType.RIGIDBODY,              # physics=PhysicStateType.RIGIDBODY：设为刚体，意味着它会参与物理仿真（有碰撞、受重力影响），但不会变形
    ),
    PrimitiveSphereCfg(
        name="sphere",                                  # 蓝色球体
        radius=0.1,                                     # radius=0.1：半径 10cm
        color=[0.0, 0.0, 1.0],
        physics=PhysicStateType.RIGIDBODY,              # 同样是刚体，会参与物理交互
    ),
    RigidObjCfg(
        name="bbq_sauce",                               # 一个 BBQ 酱瓶子（来自 LIBERO 数据集的资产）
        scale=(2, 2, 2),                                # scale=(2, 2, 2)：放大 2 倍（原始模型可能比较小）
        physics=PhysicStateType.RIGIDBODY,              
        #  MetaSim 跨仿真器的关键设计——同一个物体提供多种格式，框架根据当前使用的仿真后端自动选择对应的模型文件
        usd_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/usd/bbq_sauce.usd",         # USD 格式，给 Isaac Sim 用
        urdf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/urdf/bbq_sauce.urdf",      # URDF 格式，给 PyBullet、SAPIEN 等用
        mjcf_path="roboverse_data/assets/libero/COMMON/stable_hope_objects/bbq_sauce/mjcf/bbq_sauce.xml",       # MJCF 格式，给 MuJoCo 用
    ),
    ArticulationObjCfg(
        name="box_base",                                # 一个可活动的盒子底座（来自 RLBench 的 close_box 任务）
        fix_base_link=True,                             # fix_base_link=True：固定底座不动（盒子底部钉在桌面上，只有盖子能动）
        usd_path="roboverse_data/assets/rlbench/close_box/box_base/usd/box_base.usd",
        urdf_path="roboverse_data/assets/rlbench/close_box/box_base/urdf/box_base_unique.urdf",
        mjcf_path="roboverse_data/assets/rlbench/close_box/box_base/mjcf/box_base_unique.mjcf",
    ),
]

log.info(f"Using simulator: {args.sim}")                # 打印当前使用的仿真器名称

# handler 对象封装了所有与仿真器交互的接口（加载场景、步进仿真、获取状态等），上层代码通过统一的 API 调用它，不需要关心底层是哪个仿真器
handler = get_handler(scenario)                         # 根据 scenario 配置创建对应的仿真 handler。get_handler 内部会读取 scenario.simulator 的值（比如 "mujoco"），然后实例化对应的仿真后端（如 MujocoHandler）

if args.robot == "franka":                              # 根据选择的机器人类型，设置对应的初始关节配置
    robot_dict = {
        "franka": {                                     # 字典的 key 是机器人名称，value 是它的初始状态
            "pos": torch.tensor([0.0, 0.0, 0.0]),       # 机器人基座在世界坐标系中的位置——原点 (0, 0, 0)
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),  # 机器人基座的旋转，用四元数表示 (w, x, y, z)。[1, 0, 0, 0] 是单位四元数，表示无旋转（朝向默认方向）
            "dof_pos": {                                # Franka 的各关节初始角度（弧度）
                "panda_joint1": 0.0,                    # 7个手臂关节。这组值是 Franka 的经典"home"姿态——手臂弯曲呈一个自然的准备姿势，末端执行器大致朝下，远离关节极限和奇异点
                "panda_joint2": -0.785398,
                "panda_joint3": 0.0,
                "panda_joint4": -2.356194,
                "panda_joint5": 0.0,
                "panda_joint6": 1.570796,
                "panda_joint7": 0.785398,
                "panda_finger_joint1": 0.04,            # 两个手指关节，0.04 表示夹爪张开状态（Franka 夹爪最大开合约 0.04m 每侧）
                "panda_finger_joint2": 0.04,
            },
        }
    }
elif args.robot == "kinova_gen3_robotiq_2f85":          # Kinova Gen3 机械臂 + Robotiq 2F-85 夹爪的初始配置
    robot_dict = {
        "kinova_gen3_robotiq_2f85": {
            "pos": torch.tensor([0.0, 0.0, 0.0]),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "dof_pos": {
                "joint_1": 0.0,
                "joint_2": math.pi / 6,
                "joint_3": 0.0,
                "joint_4": math.pi / 2,
                "joint_5": 0.0,
                "joint_6": 0.0,
                "joint_7": 0.0,
                "finger_joint": 0.0,                    # 1个 finger_joint（Robotiq 夹爪是联动的，一个关节控制所有手指）
            },
        }
    }
elif args.robot == "ur5e_2f85":
    robot_dict = {
        "ur5e_2f85": {
            "pos": torch.tensor([0.0, 0.0, 0.0]),
            "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "dof_pos": {
                # UR5e 6个手臂关节（标准 home 位姿）
                "shoulder_pan_joint": 0.0,
                # "shoulder_lift_joint": -1.5708,   # -90°
                "shoulder_lift_joint": 0.0,   # -90°
                # "elbow_joint": 0.0,
                "elbow_joint": 1.5708,          # 90°
                # "wrist_1_joint": -1.5708,         # -90°
                "wrist_1_joint": 0.0,         # -90°
                # "wrist_2_joint": 0.0,
                "wrist_2_joint": 4.7124,        # 270°
                # "wrist_3_joint": 0.0,
                "wrist_3_joint": -1.5708,       # -90°
                # Robotiq 2F-85 夹爪关节（全部 0 = 张开）
                # "finger_joint": 0.0,
                # "left_inner_finger_joint": 0.0,
                # "left_inner_knuckle_joint": 0.0,
                # "right_inner_finger_joint": 0.0,
                # "right_inner_knuckle_joint": 0.0,
                # "right_outer_knuckle_joint": 0.0,

                # 应改为（MJCF 中实际的关节名）：
                "right_driver_joint": 0.0,
                "right_coupler_joint": 0.0,
                "right_spring_link_joint": 0.0,
                "right_follower_joint": 0.0,
                "left_driver_joint": 0.0,
                "left_coupler_joint": 0.0,
                "left_spring_link_joint": 0.0,
                "left_follower_joint": 0.0,

            },
        }
    }
else:   
    robot_dict = {}                                     # 其他机器人暂不设置初始关节状态，传空字典（框架会用默认值）

init_states = [
    {
        "objects": {                                    # 定义场景的完整初始状态，包含所有物体和机器人的位姿
            "cube": {                                   # 红色立方体放在 (0.3, -0.2, 0.05)，z=0.05 刚好是边长 0.1 的立方体放在地面上的中心高度，无旋转
                "pos": torch.tensor([0.3, -0.2, 0.05]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "sphere": {                                 # 蓝色球体放在 (0.4, -0.6, 0.05)，z=0.05 同理（半径 0.1 的球心高度）
                "pos": torch.tensor([0.4, -0.6, 0.05]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "bbq_sauce": {                              # BBQ 酱瓶放在 (0.7, -0.3, 0.14)，z 更高因为瓶子被放大了 2 倍，需要更高的初始位置避免穿透地面
                "pos": torch.tensor([0.7, -0.3, 0.14]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "box_base": {                               
                "pos": torch.tensor([0.5, 0.2, 0.1]),   # 盒子底座放在 (0.5, 0.2, 0.1)
                "rot": torch.tensor([0.0, 0.7071, 0.0, 0.7071]),    # 旋转四元数 [0, 0.7071, 0, 0.7071] 表示绕 Y 轴旋转 90°（让盒子朝向合适的方向）
                "dof_pos": {"box_joint": 0.0},          # "dof_pos": {"box_joint": 0.0} 设置盒盖关节角度为 0，即盒盖处于打开/初始位置
            },
        },
        "robots": robot_dict,
    }
    for _ in range(args.num_envs)                       # 用列表推导式为每个并行环境生成一份相同的初始状态。如果 num_envs=4，就会生成 4 份一样的初始配置。每个环境独立仿真，但起始状态相同
]


robot = scenario.robots[0]                              # 取出场景中第一个（也是唯一一个）机器人的配置对象，后续用于 IK 求解等操作

from metasim.utils.ik_solver import process_gripper_command, setup_ik_solver

# Setup unified IK solver
ik_solver = setup_ik_solver(robot, args.solver)         # 延迟导入 IK 求解器相关函数：setup_ik_solver：根据配置初始化 IK 求解器实例；process_gripper_command：处理夹爪开合指令，将高层的"开/关"命令转换为具体的关节位置

handler.set_states(init_states)                         # 将之前定义的初始状态（机器人关节角度、物体位姿等）写入仿真器，让仿真世界中的所有物体归位到指定的初始位置
obs = handler.get_states(mode="tensor")                 # 从仿真器读取当前状态（观测），mode="tensor" 表示返回 PyTorch tensor 格式（而不是 numpy 或字典），方便后续 GPU 计算
os.makedirs("get_started/output", exist_ok=True)        # 创建输出目录，exist_ok=True 表示目录已存在时不报错

## Main loop
obs_saver = ObsSaver(video_path=f"get_started/output/4_motion_planning_{args.sim}.mp4") # 创建观测保存器，会把每一帧的观测数据拼成视频保存。文件名包含仿真器名称，比如 4_motion_planning_mujoco.mp4
obs_saver.add(obs)                                      # 保存第一帧（初始状态）

robot_joint_limits = scenario.robots[0].joint_limits    # 获取机器人的关节限位信息（每个关节的最小/最大角度），后续 IK 求解时用来约束解在合法范围内

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()


for step in range(200):                                 # 主仿真循环，跑 200 步
    log.debug(f"Step {step}")                           # 用 debug 级别打印步数（不会在默认日志级别下显示，避免刷屏）
    states = handler.get_states()                       # 每步开始时获取当前仿真状态（所有物体和机器人的位姿、速度等）

    if scenario.robots[0].name == "franka":             # 以下的目标轨迹生成逻辑只针对 Franka 机器人
        x_target = 0.3 + 0.1 * (step / 100)             # 从 0.3 线性增长到 0.5（step=200 时），向前移动
        y_target = 0.5 - 0.5 * (step / 100)             # 从 0.5 线性减小到 -0.5，向右移动
        z_target = 0.6 - 0.2 * (step / 100)             # 从 0.6 线性减小到 0.2，向下移动

        # # Randomly assign x/y/z target for each env
        # def pick_device():                              # 选择计算设备，优先级：CUDA GPU > Apple Silicon MPS > CPU
        #     if torch.cuda.is_available():
        #         return torch.device("cuda")
        #     # Optional: Apple Silicon (PyTorch 1.12+ with MPS)
        #     if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        #         return torch.device("mps")
        #     return torch.device("cpu")

        # device = pick_device()
        ee_pos_target = torch.zeros((args.num_envs, 3), device=device)                  # 创建一个 shape 为 (num_envs, 3) 的零张量，用来存放每个并行环境的末端执行器目标位置
        for i in range(args.num_envs):                                                  # 多个并行环境中的机械臂会朝不同方向运动，用于演示/测试多环境并行仿真的能力
            if i % 3 == 0:  
                ee_pos_target[i] = torch.tensor([x_target, 0.0, 0.6], device=device)    # 环境 0, 3, 6...：沿 X 轴移动（y 和 z 固定）
            elif i % 3 == 1:
                ee_pos_target[i] = torch.tensor([0.3, y_target, 0.6], device=device)    # 环境 1, 4, 7...：沿 Y 轴移动（x 和 z 固定）
            else:
                ee_pos_target[i] = torch.tensor([0.3, 0.0, z_target], device=device)    # 环境 2, 5, 8...：沿 Z 轴移动（x 和 y 固定）
        ee_quat_target = torch.tensor(                                                  # 所有环境的末端执行器目标姿态都设为四元数 [0, 1, 0, 0]（w=0, x=1, y=0, z=0），这表示绕 X 轴旋转 180°——也就是夹爪朝下。这是抓取任务中最常见的末端姿态
            [[0.0, 1.0, 0.0, 0.0]] * args.num_envs,
            device=device,
        )
    elif scenario.robots[0].name == "kinova_gen3_robotiq_2f85":                         # Kinova 机器人的目标轨迹比 Franka 简单——只沿 X 轴移动，从 0.2 线性增长到 0.6（step=200 时），Y 固定 0，Z 固定 0.4
        ee_pos_target = torch.tensor([[0.2 + 0.2 * (step / 100), 0.0, 0.4]], device=device).repeat(args.num_envs, 1)
        ee_quat_target = torch.tensor(                                                  # Kinova 的末端目标姿态是四元数 [0, 0, 1, 0]（w=0, x=0, y=1, z=0），表示绕 Y 轴旋转 180°
            [[0.0, 0.0, 1.0, 0.0]] * args.num_envs,
            device=device,
        )
    elif scenario.robots[0].name == "ur5e_2f85":
        # UR5e 工作空间半径约 0.85m，调整目标范围
        x_target = 0.3 + 0.1 * (step / 100)
        y_target = 0.4 - 0.4 * (step / 100)
        z_target = 0.5 - 0.15 * (step / 100)

        ee_pos_target = torch.zeros((args.num_envs, 3), device=device)
        for i in range(args.num_envs):
            if i % 3 == 0:
                ee_pos_target[i] = torch.tensor([x_target, 0.0, 0.5], device=device)
            elif i % 3 == 1:
                ee_pos_target[i] = torch.tensor([0.3, y_target, 0.5], device=device)
            else:
                ee_pos_target[i] = torch.tensor([0.3, 0.0, z_target], device=device)

        # UR5e 末端朝下的四元数（和 Franka 一样，绕 X 轴 180°）
        ee_quat_target = torch.tensor(
            [[0.0, 1.0, 0.0, 0.0]] * args.num_envs, device=device
        )



    # Get current robot state for seeding
    # IK solver expects original joint order, but state uses alphabetical order
    reorder_idx = handler.get_joint_reindex(scenario.robots[0].name)                    # 获取关节重排索引，reorder_idx 就是从"原始顺序"到"仿真器顺序"的映射
    inverse_reorder_idx = [reorder_idx.index(i) for i in range(len(reorder_idx))]       # 计算逆映射——从"仿真器顺序"到"原始顺序"。比如 reorder_idx = [2, 0, 1] 表示原始第 0 个关节在仿真器中排第 2 位，那逆映射就是 [1, 2, 0]。这样就能把仿真器返回的关节数据转换回 IK 求解器需要的顺序
    curr_robot_q = obs.robots[scenario.robots[0].name].joint_pos[:, inverse_reorder_idx]        # 从观测中取出当前机器人的关节角度，并用逆映射重排成 IK 求解器期望的顺序。[:, inverse_reorder_idx] 是对第二维（关节维度）做重排，第一维是环境数

    # Solve IK
    q_solution, ik_succ = ik_solver.solve_ik_batch(ee_pos_target, ee_quat_target, curr_robot_q) # 批量求解逆运动学：给定目标末端位置 ee_pos_target、目标末端姿态 ee_quat_target、以及当前关节角度 curr_robot_q 作为初始猜测，求解出能到达目标位姿的关节角度 q_solution

    # Process gripper command (fixed open position)
    gripper_binary = torch.ones(scenario.num_envs, device=device)                       # all open，创建夹爪指令，全 1 表示所有环境的夹爪都保持张开。这里用二值表示：1 = 张开，0 = 闭合
    gripper_widths = process_gripper_command(gripper_binary, robot, device)             # 将二值夹爪指令转换为实际的关节宽度值。不同机器人的夹爪参数不同（Franka 有两个手指关节，Kinova/Robotiq 只有一个），这个函数会根据 robot 配置做适配，返回对应的关节位置值
    
    # Compose full joint command
    actions = ik_solver.compose_joint_action(q_solution, gripper_widths, curr_robot_q, return_dict=True)    # 将 IK 求解得到的手臂关节角度和夹爪宽度组合成完整的动作指令。return_dict=True 表示返回字典格式（key 是关节名，value 是目标角度），方便 handler 按名字匹配关节

    handler.set_dof_targets(actions)                                                    # 将目标关节角度发送给仿真器，仿真器内部的 PD 控制器会驱动各关节向目标角度运动（不是瞬间到达，而是有物理过程）
    handler.simulate()                                                                  # 推进仿真一步——物理引擎计算碰撞、力、加速度等，更新所有物体的状态
    obs = handler.get_states(mode="tensor")                                             # 获取仿真步进后的新状态，作为下一步的观测
    # obs, reward, success, time_out, extras = handler.step(actions)

    obs_saver.add(obs)                                                                  # 保存当前帧的观测数据（包括相机图像）到视频缓冲区
    step += 1

obs_saver.save()                                                                        # 循环结束后，将缓冲区中的所有帧编码并写入 MP4 视频文件
