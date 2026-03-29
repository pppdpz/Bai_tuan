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
from metasim.scenario.scene import SceneCfg
from metasim.utils.configclass import configclass
from dm_control import mjcf
from metasim.sim.mujoco.mujoco import MujocoHandler

import xml.etree.ElementTree as ET


@configclass
class BaituanSceneCfg(SceneCfg):
    def __post_init__(self):
        if self.mjx_mjcf_path is None:
            self.mjx_mjcf_path = self.mjcf_path

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

scenario = ScenarioCfg(
    scene=BaituanSceneCfg(
        name="baituan_satellite",
        mjcf_path="asset_baituan/example_scene_y.xml",   # 相对于 RoboVerse 根目录
    ),
    # robots=[],              # 清空！机器人已经在场景 XML 里了
    robots=["ur5e_2f85"],              # 清空！机器人已经在场景 XML 里了
    objects=[],             # 清空！不再单独添加物体
    simulator=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
    gravity=(0.0, 0.0, 0.0),   # 匹配原场景的零重力设置
)

# add cameras
# 给场景添加一个针孔相机：width=1024, height=1024：渲染分辨率 1024×1024；pos=(1.5, -1.5, 1.5)：相机在世界坐标系中的位置（x=1.5, y=-1.5, z=1.5），位于场景的右上方斜后方
# look_at=(0.0, 0.0, 0.0)：相机朝向原点，也就是场景中心，这是一个俯视角度的观察相机，用于捕获场景的 RGB 图像或深度图
# scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]

scenario.cameras = [PinholeCameraCfg(
    width=1024, height=1024,
    pos=(4.0, -4.0, 4.0),      # 拉远到约 6.9m 距离
    look_at=(0.0, 0.0, 0.0)
)]

_original_init_scene = MujocoHandler._init_scene

def resolve_includes(xml_path):
    """递归展开 MuJoCo XML 中所有 <include file="..."/> 标签，
       并将被引入文件中的资产相对路径重写为相对于主 XML 目录的路径。"""
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    tree = ET.parse(xml_path)
    root = tree.getroot()
    _resolve_includes_recursive(root, xml_dir, xml_dir)
    return ET.tostring(root, encoding='unicode')

def _resolve_includes_recursive(element, base_dir, root_dir):
    """展开 <include file="..."/>，同时把被引入内容中的 file 属性路径
       重写为相对于 root_dir 的路径。"""
    i = 0
    while i < len(element):
        child = element[i]
        if child.tag == 'include':
            file_attr = child.get('file')
            inc_path = os.path.normpath(os.path.join(base_dir, file_attr))
            inc_dir = os.path.dirname(inc_path)
            inc_tree = ET.parse(inc_path)
            inc_root = inc_tree.getroot()
            _resolve_includes_recursive(inc_root, inc_dir, root_dir)
            # 重写被引入子树中所有 file 属性的相对路径
            _rebase_file_paths(inc_root, inc_dir, root_dir)
            for j, inc_child in enumerate(list(inc_root)):
                element.insert(i + j, inc_child)
            element.remove(child)
        else:
            _resolve_includes_recursive(child, base_dir, root_dir)
            i += 1

def _rebase_file_paths(element, original_dir, target_dir):
    """将 element 子树中所有 file 属性从相对于 original_dir 转换为相对于 target_dir。"""
    for el in element.iter():
        file_val = el.get('file')
        if file_val and el.tag != 'include':
            abs_path = os.path.normpath(os.path.join(original_dir, file_val))
            new_rel = os.path.relpath(abs_path, target_dir)
            el.set('file', new_rel)

def _patched_init_scene(self):
    if self.scenario.scene is not None:
        xml_path = self.scenario.scene.mjcf_path
        xml_dir = os.path.dirname(os.path.abspath(xml_path))
        # 先递归展开所有 <include>，得到一个完整的无 include 的 XML 字符串
        xml_string = resolve_includes(xml_path)
        mjcf_model = mjcf.from_xml_string(xml_string, model_dir=xml_dir)
        log.info(f"Loaded scene from: {xml_path} (patched, model_dir={xml_dir})")
        return mjcf_model
    else:
        return mjcf.RootElement()

MujocoHandler._init_scene = _patched_init_scene

_original_add_robots = MujocoHandler._add_robots_to_model


_original_set_root_state = MujocoHandler._set_root_state

def _patched_set_root_state(self, obj_name, obj_state, zero_vel=False):
    # 对于内嵌场景中的机器人，跳过 root state 设置
    # 因为机器人位姿已经在场景 XML 中定义，且 fix_base_link=True
    if self.scenario.scene is not None:
        for robot in self.robots:
            if robot.name == obj_name:
                return  # 内嵌机器人，不需要设置 root state
    _original_set_root_state(self, obj_name, obj_state, zero_vel)

MujocoHandler._set_root_state = _patched_set_root_state


_UR5E_BODY_NAMES = [
    "ur5e_base", "shoulder_link", "upper_arm_link", "forearm_link",
    "wrist_1_link", "wrist_2_link", "wrist_3_link", "flange",
    "camera_center", "tcp_link", "robotiq_2f85", "2f85_base", "base",
    "right_driver", "right_coupler", "right_spring_link", "right_follower",
    "right_pad", "right_silicone_pad",
    "left_driver", "left_coupler", "left_spring_link", "left_follower",
    "left_pad", "left_silicone_pad",
]

_UR5E_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "right_driver_joint", "right_coupler_joint", "right_spring_link_joint", "right_follower_joint",
    "left_driver_joint", "left_coupler_joint", "left_spring_link_joint", "left_follower_joint",
]


class _InlineRobotStub:
    """模拟 mjcf.RootElement 的最小接口，供 mj_objects 使用。"""
    model = ""              # 空字符串，使 f"{model_name}/" 变成 "/"，不会匹配任何 body
    full_identifier = ""    # 空字符串前缀

    def __init__(self, root_body_name, body_names, joint_names):
        self.root_body = root_body_name
        self.inline_body_names = set(body_names)
        self.inline_joint_names = set(joint_names)

# ---- 修改 _patched_add_robots_to_model，使用 _InlineRobotStub ----
def _patched_add_robots_to_model_v2(self, mjcf_model):
    if self.scenario.scene is not None:
        for robot in self.robots:
            stub = _InlineRobotStub("ur5e_base", _UR5E_BODY_NAMES, _UR5E_JOINT_NAMES)
            self.mj_objects[robot.name] = stub
            self._mujoco_robot_names.append("")  # 空前缀
    else:
        _original_add_robots(self, mjcf_model)

MujocoHandler._add_robots_to_model = _patched_add_robots_to_model_v2

# ---- Patch _get_states ----
_original_get_states = MujocoHandler._get_states

def _patched_get_states(self, env_ids=None):
    """内嵌场景时，robot 的 body/joint 用裸名查找。"""
    if self.scenario.scene is None:
        return _original_get_states(self, env_ids=env_ids)

    # 内嵌场景：需要导入必要的类型
    import sys
    import numpy as np
    # from metasim.cfg.objects import ArticulationObjCfg
    from metasim.scenario.objects import ArticulationObjCfg

    from metasim.types import ObjectState, RobotState, CameraState, TensorState

    object_states = {}

    # --- objects 部分：内嵌场景通常 objects=[]，但保留兼容 ---
    for obj in self.objects:
        stub = self.mj_objects[obj.name]
        if hasattr(stub, 'root_body'):
            # 内嵌物体（如果有的话）
            obj_body_id = self.physics.model.body(stub.root_body).id
        else:
            model_name = stub.model
            obj_body_id = self.physics.model.body(f"{model_name}/").id

        if isinstance(obj, ArticulationObjCfg):
            joint_names = self._get_joint_names(obj.name, sort=True)
            body_ids_reindex = self._get_body_ids_reindex(obj.name)
            root_np, body_np = self._pack_state([obj_body_id] + body_ids_reindex)
            state = ObjectState(
                root_state=torch.from_numpy(root_np).float().unsqueeze(0),
                body_names=self._get_body_names(obj.name),
                body_state=torch.from_numpy(body_np).float().unsqueeze(0),
                joint_pos=torch.tensor([
                    self.physics.data.joint(jn).qpos.item() for jn in joint_names
                ]).unsqueeze(0),
                joint_vel=torch.tensor([
                    self.physics.data.joint(jn).qvel.item() for jn in joint_names
                ]).unsqueeze(0),
            )
        else:
            root_np, _ = self._pack_state([obj_body_id])
            state = ObjectState(
                root_state=torch.from_numpy(root_np).float().unsqueeze(0),
            )
        object_states[obj.name] = state

    # robots 部分：关键改动——用裸名
    robot_states = {}
    for robot in self.robots:
        stub = self.mj_objects[robot.name]
        if hasattr(stub, 'inline_body_names'):
            # 内嵌机器人：用裸名
            obj_body_id = self.physics.model.body(stub.root_body).id
            joint_names = self._get_joint_names(robot.name, sort=True)
            actuator_reindex = self._get_actuator_reindex(robot.name)
            body_ids_reindex = self._get_body_ids_reindex(robot.name)
            root_np, body_np = self._pack_state([obj_body_id] + body_ids_reindex)

            state = RobotState(
                body_names=self._get_body_names(robot.name),
                root_state=torch.from_numpy(root_np).float().unsqueeze(0),
                body_state=torch.from_numpy(body_np).float().unsqueeze(0),
                # 关键改动：joint 用裸名查找（不加 model_name/ 前缀）
                joint_pos=torch.tensor([
                    self.physics.data.joint(jn).qpos.item() for jn in joint_names
                ]).unsqueeze(0),
                joint_vel=torch.tensor([
                    self.physics.data.joint(jn).qvel.item() for jn in joint_names
                ]).unsqueeze(0),
                joint_pos_target=torch.from_numpy(
                    self.physics.data.ctrl[actuator_reindex].copy()
                ).unsqueeze(0),
                joint_vel_target=torch.from_numpy(self._current_vel_target).unsqueeze(0)
                if self._current_vel_target is not None
                else None,
                joint_effort_target=torch.from_numpy(
                    self.physics.data.actuator_force[actuator_reindex].copy()
                ).unsqueeze(0),
            )
        else:
            # 非内嵌机器人，走原始逻辑
            model_name = stub.model
            obj_body_id = self.physics.model.body(f"{model_name}/").id
            joint_names = self._get_joint_names(robot.name, sort=True)
            actuator_reindex = self._get_actuator_reindex(robot.name)
            body_ids_reindex = self._get_body_ids_reindex(robot.name)
            root_np, body_np = self._pack_state([obj_body_id] + body_ids_reindex)
            state = RobotState(
                body_names=self._get_body_names(robot.name),
                root_state=torch.from_numpy(root_np).float().unsqueeze(0),
                body_state=torch.from_numpy(body_np).float().unsqueeze(0),
                joint_pos=torch.tensor([
                    self.physics.data.joint(f"{model_name}/{jn}").qpos.item() for jn in joint_names
                ]).unsqueeze(0),
                joint_vel=torch.tensor([
                    self.physics.data.joint(f"{model_name}/{jn}").qvel.item() for jn in joint_names
                ]).unsqueeze(0),
                joint_pos_target=torch.from_numpy(
                    self.physics.data.ctrl[actuator_reindex].copy()
                ).unsqueeze(0),
                joint_vel_target=torch.from_numpy(self._current_vel_target).unsqueeze(0)
                if self._current_vel_target is not None
                else None,
                joint_effort_target=torch.from_numpy(
                    self.physics.data.actuator_force[actuator_reindex].copy()
                ).unsqueeze(0),
            )
        robot_states[robot.name] = state

    # cameras 部分：不受影响，直接调用原始逻辑中的 camera 处理
    # 但由于 _get_states 是一个整体方法，我们需要完整处理 camera
    camera_states = {}
    for camera in self.cameras:
        if camera.mount_to is not None:
            mount_obj = self.mj_objects[camera.mount_to]
            if hasattr(mount_obj, 'inline_body_names'):
                camera_id = f"{camera.name}_custom"
            else:
                camera_id = f"{mount_obj.model}/{camera.name}_custom"
        else:
            camera_id = f"{camera.name}_custom"

        rgb = None
        depth = None
        if "rgb" in camera.data_types:
            import sys as _sys
            if _sys.platform == "darwin":
                import mujoco as _mj
                with self._mj_lock:
                    if self.renderer is None or (self.renderer.width, self.renderer.height) != (
                        camera.width, camera.height,
                    ):
                        self.renderer = _mj.Renderer(self._mj_model, width=camera.width, height=camera.height)
                    self._mirror_state_to_native()
                    self.renderer.update_scene(self._mj_data, camera=camera_id)
                    rgb_np = self.renderer.render()
                rgb = torch.from_numpy(rgb_np.copy()).unsqueeze(0)
            else:
                rgb_np = self.physics.render(
                    width=camera.width, height=camera.height, camera_id=camera_id, depth=False
                )
                rgb = torch.from_numpy(np.ascontiguousarray(rgb_np)).unsqueeze(0)

        if "depth" in camera.data_types:
            import sys as _sys
            if _sys.platform == "darwin":
                import mujoco as _mj
                with self._mj_lock:
                    if self.renderer is None or (self.renderer.width, self.renderer.height) != (
                        camera.width, camera.height,
                    ):
                        self.renderer = _mj.Renderer(self._mj_model, width=camera.width, height=camera.height)
                    self._mirror_state_to_native()
                    self.renderer.update_scene(self._mj_data, camera=camera_id)
                    if hasattr(self.renderer, "enable_depth_rendering"):
                        self.renderer.enable_depth_rendering()
                        depth_np = self.renderer.render()
                        self.renderer.disable_depth_rendering()
                    else:
                        raise RuntimeError("Depth rendering not supported.")
                depth = torch.from_numpy(depth_np.copy()).unsqueeze(0)
            else:
                depth_np = self.physics.render(
                    width=camera.width, height=camera.height, camera_id=camera_id, depth=True
                )
                depth = torch.from_numpy(np.ascontiguousarray(depth_np)).unsqueeze(0)

        state = CameraState(rgb=locals().get("rgb", None), depth=locals().get("depth", None))
        camera_states[camera.name] = state

    extras = self.get_extra()
    return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, extras=extras)

MujocoHandler._get_states = _patched_get_states


_original_get_body_names = MujocoHandler._get_body_names

def _patched_get_body_names(self, obj_name, sort=True):
    stub = self.mj_objects.get(obj_name)
    if stub is not None and hasattr(stub, 'inline_body_names'):
        # 内嵌机器人：从 stub 中获取 body 名列表（排除 root body）
        names = [n for n in stub.inline_body_names if n != stub.root_body]
        if sort:
            names.sort()
        return names
    return _original_get_body_names(self, obj_name, sort)

MujocoHandler._get_body_names = _patched_get_body_names


# ---- Patch _get_body_ids_reindex ----
_original_get_body_ids_reindex = MujocoHandler._get_body_ids_reindex

def _patched_get_body_ids_reindex(self, obj_name):
    stub = self.mj_objects.get(obj_name)
    if stub is not None and hasattr(stub, 'inline_body_names'):
        # 内嵌机器人：用 stub 中的 body 名集合过滤物理引擎中的 body
        if not hasattr(self, '_body_ids_reindex_cache'):
            self._body_ids_reindex_cache = {}
        if obj_name not in self._body_ids_reindex_cache:
            body_ids_origin = []
            for bi in range(self.physics.model.nbody):
                body_name = self.physics.model.body(bi).name
                # 属于机器人、且不是 root body 的裸名
                if body_name in stub.inline_body_names and body_name != stub.root_body:
                    body_ids_origin.append(bi)

            # get_body_reindex 内部调用 _get_body_names，已经被 patch 过了
            body_ids_reindex = [body_ids_origin[i] for i in self.get_body_reindex(obj_name)]
            self._body_ids_reindex_cache[obj_name] = body_ids_reindex
        return self._body_ids_reindex_cache[obj_name]
    return _original_get_body_ids_reindex(self, obj_name)

MujocoHandler._get_body_ids_reindex = _patched_get_body_ids_reindex


# ---- Patch _get_joint_names（也需要修复，裸名没有前缀） ----
_original_get_joint_names = MujocoHandler._get_joint_names

def _patched_get_joint_names(self, obj_name, sort=True):
    stub = self.mj_objects.get(obj_name)
    if stub is not None and hasattr(stub, 'inline_joint_names'):
        # 内嵌机器人：从物理引擎中筛选属于机器人的 joint（裸名）
        joint_names = []
        for jid in range(self.physics.model.njnt):
            jname = self.physics.model.joint(jid).name
            if jname in stub.inline_joint_names:
                joint_names.append(jname)
        if sort:
            joint_names.sort()
        return joint_names
    return _original_get_joint_names(self, obj_name, sort)

MujocoHandler._get_joint_names = _patched_get_joint_names


# ---- Patch _get_actuator_names（actuator 也是裸名） ----
_original_get_actuator_names = MujocoHandler._get_actuator_names

def _patched_get_actuator_names(self, robot_name):
    stub = self.mj_objects.get(robot_name)
    if stub is not None and hasattr(stub, 'inline_joint_names'):
        # 内嵌机器人：actuator 名也是裸名，前缀为空字符串
        # 原始逻辑用 startswith("") 会匹配所有 actuator，
        # 需要用 joint_names 集合过滤
        joint_names_set = set(self._get_joint_names(robot_name))
        robot_actuator_names = []
        for i in range(self.physics.model.nu):
            aname = self.physics.model.actuator(i).name
            # actuator 名不一定等于 joint 名，但去掉前缀后应该在 joint_names 中
            # 对于内嵌场景，actuator 名就是裸名（如 "shoulder_pan"、"fingers_actuator"）
            # 需要检查 actuator 对应的 joint/tendon 是否属于机器人
            # 最可靠的方式：检查 actuator 的 trnid 指向的 joint 是否在机器人 joint 集合中
            actuator = self.physics.model.actuator(i)
            trntype = self.physics.model.actuator_trntype[i]
            if trntype == 0:  # joint
                joint_id = self.physics.model.actuator_trnid[i][0]
                joint_name = self.physics.model.joint(joint_id).name
                if joint_name in stub.inline_joint_names:
                    robot_actuator_names.append(aname)
            elif trntype == 1:  # tendon
                # tendon-driven actuator (如 Robotiq 的 fingers_actuator)
                # 检查 tendon 关联的 joint 是否属于机器人
                tendon_id = self.physics.model.actuator_trnid[i][0]
                # 遍历 tendon 的 joint 列表
                # dm_control physics 中可以通过 model.tendon_jnt 获取
                # 简化处理：如果 tendon 名在已知列表中，认为属于机器人
                robot_actuator_names.append(aname)
        return robot_actuator_names
    return _original_get_actuator_names(self, robot_name)

MujocoHandler._get_actuator_names = _patched_get_actuator_names


# ---- Patch _get_actuator_states（也用了前缀匹配） ----
_original_get_actuator_states = MujocoHandler._get_actuator_states

def _patched_get_actuator_states(self, obj_name):
    stub = self.mj_objects.get(obj_name)
    if stub is not None and hasattr(stub, 'inline_joint_names'):
        actuator_states = {
            "dof_pos_target": {},
            "dof_vel_target": {},
            "dof_torque": {},
        }
        robot_actuator_names = self._get_actuator_names(obj_name)
        for i in range(self.physics.model.nu):
            aname = self.physics.model.actuator(i).name
            if aname in robot_actuator_names:
                actuator_states["dof_pos_target"][aname] = float(
                    self.physics.data.ctrl[i].item()
                )
                actuator_states["dof_vel_target"][aname] = None
                actuator_states["dof_torque"][aname] = float(
                    self.physics.data.actuator_force[i].item()
                )
        return actuator_states
    return _original_get_actuator_states(self, obj_name)

MujocoHandler._get_actuator_states = _patched_get_actuator_states

# ---- Patch set_dof_targets：URDF joint 名 → 场景 XML actuator 名映射 ----
_JOINT_TO_ACTUATOR = {
    "shoulder_pan_joint": "shoulder_pan",
    "shoulder_lift_joint": "shoulder_lift",
    "elbow_joint": "elbow",
    "wrist_1_joint": "wrist_1",
    "wrist_2_joint": "wrist_2",
    "wrist_3_joint": "wrist_3",
}

# 这 8 个夹爪 joint 全部映射到一个 tendon 驱动的 fingers_actuator
_GRIPPER_JOINTS = {
    "right_driver_joint", "right_coupler_joint",
    "right_spring_link_joint", "right_follower_joint",
    "left_driver_joint", "left_coupler_joint",
    "left_spring_link_joint", "left_follower_joint",
}

_original_set_dof_targets = MujocoHandler.set_dof_targets

def _patched_set_dof_targets(self, actions):
    if self.scenario.scene is None:
        return _original_set_dof_targets(self, actions)

    # actions 可能是 list（多环境）或 dict（单环境，set_dof_targets 内部会取 [0]）
    # compose_joint_action 返回的是 list of dict
    if isinstance(actions, list):
        remapped_actions = []
        for env_action in actions:
            remapped_env = {}
            for robot_name, payload in env_action.items():
                new_payload = {}
                for key, val in payload.items():
                    if key == "dof_pos_target":
                        new_targets = {}
                        gripper_val = None
                        for jn, jval in val.items():
                            if jn in _JOINT_TO_ACTUATOR:
                                # 手臂关节：重命名
                                new_targets[_JOINT_TO_ACTUATOR[jn]] = jval
                            elif jn in _GRIPPER_JOINTS:
                                # 夹爪关节：取 right_driver_joint 的值作为 fingers_actuator 输入
                                # fingers_actuator ctrlrange 是 0~255，但 driver_joint range 是 0~0.8
                                # 需要做线性映射：driver_val / 0.8 * 255
                                if gripper_val is None and jn == "right_driver_joint":
                                    gripper_val = jval
                            else:
                                new_targets[jn] = jval
                        if gripper_val is not None:
                            new_targets["fingers_actuator"] = gripper_val / 0.8 * 255.0
                        else:
                            # 夹爪全开（driver=0）→ actuator=0
                            new_targets["fingers_actuator"] = 0.0
                        new_payload[key] = new_targets
                    elif key == "dof_vel_target" and val is not None:
                        new_vel = {}
                        for jn, jval in val.items():
                            if jn in _JOINT_TO_ACTUATOR:
                                new_vel[_JOINT_TO_ACTUATOR[jn]] = jval
                            elif jn not in _GRIPPER_JOINTS:
                                new_vel[jn] = jval
                        new_payload[key] = new_vel
                    else:
                        new_payload[key] = val
                remapped_env[robot_name] = new_payload
            remapped_actions.append(remapped_env)
        return _original_set_dof_targets(self, remapped_actions)
    else:
        # dict 格式（单环境），包装成 list 递归处理
        return _patched_set_dof_targets(self, [actions])

MujocoHandler.set_dof_targets = _patched_set_dof_targets



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
