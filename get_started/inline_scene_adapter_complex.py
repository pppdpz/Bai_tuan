# inline_scene_adapter.py
"""Adapter for MuJoCo scenes with inline (non-namespaced) robots.

Overrides MujocoHandler methods to handle robots embedded directly
in scene XML without dm_control's namespace prefix.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from dm_control import mjcf
from loguru import logger as log

from metasim.scenario.objects import ArticulationObjCfg
from metasim.sim.mujoco.mujoco import MujocoHandler
from metasim.types import CameraState, ObjectState, RobotState, TensorState


# ---------------------------------------------------------------------------
# XML include resolver
# ---------------------------------------------------------------------------

def resolve_includes(xml_path: str) -> str:
    """Recursively expand all <include file="..."/> in a MuJoCo XML.

    Rewrites asset paths in included files to be relative to the root XML dir.
    """
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    tree = ET.parse(xml_path)
    root = tree.getroot()
    _expand(root, xml_dir, xml_dir)
    return ET.tostring(root, encoding="unicode")


def _expand(element: ET.Element, base_dir: str, root_dir: str) -> None:
    i = 0
    while i < len(element):
        child = element[i]
        if child.tag == "include":
            inc_path = os.path.normpath(os.path.join(base_dir, child.get("file")))
            inc_dir = os.path.dirname(inc_path)
            inc_root = ET.parse(inc_path).getroot()
            _expand(inc_root, inc_dir, root_dir)
            _rebase_paths(inc_root, inc_dir, root_dir)
            for j, inc_child in enumerate(list(inc_root)):
                element.insert(i + j, inc_child)
            element.remove(child)
        else:
            _expand(child, base_dir, root_dir)
            i += 1


def _rebase_paths(element: ET.Element, src_dir: str, dst_dir: str) -> None:
    for el in element.iter():
        fv = el.get("file")
        if fv and el.tag != "include":
            el.set("file", os.path.relpath(
                os.path.normpath(os.path.join(src_dir, fv)), dst_dir
            ))


# ---------------------------------------------------------------------------
# Inline robot descriptor
# ---------------------------------------------------------------------------

@dataclass
class InlineRobotDescriptor:
    """Describes a robot embedded directly in the scene XML (no namespace)."""

    root_body: str
    body_names: Set[str]
    joint_names: Set[str]

    # Actuator name mapping: URDF joint name -> scene XML actuator name
    joint_to_actuator: Dict[str, str] = field(default_factory=dict)
    # Joint names that map to a single tendon-driven gripper actuator
    gripper_joints: Set[str] = field(default_factory=set)
    gripper_actuator_name: str = "fingers_actuator"
    # Linear mapping: driver_joint_range -> actuator_ctrl_range
    gripper_driver_joint: str = "right_driver_joint"
    gripper_driver_range: float = 0.8
    gripper_actuator_range: float = 255.0

    # Stub fields expected by MujocoHandler internals
    model: str = ""
    full_identifier: str = ""


# ---------------------------------------------------------------------------
# Pre-built UR5e + Robotiq 2F-85 descriptor
# ---------------------------------------------------------------------------

UR5E_2F85_DESCRIPTOR = InlineRobotDescriptor(
    root_body="ur5e_base",
    body_names={
        "ur5e_base", "shoulder_link", "upper_arm_link", "forearm_link",
        "wrist_1_link", "wrist_2_link", "wrist_3_link", "flange",
        "camera_center", "tcp_link", "robotiq_2f85", "2f85_base", "base",
        "right_driver", "right_coupler", "right_spring_link", "right_follower",
        "right_pad", "right_silicone_pad",
        "left_driver", "left_coupler", "left_spring_link", "left_follower",
        "left_pad", "left_silicone_pad",
    },
    joint_names={
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        "right_driver_joint", "right_coupler_joint",
        "right_spring_link_joint", "right_follower_joint",
        "left_driver_joint", "left_coupler_joint",
        "left_spring_link_joint", "left_follower_joint",
    },
    joint_to_actuator={
        "shoulder_pan_joint": "shoulder_pan",
        "shoulder_lift_joint": "shoulder_lift",
        "elbow_joint": "elbow",
        "wrist_1_joint": "wrist_1",
        "wrist_2_joint": "wrist_2",
        "wrist_3_joint": "wrist_3",
    },
    gripper_joints={
        "right_driver_joint", "right_coupler_joint",
        "right_spring_link_joint", "right_follower_joint",
        "left_driver_joint", "left_coupler_joint",
        "left_spring_link_joint", "left_follower_joint",
    },
)


# ---------------------------------------------------------------------------
# Adapter: apply all patches in one place
# ---------------------------------------------------------------------------

def _is_inline(stub) -> bool:
    return isinstance(stub, InlineRobotDescriptor)


def install_inline_scene_patches(
    descriptor: InlineRobotDescriptor = UR5E_2F85_DESCRIPTOR,
) -> None:
    """Monkey-patch MujocoHandler to support inline-robot scenes.

    Call once before ``get_handler()``.
    """
    _patch_init_scene()
    _patch_add_robots(descriptor)
    _patch_set_root_state()
    _patch_name_lookups()
    _patch_actuator_lookups()
    _patch_get_states()
    _patch_set_dof_targets(descriptor)


# -- individual patches (private) ------------------------------------------

def _patch_init_scene() -> None:
    orig = MujocoHandler._init_scene

    def _init_scene(self):
        if self.scenario.scene is not None:
            xml_path = self.scenario.scene.mjcf_path
            xml_dir = os.path.dirname(os.path.abspath(xml_path))
            xml_string = resolve_includes(xml_path)
            model = mjcf.from_xml_string(xml_string, model_dir=xml_dir)
            log.info(f"Loaded inline scene: {xml_path}")
            return model
        return orig(self)

    MujocoHandler._init_scene = _init_scene


def _patch_add_robots(desc: InlineRobotDescriptor) -> None:
    orig = MujocoHandler._add_robots_to_model

    def _add_robots(self, mjcf_model):
        if self.scenario.scene is not None:
            for robot in self.robots:
                self.mj_objects[robot.name] = desc
                self._mujoco_robot_names.append("")
        else:
            orig(self, mjcf_model)

    MujocoHandler._add_robots_to_model = _add_robots


def _patch_set_root_state() -> None:
    orig = MujocoHandler._set_root_state

    def _set_root_state(self, obj_name, obj_state, zero_vel=False):
        if self.scenario.scene is not None:
            for robot in self.robots:
                if robot.name == obj_name:
                    return  # pose defined in scene XML
        orig(self, obj_name, obj_state, zero_vel)

    MujocoHandler._set_root_state = _set_root_state


def _patch_name_lookups() -> None:
    """Patch _get_body_names, _get_body_ids_reindex, _get_joint_names."""
    _orig_body_names = MujocoHandler._get_body_names
    _orig_body_ids = MujocoHandler._get_body_ids_reindex
    _orig_joint_names = MujocoHandler._get_joint_names

    def _get_body_names(self, obj_name, sort=True):
        stub = self.mj_objects.get(obj_name)
        if _is_inline(stub):
            names = [n for n in stub.body_names if n != stub.root_body]
            return sorted(names) if sort else names
        return _orig_body_names(self, obj_name, sort)

    def _get_body_ids_reindex(self, obj_name):
        stub = self.mj_objects.get(obj_name)
        if not _is_inline(stub):
            return _orig_body_ids(self, obj_name)
        cache = getattr(self, "_body_ids_reindex_cache", {})
        if obj_name not in cache:
            ids = [
                bi for bi in range(self.physics.model.nbody)
                if self.physics.model.body(bi).name in stub.body_names
                and self.physics.model.body(bi).name != stub.root_body
            ]
            cache[obj_name] = [ids[i] for i in self.get_body_reindex(obj_name)]
            self._body_ids_reindex_cache = cache
        return cache[obj_name]

    def _get_joint_names(self, obj_name, sort=True):
        stub = self.mj_objects.get(obj_name)
        if _is_inline(stub):
            names = [
                self.physics.model.joint(j).name
                for j in range(self.physics.model.njnt)
                if self.physics.model.joint(j).name in stub.joint_names
            ]
            return sorted(names) if sort else names
        return _orig_joint_names(self, obj_name, sort)

    MujocoHandler._get_body_names = _get_body_names
    MujocoHandler._get_body_ids_reindex = _get_body_ids_reindex
    MujocoHandler._get_joint_names = _get_joint_names


def _patch_actuator_lookups() -> None:
    _orig_names = MujocoHandler._get_actuator_names
    _orig_states = MujocoHandler._get_actuator_states

    def _get_actuator_names(self, robot_name):
        stub = self.mj_objects.get(robot_name)
        if not _is_inline(stub):
            return _orig_names(self, robot_name)
        result = []
        for i in range(self.physics.model.nu):
            trntype = self.physics.model.actuator_trntype[i]
            if trntype == 0:  # joint-driven
                jid = self.physics.model.actuator_trnid[i][0]
                if self.physics.model.joint(jid).name in stub.joint_names:
                    result.append(self.physics.model.actuator(i).name)
            elif trntype == 1:  # tendon-driven (e.g. Robotiq fingers)
                result.append(self.physics.model.actuator(i).name)
        return result

    def _get_actuator_states(self, obj_name):
        stub = self.mj_objects.get(obj_name)
        if not _is_inline(stub):
            return _orig_states(self, obj_name)
        names = set(self._get_actuator_names(obj_name))
        states = {"dof_pos_target": {}, "dof_vel_target": {}, "dof_torque": {}}
        for i in range(self.physics.model.nu):
            aname = self.physics.model.actuator(i).name
            if aname in names:
                states["dof_pos_target"][aname] = float(self.physics.data.ctrl[i])
                states["dof_vel_target"][aname] = None
                states["dof_torque"][aname] = float(self.physics.data.actuator_force[i])
        return states

    MujocoHandler._get_actuator_names = _get_actuator_names
    MujocoHandler._get_actuator_states = _get_actuator_states


def _patch_get_states() -> None:
    """Patch _get_states for inline robots (bare joint/body names)."""
    orig = MujocoHandler._get_states

    def _get_states(self, env_ids=None):
        if self.scenario.scene is None:
            return orig(self, env_ids=env_ids)

        # Objects
        object_states = {}
        for obj in self.objects:
            stub = self.mj_objects[obj.name]
            bid = (
                self.physics.model.body(stub.root_body).id
                if _is_inline(stub)
                else self.physics.model.body(f"{stub.model}/").id
            )
            if isinstance(obj, ArticulationObjCfg):
                jnames = self._get_joint_names(obj.name, sort=True)
                bids = self._get_body_ids_reindex(obj.name)
                root_np, body_np = self._pack_state([bid] + bids)
                state = ObjectState(
                    root_state=torch.from_numpy(root_np).float().unsqueeze(0),
                    body_names=self._get_body_names(obj.name),
                    body_state=torch.from_numpy(body_np).float().unsqueeze(0),
                    joint_pos=torch.tensor([self.physics.data.joint(j).qpos.item() for j in jnames]).unsqueeze(0),
                    joint_vel=torch.tensor([self.physics.data.joint(j).qvel.item() for j in jnames]).unsqueeze(0),
                )
            else:
                root_np, _ = self._pack_state([bid])
                state = ObjectState(root_state=torch.from_numpy(root_np).float().unsqueeze(0))
            object_states[obj.name] = state

        # Robots
        robot_states = {}
        for robot in self.robots:
            stub = self.mj_objects[robot.name]
            inline = _is_inline(stub)
            bid = (
                self.physics.model.body(stub.root_body).id
                if inline
                else self.physics.model.body(f"{stub.model}/").id
            )
            jnames = self._get_joint_names(robot.name, sort=True)
            act_idx = self._get_actuator_reindex(robot.name)
            bids = self._get_body_ids_reindex(robot.name)
            root_np, body_np = self._pack_state([bid] + bids)

            def _jq(jn):
                key = jn if inline else f"{stub.model}/{jn}"
                return self.physics.data.joint(key).qpos.item()

            def _jv(jn):
                key = jn if inline else f"{stub.model}/{jn}"
                return self.physics.data.joint(key).qvel.item()

            robot_states[robot.name] = RobotState(
                body_names=self._get_body_names(robot.name),
                root_state=torch.from_numpy(root_np).float().unsqueeze(0),
                body_state=torch.from_numpy(body_np).float().unsqueeze(0),
                joint_pos=torch.tensor([_jq(j) for j in jnames]).unsqueeze(0),
                joint_vel=torch.tensor([_jv(j) for j in jnames]).unsqueeze(0),
                joint_pos_target=torch.from_numpy(self.physics.data.ctrl[act_idx].copy()).unsqueeze(0),
                joint_vel_target=(
                    torch.from_numpy(self._current_vel_target).unsqueeze(0)
                    if self._current_vel_target is not None else None
                ),
                joint_effort_target=torch.from_numpy(
                    self.physics.data.actuator_force[act_idx].copy()
                ).unsqueeze(0),
            )

        # Cameras — delegate to a helper to avoid duplicating render logic
        camera_states = _render_cameras(self)

        return TensorState(
            objects=object_states,
            robots=robot_states,
            cameras=camera_states,
            extras=self.get_extra(),
        )

    MujocoHandler._get_states = _get_states


def _render_cameras(handler) -> dict:
    """Render all cameras. Extracted to keep _get_states focused."""
    import sys as _sys

    camera_states = {}
    for camera in handler.cameras:
        if camera.mount_to is not None:
            mount = handler.mj_objects[camera.mount_to]
            prefix = "" if _is_inline(mount) else f"{mount.model}/"
            camera_id = f"{prefix}{camera.name}_custom"
        else:
            camera_id = f"{camera.name}_custom"

        rgb = depth = None

        if "rgb" in camera.data_types:
            if _sys.platform == "darwin":
                rgb = _render_rgb_macos(handler, camera, camera_id)
            else:
                rgb_np = handler.physics.render(
                    width=camera.width, height=camera.height,
                    camera_id=camera_id, depth=False,
                )
                rgb = torch.from_numpy(np.ascontiguousarray(rgb_np)).unsqueeze(0)

        if "depth" in camera.data_types:
            if _sys.platform == "darwin":
                depth = _render_depth_macos(handler, camera, camera_id)
            else:
                depth_np = handler.physics.render(
                    width=camera.width, height=camera.height,
                    camera_id=camera_id, depth=True,
                )
                depth = torch.from_numpy(np.ascontiguousarray(depth_np)).unsqueeze(0)

        camera_states[camera.name] = CameraState(rgb=rgb, depth=depth)
    return camera_states


def _ensure_renderer(handler, camera):
    import mujoco as _mj
    if handler.renderer is None or (handler.renderer.width, handler.renderer.height) != (
        camera.width, camera.height,
    ):
        handler.renderer = _mj.Renderer(handler._mj_model, width=camera.width, height=camera.height)


def _render_rgb_macos(handler, camera, camera_id):
    with handler._mj_lock:
        _ensure_renderer(handler, camera)
        handler._mirror_state_to_native()
        handler.renderer.update_scene(handler._mj_data, camera=camera_id)
        return torch.from_numpy(handler.renderer.render().copy()).unsqueeze(0)


def _render_depth_macos(handler, camera, camera_id):
    with handler._mj_lock:
        _ensure_renderer(handler, camera)
        handler._mirror_state_to_native()
        handler.renderer.update_scene(handler._mj_data, camera=camera_id)
        if not hasattr(handler.renderer, "enable_depth_rendering"):
            raise RuntimeError("Depth rendering not supported on this mujoco version.")
        handler.renderer.enable_depth_rendering()
        depth_np = handler.renderer.render()
        handler.renderer.disable_depth_rendering()
        return torch.from_numpy(depth_np.copy()).unsqueeze(0)


def _patch_set_dof_targets(desc: InlineRobotDescriptor) -> None:
    orig = MujocoHandler.set_dof_targets

    def _remap_action(env_action: dict) -> dict:
        remapped = {}
        for robot_name, payload in env_action.items():
            new_payload = {}
            for key, val in payload.items():
                if key == "dof_pos_target":
                    new_payload[key] = _remap_pos_targets(val, desc)
                elif key == "dof_vel_target" and val is not None:
                    new_payload[key] = {
                        desc.joint_to_actuator.get(jn, jn): jv
                        for jn, jv in val.items()
                        if jn not in desc.gripper_joints
                    }
                else:
                    new_payload[key] = val
            remapped[robot_name] = new_payload
        return remapped


    def _set_dof_targets(self, actions):
        if self.scenario.scene is None:
            return orig(self, actions)
        
        # 处理 list/dict 格式
        if isinstance(actions, list):
            actions = actions[0]
        
        self._actions_cache = actions
        
        for robot_name, payload in actions.items():
            targets = payload["dof_pos_target"]
            remapped = _remap_pos_targets(targets, desc)
            
            for actuator_name, value in remapped.items():
                aid = self.physics.model.actuator(actuator_name).id
                self.physics.data.ctrl[aid] = value



    MujocoHandler.set_dof_targets = _set_dof_targets


def _remap_pos_targets(targets: dict, desc: InlineRobotDescriptor) -> dict:
    new = {}
    gripper_val = None
    for jn, jv in targets.items():
        if jn in desc.joint_to_actuator:
            new[desc.joint_to_actuator[jn]] = jv
        elif jn in desc.gripper_joints:
            if gripper_val is None and jn == desc.gripper_driver_joint:
                gripper_val = jv
        else:
            new[jn] = jv
    scale = desc.gripper_actuator_range / desc.gripper_driver_range
    new[desc.gripper_actuator_name] = (gripper_val or 0.0) * scale
    return new
