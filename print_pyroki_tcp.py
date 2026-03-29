import torch
import metasim.utils.ik_solver as ik
from metasim.scenario.scene import SceneCfg

robot_cfg = ik.load_robot_cfg("ur5e_2f85", "pyroki")
print("Pyroki EE link:", getattr(robot_cfg, "ee_link", "UNKNOWN"))
