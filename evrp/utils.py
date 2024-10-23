import torch
import torch.nn.functional as F
import numpy as np
from tensordict.tensordict import TensorDict
from typing import Optional

class MDPPUtils:
    """
    多旅行商路径问题(MDPP)工具类，包含距离计算、动作掩码更新、奖励计算和完成条件检查等功能。
    """

    def __init__(self, generator, dist_mode: str = "L2", reward_mode: str = "minsum"):
        """
        初始化 MDPPUtils。

        参数：
            generator (MDPPGenerator): 数据生成器实例。
        """
        self.generator = generator


    def calculate_distance(self, selected_nodes: torch.Tensor) -> torch.Tensor:
        """
        计算所选节点的距离。

        参数：
            selected_nodes (Tensor): 选择的节点索引 [num_vehicles]

        返回：
            Tensor: 每辆车移动的距离 [num_vehicles]
        """
        # 获取车辆当前的位置和目标位置
        current_positions = self.generator.loc_tensor[self.generator.vehicle_start_positions]
        target_positions = self.generator.loc_tensor[selected_nodes]

        distances = torch.norm(target_positions - current_positions, p=2, dim=-1)
        return distances

    def update_action_mask(self, selected_nodes: torch.Tensor) -> torch.Tensor:
        """
        更新动作掩码，防止车辆选择无效或已访问的节点。

        参数：
            selected_nodes (Tensor): 选择的节点索引 [num_vehicles]

        返回：
            Tensor: 更新后的动作掩码 [num_vehicles, num_loc + num_depot]
        """
        batch_size, num_vehicles = selected_nodes.shape
        action_mask = self.generator.action_mask.clone()

        for b in range(batch_size):
            for v in range(num_vehicles):
                node = selected_nodes[b, v]
                # 假设选择节点后该节点不可再次选择
                action_mask[b, v, node] = False
        return action_mask

    def check_done(self, action_mask: torch.Tensor) -> torch.Tensor:
        """
        检查是否所有车辆都已完成任务。

        参数：
            action_mask (Tensor): 当前的动作掩码 [batch_size, num_vehicles, num_loc + num_depot]

        返回：
            Tensor: 完成标志 [batch_size, num_vehicles]
        """
        done = ~action_mask.any(dim=-1)
        return done

    def calculate_reward(self, td: TensorDict, distances: torch.Tensor) -> torch.Tensor:
        """
        根据奖励模式计算奖励。

        参数：
            td (TensorDict): 当前的状态字典。
            distances (Tensor): 每辆车移动的距离 [num_vehicles]

        返回：
            Tensor: 奖励值 [num_vehicles]
        """
        if self.reward_mode == "minsum":
            reward = -distances
        elif self.reward_mode == "minmax":
            max_distance = td["total_distance"].max(dim=-1)[0]
            reward = -max_distance
        else:
            raise ValueError(f"不支持的奖励模式: {self.reward_mode}")
        return reward

    def generate_initial_data(self, batch_size: Optional[int] = 1) -> TensorDict:
        """
        生成初始数据。

        参数：
            batch_size (int): 批量大小。

        返回：
            TensorDict: 初始数据字典。
        """
        # 从生成器获取初始数据
        initial_data = self.generator._generate(batch_size)
        return initial_data

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        获取当前的动作掩码。

        参数：
            td (TensorDict): 当前的状态字典。

        返回：
            Tensor: 动作掩码 [num_vehicles, num_loc + num_depot]
        """
        return td["action_mask"]

    def update_state_after_step(self, td: TensorDict, selected_nodes: torch.Tensor, distances: torch.Tensor):
        """
        更新状态字典在执行一步动作后的信息。

        参数：
            td (TensorDict): 当前的状态字典。
            selected_nodes (Tensor): 选择的节点索引 [num_vehicles]
            distances (Tensor): 每辆车移动的距离 [num_vehicles]
        """
        td["vehicle_positions"] = selected_nodes
        td["total_distance"] += distances
        td["action_mask"] = self.update_action_mask(selected_nodes)
        td["done"] = self.check_done(td["action_mask"])
        td["reward"] = self.calculate_reward(td, distances)
