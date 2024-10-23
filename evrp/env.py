import torch
from tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)
from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index

from evrp.generator import EVRGenerator

from typing import Optional


from .utils import MDPPUtils

class EVREnv(RL4COEnvBase):
    """Electric Vehicle Routing Planning environment
    At each step,the agent choose a bus_stop to visit for each vehicle.
    Currently,the reward is 0 unless all vehicles has arrived or stopped.
    In this case, the reward is the sum of the remaining power of all vehicles.

    Observations:
        - locations of each bus stop.
        - the current location of each vehicle
        - 

    Constraints:
        - 
        - 

    Finish condition:
        - each vehicles has arrived end points or stopped(time consumed or energy comsumed)
    
    Reward:
        - (max) the sum of the remaining power of all vehicles

    Args:


    """


    name = "evr"

    def __init__(
        self,
        generator: EVRGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = EVRGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)
        self.utils = MDPPUtils(self.generator)



    def _reset(self, td: Optional[TensorDict] = None,  batch_size=None) -> TensorDict:
        # Initialize locations
        device = td.device
        init_locs = td["locs"]
        min_locs = init_locs.min(dim=0, keepdim=True).values
        max_locs = init_locs.max(dim=0, keepdim=True).values
        normalized_locs = (init_locs - min_locs) / (max_locs - min_locs)

        # # We do not enforce loading from self for flexibility
        num_loc = normalized_locs.shape[-2]

        if batch_size is None:
            batch_size = 1
        
        dist_mat = td["distance_matrix"] # distance_matirx
        chg_mat = td["charging_matrix"]  # charging_matirx 生成充电路段矩阵
        car_speeds = td["vehicle_speeds"]   # the speed of car
        cur_battery = td["initial_battery"]    # the battery of battery
        bat_cap = td["battery_capacities"]  # the capacity of battery capacity
        energy_consume_velo = td["energy_consumption"] # the velocity of energy consume velocity
        deadline_times = td["deadline_times"] # the deadtime of arrive time of cars

        # 生成车辆起点和终点
        current_nodes = torch.randint(0, td["distance_matrix"].shape[1], batch_size)
        end_nodes = torch.randint(0, td["distance_matrix"].shape[1], batch_size)
        while torch.any(current_nodes == end_nodes):
            end_nodes = torch.where(current_nodes == end_nodes, torch.randint(0, td["distance_matrix"].shape[1], batch_size), end_nodes)
        available = torch.ones(
            (*batch_size, num_loc), dtype=torch.bool, device=device
        )  # 1 means not visited, i.e. action is allowed
        available[torch.arange(batch_size[0]), current_nodes] = False  # 将current_nodes对应的索引元素值设为False

        #available[torch.arange(batch_size).unsqueeze(1), end_nodes.squeeze(-1)] = False

        charging_ratio = td["charging_ratio"]
        i = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        stopped = (current_nodes == end_nodes)
        arrived = (current_nodes == end_nodes)



        return TensorDict(
            {
                "locs": normalized_locs,
                "first_node": current_nodes,
                "current_node": current_nodes,
                "prior_node": current_nodes,
                "end_node": end_nodes,
                "dist_mat": dist_mat,
                "charge_mat" : chg_mat,
                "charging_ratio": charging_ratio,
                "car_speeds" : car_speeds,
                "remaining_battery" : cur_battery,
                "bat_cap": bat_cap,
                "energy_consume_velocity" : energy_consume_velo,
                "remaining_time": deadline_times,
                "i": i,
                "action_mask": available,
                "stopped": stopped,
                "arrived" : arrived,
                "reward": torch.zeros((*batch_size, 1), dtype=torch.float32),
            },
            batch_size=batch_size,
        )

    def _step(self, td: TensorDict) -> TensorDict:
        
        prior_node= td["prior_node"] # [batch]
        current_node = td["action"]  # [batch]
        end_node = td["end_node"]
        first_node = current_node if td["i"].all() == 0 else td["first_node"]
        remaining_time = td["remaining_time"]
        remaining_battery = td["remaining_battery"]
        battery_capacity = td["bat_cap"]
        energy_comsume_velo = td["energy_consume_velocity"]
        dist_mat = td["dist_mat"]
        chg_mat = td["charge_mat"]
        car_speeds = td["car_speeds"]
        charging_ratio = td["charging_ratio"]
        stopped = td["stopped"]
        arrived = td["arrived"]

        # # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )
        
        # 已经达到目的地或不满足约束的车俩不再计算
        batch_indices = torch.arange(*td.batch_size).to(self.device)
        stopped_indices = td["stopped"].squeeze(0)
        active_indices =  batch_indices[~stopped_indices]

        # 时间约束
        remaining_time[active_indices] = (
            remaining_time[active_indices]
              - (dist_mat[active_indices,prior_node[active_indices], current_node[active_indices]] / car_speeds[active_indices,prior_node[active_indices], current_node[active_indices]])
        )
        # 将remaining_time小于0的值对应的索引位置在stopped中置为True
        time_out_mask = (remaining_time[active_indices]).squeeze(0) <= 0
        stopped[active_indices[time_out_mask].unsqueeze(0)] = True
        
        # 能量约束
        remaining_battery[active_indices] = torch.min(
            remaining_battery[active_indices] 
                - energy_comsume_velo[active_indices]
                  *dist_mat[active_indices,prior_node[active_indices], current_node[active_indices]]
                    + chg_mat[active_indices, prior_node[active_indices], current_node[active_indices]].float()
                        *charging_ratio[active_indices]*(dist_mat[active_indices, prior_node[active_indices], current_node[active_indices]]/car_speeds[active_indices,prior_node[active_indices], current_node[active_indices]]),
                battery_capacity[active_indices]
        )
        # 将remaining_battery小于0的值对应的索引位置在stopped中置为True
        battery_out_mask = remaining_battery[active_indices].squeeze(0) <= 0
        stopped[active_indices[battery_out_mask].unsqueeze(0)] = True
        
        remaining_time = torch.clamp(remaining_time, min=0)
        remaining_battery = torch.clamp(remaining_battery, min=0)

        unstopped_indices = torch.where(~stopped)[0]
        if unstopped_indices.shape[0]==0:
            arrived[0] = False
        else:
            arrived[unstopped_indices] = (current_node[unstopped_indices]==end_node[unstopped_indices])


        # 更新剩余电量
        #remaining_battery = torch.where(stopped, torch.zeros_like(remaining_battery), remaining_battery)
        # 更新剩余时间  
        #remaining_time = torch.where(stopped, torch.zeros_like(remaining_time), remaining_time)
        # 更新停止和到达标志
         #stopped = torch.where(stopped, torch.ones_like(stopped), stopped)
        #arrived = torch.where(arrived, torch.ones_like(arrived), arrived)
        # We are done there are no unvisited locations


        if "remaining_battery_sequence" in td:
            remaining_battery_sequence = torch.cat([td["remaining_battery_sequence"], remaining_battery.unsqueeze(1)], dim=1)
        else:
            remaining_battery_sequence = remaining_battery.unsqueeze(1)



        if "remaining_time_sequence" in td:
            remaining_time_sequence = torch.cat([td["remaining_time_sequence"], remaining_time.unsqueeze(1)], dim=1)

        else:
            remaining_time_sequence = remaining_time.unsqueeze(1)



        done = torch.logical_or(stopped, arrived)

        # print(done.shape)
        # print(remaining_battery_sequence)

        td.update(
            {
                "remaining_battery": remaining_battery,
                "remaining_time": remaining_time,
                "stopped": stopped,
                "action_mask": available,
                "arrived": arrived,
                "i": td["i"] + 1,
                "done": done,
                "remaining_time_sequence": remaining_time_sequence,
                "remaining_battery_sequence": remaining_battery_sequence
            },
        )

        return td

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> TensorDict:
        
        #reward = torch.zeros_like(td["remaining_battery"])
        #arrived_indices = td["arrived"]==True
        #remaining_battery = td["remaining_battery"]
        reward = td["remaining_battery"]


        return reward

    def _make_spec(self, generator: EVRGenerator):
        self.observation_spec = CompositeSpec(
            locs= BoundedTensorSpec(
                low=min(generator.min_longitude,generator.min_latitude),
                high=max(generator.max_longitude,generator.max_latitude),
                shape=(generator.num_bus_stops, 2),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            remaining_battery=BoundedTensorSpec(
                low=generator.min_battery_capacity,
                high=generator.max_battery_capacity,
                shape=(1),
                dtype=torch.float32,
            )
            ,
            remaining_time=BoundedTensorSpec(
                low=0,
                high=100,
                shape=(1),
                dtype=torch.float32,                
            )
            ,
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(self.generator.num_bus_stops),
                dtype=torch.bool,
            ),
            i=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.int64,
            ),
            stopped=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.bool,
            ),
            arrived=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.bool,
            ),          
            done=UnboundedDiscreteTensorSpec(
                shape=(1),
                dtype=torch.bool,
            ),
            shape=(),
        )

        self.action_spec = BoundedTensorSpec(
            shape=(1),
            dtype=torch.int64,
            low=0,
            high=generator.num_bus_stops,
        )

        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1))
        self.stopped_spec = UnboundedDiscreteTensorSpec(shape=(1), dtype=torch.bool)
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1), dtype=torch.bool)


    def check_solution_validity(self, td: TensorDict, actions: torch.Tensor) -> None:
        pass
