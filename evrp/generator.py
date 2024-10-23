from typing import Optional,Union,Callable
import random

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class EVRGenerator(Generator):
    """
    电动汽车调度问题(electric vehicle routing, evr)数据生成器。

    Args:
        num_vehicles (int): 电动汽车的数量。
        num_bus_stops (int): 巴士站台的数量。
        vehicles_velocity(int): 行驶速度(km/h)
        seed (Optional[int]): 随机种子，用于结果复现。

    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_bus_stops, 2]: locations of each bus stop

    """

    def __init__(
        self,
        num_bus_stops: int = 50,

        #站台位置
        # 经度范围
        min_longitude: float = 40.5, 
        max_longitude: float = 40.9,  
        # 纬度范围
        min_latitude: float = -74.1,
        max_latitude: float = -73.7,

        # 行驶速度
        min_velocity: float = 30, # km/h
        max_velocity: float = 60, # km/h

        # 初始电量
        min_init_charge: float = 30, # kWh
        max_init_charge: float = 50, # kWh

        # 电池容量
        min_battery_capacity: float = 80, # kWh
        max_battery_capacity: float = 100, # kWh

        # 行驶能耗
        min_energy_consumption: float = 0.1, # kWh/km
        max_energy_consumption: float = 0.15, # kWh/km

        MPT_charging_power: float = 100, # kW

        mu: float = 1.7,       # 截止时间正态分布均值（小时）
        sigma: float = 0.1,     # 截止时间正态分布标准差（小时）
        seed: Optional[int] = 1234,
        init_sol_type: str = "random",
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        **kwargs,
    ):
        super().__init__()

        # 经度范围
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude
        # 纬度范围
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude

        self.num_bus_stops = num_bus_stops

        self.min_velocity = min_velocity
        self.max_velocity = max_velocity

        self.min_init_charge = min_init_charge
        self.max_init_charge = max_init_charge

        self.min_battery_capacity = min_battery_capacity
        self.max_battery_capacity = max_battery_capacity

        self.min_energy_consumption = min_energy_consumption
        self.max_energy_consumption = max_energy_consumption

        self.MPT_charging_power = MPT_charging_power
        self.mu = torch.tensor(mu)
        self.sigma = torch.tensor(sigma)
        self.seed = seed
        self.distance_matrix = None

        self.init_sol_type = init_sol_type

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.longitude_sampler = kwargs["longitude_sampler"]
            self.latitude_sampler = kwargs["latitude_sampler"]
        else:
            self.longitude_sampler = get_sampler(
                "loc", loc_distribution, min_longitude, max_longitude, **kwargs
            )
            self.latitude_sampler = get_sampler(
                "loc", loc_distribution, min_latitude, max_latitude, **kwargs
            )
        
        if self.seed is not None:
            torch.manual_seed(self.seed)


    def _generate(self, batch_size: int = 1):
        """
        生成一个EVR实例。

        参数：
            batch_size (int): 批量大小。

        返回：
            dict: 包含生成实例数据的字典。
        """

        # 生成站台的经纬度
        longtitude_val = self.longitude_sampler.sample((*batch_size, self.num_bus_stops, 1))
        latitude_val = self.latitude_sampler.sample((*batch_size, self.num_bus_stops, 1))
        # 经纬度拼接
        coords = torch.cat((longtitude_val, latitude_val), dim=2)
        
        if type(batch_size) == list:
            batch_size = batch_size[0]

        # 生成站台的距离矩阵
        # 使用haversine_distance_batch函数计算距离矩阵
        distance_matrix = torch.zeros(batch_size, self.num_bus_stops, self.num_bus_stops)
        for i in range(self.num_bus_stops):
            for j in range(i+1, self.num_bus_stops):
                distance = self.haversine_distance_batch(coords[:, i, :], coords[:, j, :])
                distance_matrix[:, i, j] = distance
                distance_matrix[:, j, i] = distance
        distance_matrix.diagonal(dim1=-2, dim2=-1).fill_(float('inf'))

        # 随机充电路径的比例，通常充电路径占所有路径的0.2左右
        dym_ratio = 0.01
        min_prob = 0.2 - dym_ratio
        max_prob = 0.2 + dym_ratio
        prob = random.uniform(min_prob, max_prob)
        charging_matrix = torch.bernoulli(torch.full(
            (batch_size, self.num_bus_stops, self.num_bus_stops), prob)).int()
        #for i in range(batch_size):
        #    charging_matrix[i,:,:].fill_diagonal_(0)

        # 生成充电站矩阵
        #charging_matrix = torch.randint(
        #    low=0,
        #    high=2,
        #    size=(batch_size, self.num_bus_stops, self.num_bus_stops),
        #    dtype=torch.int64
        #)
        charging_matrix.diagonal(dim1=-2, dim2=-1).fill_(0)
        charging_matrix = torch.logical_or(charging_matrix, charging_matrix.transpose(1, 2))

        # 生成其他参数
        vehicle_speeds = torch.empty(batch_size,self.num_bus_stops,self.num_bus_stops).uniform_(self.min_velocity, self.max_velocity)
        vehicle_speeds.diagonal(dim1=-2, dim2=-1).fill_(0)
        initial_battery = torch.empty(batch_size).uniform_(self.min_init_charge, self.max_init_charge)
        battery_capacities = torch.empty(batch_size).uniform_(self.min_battery_capacity, self.max_battery_capacity)
        energy_consumption = torch.empty(batch_size).uniform_(self.min_energy_consumption, self.max_energy_consumption)
        deadline_times = torch.clamp(torch.normal(mean=self.mu, std=self.sigma, size=(batch_size,)), min=0.1)
        charging_power = torch.ones(batch_size) * self.MPT_charging_power

        return TensorDict(
            {   
                "locs" : coords,
                "distance_matrix": distance_matrix,
                "charging_matrix": charging_matrix,
                "vehicle_speeds": vehicle_speeds,
                "initial_battery": initial_battery,
                "battery_capacities": battery_capacities,
                "energy_consumption": energy_consumption,
                "deadline_times": deadline_times,
                "charging_ratio" : charging_power
            },
            batch_size=batch_size,
        )


    def haversine_distance_batch(self,locs1, locs2):
        """
        计算批次中每对点之间的距离（以公里为单位）。
        
        参数：
            locs1 (torch.Tensor): 第一个点的经纬度，形状为 (batch_size, 2)。
            locs2 (torch.Tensor): 第二个点的经纬度，形状为 (batch_size, 2)。
            
        返回：
            torch.Tensor: 每对点之间的距离（以公里为单位），形状为 (batch_size,)。
        """
        # 将经纬度从度转换为弧度
        locs1 = torch.deg2rad(locs1)
        locs2 = torch.deg2rad(locs2)
        
        # 提取经纬度
        lat1, lon1 = locs1[:, 0], locs1[:, 1]
        lat2, lon2 = locs2[:, 0], locs2[:, 1]
        
        # 计算差值
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Haversine 公式
        a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
        c = 2 * torch.asin(torch.sqrt(a))
        
        # 地球半径（公里）
        r = 6371
        
        # 计算距离
        distance = r * c
        return distance
