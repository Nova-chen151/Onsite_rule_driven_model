import matplotlib
import pickle
import numpy as np
import pandas as pd
matplotlib.use('TkAgg')  # 选择 TkAgg 作为交互框架
from matplotlib import animation
import matplotlib.pyplot as plt
from vehicles.vehicle import Vehicle
from util.state import VehicleType
from roads.road import Road
from util.state import State
from util.interactionsolver import InteractionSolver
from vehicles import D1Vehicles
from paramset.GUI_paramset import VEHs_params
from tkinter import *
import xml.dom.minidom
from roads import D1xodrRoads
import time
from util.find_connection import find_legal_connections_between_o_d
from util.find_connection import organize_legal_connections, generate_basic_od_pairs
from paramset.GUI_paramset import App

class Sim:
    def __init__(self, xodr, root, vehicles, stop_time, show_plot=False):  # 初始化模拟类，接收相关参数
        self.timestamp = 0.0            # 仿真时间戳，初始化为0
        self.simulation_step = 0.1      # 仿真步长，每次时间推进0.1秒
        self.start_time = time.time()   # 记录仿真开始的时间
        self.stop_time = stop_time      # 设置仿真停止的时间（单位：秒）

        # 如果需要绘图显示仿真过程
        if show_plot:
            self.fig, self.ax = plt.subplots()  # 创建一个新的图表和坐标轴
            plt.xlim(-30, 30)  # 设置x轴的显示范围
            plt.ylim(-30, 30)  # 设置y轴的显示范围
            plt.axis('equal')  # 保证坐标轴比例相等
        else:
            self.fig = None  # 如果不绘图，fig为None
            self.ax = None  # 如果不绘图，ax为None

        self.obj_list = {}  # 初始化一个字典，存储路网中的车辆信息
        self.all_id = len(vehicles)  # 存储所有车辆的数量
        self.obj_num = 0  # 当前仿真中的车辆数量初始化为0

        # 初始化xodr路网图
        roadD1 = D1xodrRoads.Graph()  # 创建一个xodr道路图的对象

        shp_flag = 0  # 标志，是否绘制路网形状（默认不绘制）

        # 从xodr文件读取道路信息，创建路网，并可选择性地绘制
        net = D1xodrRoads.create_road(roadD1, xodr, self.ax if show_plot else None)
        self.net = net  # 存储路网对象

        # 寻找起点和终点的连接（车辆OD）
        o_points = [link_id for link_id, link in net.link_map.items() if not link.in_link_lst]
        d_points = [link_id for link_id, link in net.link_map.items() if not link.out_link_lst]

        # 寻找合法的起点-终点连接
        legal_connections = find_legal_connections_between_o_d(net.link_map, o_points, d_points)

        # 组织合法的连接
        path_set = organize_legal_connections(legal_connections)

        # 如果没有合法的连接路径，则生成基础的OD对（起点-终点对）
        if not path_set:
            path_set = generate_basic_od_pairs(net.link_map)

        # 根据路网拓扑结构创建路径
        roadD1.create_path(path_set)

        # 初始化路径的时间间隔
        for path in roadD1.path_map.values():
            path.interval = 0

        # 约束路网，更新world链接
        self.constrain_world(roadD1)

        # 存储路网对象
        self.roadD1 = roadD1

        # 存储车辆列表
        self.vehicle_0 = vehicles

        # 为所有路径创建路径ID列表
        all_path_list = []
        for path in self.roadD1.path_map.values():
            path_id_lst = self.roadD1.find_path(path.oid, path.did)  # 获取路径的ID列表
            all_path_list.append(path_id_lst)
        self.all_path_list = all_path_list  # 存储所有路径的ID列表

        # 更新起点和终点点的连接（完善路网）
        o_points = [link_id for link_id, link in net.link_map.items()]
        d_points = [link_id for link_id, link in net.link_map.items()]

        # 再次查找合法的连接
        legal_connections = find_legal_connections_between_o_d(net.link_map, o_points, d_points)

        # 组织合法的连接
        path_set = organize_legal_connections(legal_connections)

        # 如果路径为空，生成基础OD对
        if not path_set:
            path_set = generate_basic_od_pairs(net.link_map)

        # 参数配置（例如车辆参数）
        roadD1.create_path(path_set)  # 根据路网拓扑结构创建路径
        params = VEHs_params()  # 初始化车辆的参数设置

        # 初始化路径的时间间隔
        for path in roadD1.path_map.values():
            path.interval = 0

        # 约束世界（设置可用的道路ID列表）
        self.constrain_world(roadD1)

        # 存储路网数据
        self.roadD1 = roadD1

        # 仿真设置的其他参数
        self.sim_duration = 1000    # 仿真总时长
        self.truck_rate = 0.1       # 卡车的出现率
        self.agsv_rate = 0.1        # 自动驾驶小车的出现率
        self.cnsv_rate = 0.1        # 储能车的出现率

        # 存储车辆参数
        self.veh_params = params
        self.d2_on = 0  # 可能是一个开关标志

        # 创建道路对象，并将图形转换为仿真所需的格式
        road = Road()
        road.graph2road(roadD1, shp_flag)

        road.SetOverlap()  # 设置道路重叠

        self.road = road  # 存储道路对象
        self.connector_car_dict = {}  # 存储每个连接处的车辆
        self.section_car_dict = {}  # 存储每个路段的车辆

        # 初始化连接点和路段的车辆字典
        for connector in self.road.connector_list.values():
            self.connector_car_dict[connector.id] = []
        for section in road.section_list.values():
            self.section_car_dict[section.id] = []

        # 绘制路网（可选）
        self.roadD1.draw(self.ax)  # 如果需要绘制图形，绘制路网

        # 初始化物理图和逻辑图
        self.physical_graph = {}
        self.logical_graph = {}

        # 存储轨迹数据
        self.traj_dict = {}

        # 设置仿真已运行的标志
        self.has_run = True

        # 初始化车辆数据字典和数组数据字典
        self.vehicle_data = {}
        self.array_data = {}

        # 可视化的时候需要取消注释
        # params = App(path_set, root)
        # self.sim_duration = float(params.sim_duration.get())
        # self.truck_rate = float(params.truck_rate.get())
        # self.agsv_rate = float(params.driver_agsv.get())
        # self.cnsv_rate = float(params.driver_cnsv.get())
        # self.veh_params = params.params
        # self.d2_on = int(params.d2_on.get())

    # 限制世界范围，设定可用的路网链接
    def constrain_world(self, graph):
        world_link_ids = []  # 存储所有的世界路链ID
        for path in graph.path_map.values():
            roads = path.path_id_lst
            for road in roads:
                if road not in world_link_ids:
                    world_link_ids.append(road)
        graph.world = world_link_ids  # 设置图的世界范围（可用道路ID列表）

    # 生成新车并将其加入仿真
    def BornNewCar(self):
        for vehicle in self.vehicle_0:  # 对于每辆车
            # 如果需要绘图，初始化绘制车辆的图形
            if self.fig:
                plot_vehicle, = self.ax.plot([], [], 'r')  # 在图表中绘制车辆
            else:
                plot_vehicle = None  # 如果不需要绘图，设置为None

            # 根据车辆类型决定参数（是否为AGSV自动驾驶车辆）
            if vehicle[-4] < 5:  # 判断是否为AGSV（自动驾驶小车）
                params = self.veh_params.params_car_agsv  # 设置AGSV车辆的参数
                veh_type = 'car'  # 车辆类型为‘car’
            else:  # 如果是卡车
                params = self.veh_params.params_truck_agsv  # 设置卡车的参数
                veh_type = 'truck'  # 车辆类型为‘truck’

            # 创建新的车辆对象并加入仿真
            D1Vehicles.Vehicle(self.roadD1, vehicle, veh_type, plot_vehicle, params, self.simulation_step,
                               self.all_path_list)

    def UpdateSignals(self):
        # 更新信号状态
        self.roadD1.update_signal(self.timestamp)

    def SolveInteraction(self):
        # 创建交互求解器实例
        inter_solver = InteractionSolver()
        inter_solver.logical_graph = self.logical_graph  # 设置逻辑图
        inter_solver.obj_list = self.obj_list  # 设置物体列表
        inter_solver.road = self.road  # 设置道路
        inter_solver.connector_car_dict = self.connector_car_dict  # 设置连接器与车辆字典
        inter_solver.section_car_dict = self.section_car_dict  # 设置路段与车辆字典
        inter_solver.simulation_step = self.simulation_step  # 设置仿真步长
        inter_solver.FirstAnalyze()  # 进行初步分析
        inter_solver.SolvePhysicalRelation()  # 解决物理关系

        wait_list = []  # 等待队列
        exit_flag = 0  # 退出标志
        for vehicle in self.obj_list.values():
            # 如果车辆的方向是左转或右转
            if vehicle.direction == 'left turn' or vehicle.direction == 'right turn':
                if self.logical_graph != {}:
                    '''
                    for (k, v) in self.logical_graph.items():
                        if vehicle.id == k or vehicle.id in v:
                            exit_flag = 1
                            break
                    '''
                    if exit_flag == 0:
                        # 如果没有触发exit_flag，则将该车辆加入等待队列
                        wait_list.append(vehicle)
                else:
                    # 如果没有逻辑图，直接加入等待队列
                    wait_list.append(vehicle)
            exit_flag = 0  # 重置exit_flag
        # 处理等待队列中的车辆
        while wait_list != []:
            vehicle = wait_list.pop()  # 从等待队列中取出车辆
            if len(vehicle.trajectory) > 2:  # 如果车辆轨迹长度大于2，进行处理
                inter_solver.AddObj(vehicle)  # 将车辆添加到求解器中

        # 更新对象列表和逻辑图
        self.obj_list = inter_solver.obj_list
        self.logical_graph = inter_solver.logical_graph

    def UpdateVeh(self):
        del_list = []  # 用于存储待删除的车辆id
        s_threshold = 20  # 设置车辆位置的阈值
        for vehicle in self.obj_list.values():
            connector = self.road.connector_list[vehicle.connector_id]  # 获取车辆所在的连接器
            connector_length = connector.GetLength()  # 获取连接器的长度
            vehicle_s = connector.GetSPosition(vehicle)  # 获取车辆在连接器上的位置
            vehicle.RunSim()  # 运行车辆的仿真步骤
            if vehicle_s + s_threshold > connector_length:  # 如果车辆的当前位置加上阈值超过了连接器的长度
                del_list.append(vehicle.id)  # 将该车辆加入删除列表
                self.D2veh_to_D1veh(vehicle, True)  # 将车辆从D2更新到D1
            else:
                vehicle.Draw()  # 绘制车辆
                self.D2veh_to_D1veh(vehicle, False)  # 车辆状态更新

        # 删除车辆
        for del_veh_id in del_list:
            if self.fig:
                # 如果有图形界面，则清除车辆的图形显示
                self.obj_list[del_veh_id].plot_vehicle.set_data([], [])
                self.obj_list[del_veh_id].plot_line.set_data([], [])
            self.connector_car_dict[self.obj_list[del_veh_id].connector_id].remove(del_veh_id)  # 从连接器车辆字典中移除
            del self.obj_list[del_veh_id]  # 删除车辆对象

    def thread_upd(self, vehicles):
        for vehicle in vehicles:
            vehicle.update()  # 更新车辆状态
            self.roadD1.update_veh(vehicle)  # 更新道路中的车辆
            if self.fig:
                vehicle.draw()  # 如果有图形界面，绘制车辆
            if not vehicle.on_link and self.d2_on:
                self.D1veh_to_D2veh(vehicle)  # 如果车辆不在链路上且D2启用，则将车辆从D1转到D2
                vehicle.status = 2  # 设置车辆状态为2

    def UpdateVehD1(self):
        del_list = []  # 用于存储待删除的车辆id
        vehs = []  # 存储当前车辆
        veh_group = []  # 存储车辆分组
        group_num = 200  # 每组最多200辆车
        veh_count = 0  # 车辆计数
        frame_id = self.get_current_frame_id()  # 获取当前帧的ID

        for vehicle in self.roadD1.vehicles.values():
            # 如果车辆数量超过设定的group_num，则将当前车辆分组
            if veh_count > group_num:
                veh_group.append(vehs)
                vehs = []
                veh_count = 0
            veh_count += 1
            if vehicle.status == 0:
                del_list.append(vehicle.id)  # 如果车辆状态为0，加入删除列表
            elif vehicle.status == 1:
                vehs.append(vehicle)  # 如果车辆状态为1，加入当前车辆列表
                # 存储车辆数据
                if vehicle.id not in self.vehicle_data:
                    self.vehicle_data[vehicle.id] = {
                        'positions': [],
                        'headings': [],
                        'velocities': []
                    }
                self.vehicle_data[vehicle.id]['positions'].append([vehicle.world_x, vehicle.world_y])
                self.vehicle_data[vehicle.id]['headings'].append(vehicle.heading)
                self.vehicle_data[vehicle.id]['velocities'].append(vehicle.speed)

        veh_group.append(vehs)  # 将当前车辆添加到分组中
        self.thread_upd(vehs)  # 更新当前车辆

        # 删除车辆
        for del_veh_id in del_list:
            if self.fig:
                # 如果有图形界面，清除车辆显示
                self.roadD1.vehicles[del_veh_id].plot_vehicle.set_data([], [])
            del self.roadD1.vehicles[del_veh_id]  # 删除车辆对象

    def save_data(self):
        # 获取所有存在的车辆ID（浮点形式）
        existing_vids = list(self.vehicle_data.keys())

        # 确定最小和最大ID
        min_vid = 1
        max_vid = self.all_id

        # 生成完整的连续ID列表（从min_vid到max_vid，步长1.0）
        complete_vids = []
        current_vid = min_vid
        while current_vid <= max_vid:
            complete_vids.append(current_vid)
            current_vid += 1.0

        number_agents = len(complete_vids)

        # 获取最大帧数
        max_frames = max(len(data['positions']) for data in self.vehicle_data.values())   # 可视化的时候任意取值100

        # 初始化数组，行数对应完整ID列表
        positions = np.zeros((number_agents, max_frames, 2), dtype=np.float32)
        headings = np.zeros((number_agents, max_frames), dtype=np.float32)
        velocities = np.zeros((number_agents, max_frames), dtype=np.float32)

        # 填充数据：存在的ID填入数据，缺失的保持为0
        for i, vid in enumerate(complete_vids):
            if vid in self.vehicle_data:
                agent_data = self.vehicle_data[vid]
                num_frames = len(agent_data['positions'])
                positions[i, :num_frames] = agent_data['positions']
                headings[i, :num_frames] = agent_data['headings']
                velocities[i, :num_frames] = agent_data['velocities']

        self.array_data = {
            'positions': positions,
            'headings': headings,
            'velocities': velocities
        }

        return self.array_data

    def get_current_frame_id(self):
        # 检查当前对象是否已经存在 _frame_id 属性
        if not hasattr(self, '_frame_id'):
            self._frame_id = 0  # 如果没有，则初始化 _frame_id 为 0
        # 增加 0.1 的偏移量，每次调用该函数时，帧数会递增
        self._frame_id += 0.1
        # 返回当前的帧ID
        return self._frame_id

    def D1veh_to_D2veh(self, vehicle):
        # 如果需要绘制图形，创建一个红色的车辆图形和绿色的线路图形
        if self.fig:
            plot_vehicle, = self.ax.plot([], [], 'r')  # 车辆图形
            plot_line, = self.ax.plot([], [], 'g')  # 路线图形
        else:
            plot_vehicle = None  # 如果不需要绘制图形，设置为 None
            plot_line = None

        # 创建一个新的车辆对象
        new_vehicle = Vehicle(plot_vehicle, plot_line)

        # 将原始车辆的属性赋值给新车辆
        new_vehicle.id = vehicle.id
        new_vehicle.start_lane_id = vehicle.current_lane.start_lane_id
        new_vehicle.end_lane_id = vehicle.current_lane.end_lane_id
        new_vehicle.vehicle_type = VehicleType(vehicle.type)  # 车辆类型
        new_vehicle.connector_id = vehicle.current_link.id  # 当前连接ID

        # 获取连接器并设置方向
        connector = self.road.connector_list[new_vehicle.connector_id]
        new_vehicle.direction = connector.direction

        # 计算起始车道和结束车道的分段位置
        slane_id = new_vehicle.start_lane_id
        elane_id = new_vehicle.end_lane_id

        # 记录连接器中车辆的 ID
        self.connector_car_dict[connector.id].append(new_vehicle.id)

        # 计算起始位置和结束位置
        start_divide = 1 - (slane_id - 0.5) / connector.start_section.lane_number
        start_position = connector.start_section.GetDividePos(start_divide)
        end_divide = 1 - (elane_id - 0.5) / connector.end_section.lane_number
        end_position = connector.end_section.GetDividePos(end_divide)

        # 更新新车辆的状态
        new_vehicle.state.x = start_position.x
        new_vehicle.state.y = start_position.y
        new_vehicle.state.yaw = start_position.yaw
        new_vehicle.state.speed = vehicle.speed
        new_vehicle.state.acc = vehicle.acc

        # 设置目标位置
        target = State()
        target.x = end_position.x
        target.y = end_position.y
        target.yaw = end_position.yaw
        new_vehicle.targets.append(target)

        # 将新车辆添加到对象列表中
        self.obj_list[new_vehicle.id] = new_vehicle

    def D2veh_to_D1veh(self, vehicle, flag):
        # 获取 D1 中的车辆对象
        D1vehicle = self.roadD1.vehicles[vehicle.id]
        # 获取车辆的当前连接器
        current_link = self.roadD1.get_link(vehicle.connector_id)

        # 更新 D1 车辆的世界坐标和状态
        D1vehicle.world_x = vehicle.state.x
        D1vehicle.world_y = vehicle.state.y
        D1vehicle.heading = vehicle.state.yaw
        D1vehicle.speed = vehicle.state.speed

        # 根据车辆的世界坐标转换为车道位置
        [veh_lane, veh_pos] = D1shpRoads.worldxy2_lanepos(D1vehicle.world_x, D1vehicle.world_y, current_link, flag)
        D1vehicle.current_lane = veh_lane
        D1vehicle.current_link = veh_lane.ownner
        D1vehicle.position = veh_pos

        # 如果需要绘图，则绘制该车辆
        if self.fig:
            D1vehicle.draw()

        # 如果 flag 为 True，更新车辆的路径顺序和状态
        if flag is True:
            D1vehicle.path_order += 1
            D1vehicle.status = 1

    def UpdateLines(self):
        # 如果需要绘图，检查并移除没有任何数据的线
        if self.fig:
            lines_to_remove = []
            for line in self.ax.lines:
                if len(line._xorig) == 0:  # 如果线没有原始数据
                    lines_to_remove.append(line)  # 记录要删除的线

            # 移除记录的线
            for line in lines_to_remove:
                line.remove()

    def Update(self, frame):
        # 获取当前的时间
        current_time = time.time() - self.start_time
        # 如果模拟时间超过停止时间，结束模拟
        if current_time >= self.stop_time:
            if self.fig:
                plt.close()  # 如果需要，关闭绘图
            return []  # 返回空列表表示模拟结束

        # 更新时间戳
        self.timestamp += self.simulation_step

        # 更新信号
        self.UpdateSignals()

        # 如果需要绘图，更新图形中的线条
        if self.fig:
            self.UpdateLines()

        # 如果模拟尚未运行，生成新的车辆
        if self.has_run:
            self.BornNewCar()
            self.has_run = False

        # 更新 D1 和 D2 车辆
        time_s = time.time()
        self.UpdateVehD1()
        time_e = time.time()
        self.UpdateVeh()

        # 求解车辆之间的交互
        self.SolveInteraction()

        return []

    def Run(self, show_plot=False):
        # 首先生成新车辆
        self.BornNewCar()

        # 计算最大帧数
        max_frames = self.stop_time * 1000 // 100

        if show_plot and self.fig:

            # 如果需要显示绘图，则使用动画功能更新图形
            ani = animation.FuncAnimation(
                self.fig,
                lambda frame: self.Update(frame),  # 使用更新函数
                frames=max_frames,
                interval=50,
                blit=False,
                cache_frame_data=False
            )
            plt.show()  # 显示图形
        else:
            # 如果不需要显示图形，则直接进行模拟
            for _ in range(int(max_frames)):
                self.Update(None)

        # # 保存数据
        self.save_data()

    def get_array_data(self):
        # 返回数组数据
        return self.array_data