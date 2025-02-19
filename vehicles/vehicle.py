import numpy as np

from planner.planner import PathPlanner
from util.state import State
from util.state import VehicleType
from util.state import FrameManager


class Vehicle:
    # 初始化函数，接收两个参数 plot_vehicle 和 plot_line，初始化一些车辆的属性
    def __init__(self, plot_vehicle, plot_line):
        self.scenario = None                    # 场景
        self.vehicles = None                    # 车辆集合
        self.signals = None                     # 信号集合
        self.frame_manager = FrameManager()     # 帧管理器实例化
        self.id = 0                             # 车辆ID
        self.isdead = False                     # 车辆是否死亡
        self.vehicle_type = VehicleType('car')  # 车辆类型，默认为 'car'
        self.state = State()                    # 当前车辆的状态
        self.targets = []                       # 目标点列表
        self.time_limit_to_targets = []         # 到目标的时间限制
        self.trajectory = []                    # 轨迹列表
        self.connector_id = 0                   # 连接器ID
        self.section_id = 0                     # 区段ID
        self.start_lane_id = 0                  # 起始车道ID
        self.direction = 'straight'             # 方向，默认为 'straight'（直行）
        self.wait_state = False                 # 等待状态
        self.plot_vehicle = plot_vehicle        # 车辆绘制对象
        self.plot_line = plot_line              # 轨迹线绘制对象

    # 获取车辆的矩形包围盒
    def GetRect(self, *arg):
        if len(arg) > 0:
            state = arg[0]  # 如果传入参数，则使用参数作为状态
        else:
            state = self.state  # 否则使用当前状态

        # 获取车辆的 x, y 坐标、宽度、高度和朝向（yaw）
        x = state.x
        y = state.y
        width = self.vehicle_type.width  # 车辆宽度
        height = self.vehicle_type.length  # 车辆长度
        angle = state.yaw  # 车辆的朝向（以弧度表示）

        # 车辆的矩形顶点坐标
        rect = np.array([[0, width / 2.0],
                         [0, -1 * width / 2.0],
                         [-1 * height, -1 * width / 2.0],
                         [-1 * height, width / 2.0],
                         [0, width / 2.0]])

        # 旋转矩阵，表示车辆的朝向
        R = np.array([[np.cos(angle), np.sin(angle)],
                      [-np.sin(angle), np.cos(angle)]])

        # 位移向量，表示车辆的位置
        offset = np.array([x, y])

        # 变换后的矩形位置
        transformed_rect = np.dot(rect, R) + offset
        return transformed_rect

    # 获取车辆前端的轨迹
    def GetFrontEndTrj(self):
        left_trj = [[], []]     # 左侧轨迹
        right_trj = [[], []]    # 右侧轨迹
        trajectory_idx = []     # 轨迹索引
        if self.trajectory == []:  # 如果轨迹为空，返回 None
            print("current vehicle has no trj")
            return None

        sample_interval = 1.5  # 采样间隔
        add_up_dis = 0.0  # 累积距离
        for i in range(len(self.trajectory)):
            state = self.trajectory[i]  # 当前轨迹点的状态
            if i >= 1:  # 如果不是第一个轨迹点
                previous_state = self.trajectory[i - 1]  # 上一个轨迹点
                dis = np.sqrt((previous_state.x - state.x) ** 2 + (previous_state.y - state.y) ** 2)  # 计算两点之间的距离
                add_up_dis += dis  # 累加距离
                if add_up_dis > sample_interval:  # 如果累积的距离大于采样间隔
                    trajectory_idx.append(i)  # 添加索引
                    add_up_dis = 0.0  # 重置累积距离
                    rect = self.GetRect(state)  # 获取当前状态的矩形
                    left_trj[0].append(rect[0][0])  # 更新左侧轨迹
                    left_trj[1].append(rect[0][1])
                    right_trj[0].append(rect[1][0])  # 更新右侧轨迹
                    right_trj[1].append(rect[1][1])
            else:
                trajectory_idx.append(i)  # 第一个点直接添加索引
                rect = self.GetRect(state)  # 获取第一个轨迹点的矩形
                left_trj[0].append(rect[0][0])  # 更新左侧轨迹
                left_trj[1].append(rect[0][1])
                right_trj[0].append(rect[1][0])  # 更新右侧轨迹
                right_trj[1].append(rect[1][1])
        return left_trj, right_trj, trajectory_idx

    # 获取车辆后端的轨迹
    def GetRearEndTrj(self):
        left_trj = [[], []]     # 左侧轨迹
        right_trj = [[], []]    # 右侧轨迹
        trajectory_idx = []     # 轨迹索引
        sample_interval = 1.5   # 采样间隔
        add_up_dis = 0.0        # 累积距离
        if self.trajectory == []:  # 如果轨迹为空，返回 None
            print("current vehicle has no trj")
            return None
        for i in range(len(self.trajectory)):
            state = self.trajectory[i]  # 当前轨迹点的状态
            if i >= 1:  # 如果不是第一个轨迹点
                previous_state = self.trajectory[i - 1]  # 上一个轨迹点
                dis = np.sqrt((previous_state.x - state.x) ** 2 + (previous_state.y - state.y) ** 2)  # 计算两点之间的距离
                add_up_dis += dis  # 累加距离
                if add_up_dis > sample_interval:  # 如果累积的距离大于采样间隔
                    trajectory_idx.append(i)  # 添加索引
                    add_up_dis = 0.0  # 重置累积距离
                    rect = self.GetRect(state)  # 获取当前状态的矩形
                    left_trj[0].append(rect[3][0])  # 更新左侧轨迹
                    left_trj[1].append(rect[3][1])
                    right_trj[0].append(rect[2][0])  # 更新右侧轨迹
                    right_trj[1].append(rect[2][1])
            else:
                trajectory_idx.append(i)  # 第一个点直接添加索引
                rect = self.GetRect(state)  # 获取第一个轨迹点的矩形
                left_trj[0].append(rect[3][0])  # 更新左侧轨迹
                left_trj[1].append(rect[3][1])
                right_trj[0].append(rect[2][0])  # 更新右侧轨迹
                right_trj[1].append(rect[2][1])
        return left_trj, right_trj, trajectory_idx

    # 轨迹规划，接收一个目标点，规划车辆行驶轨迹
    def TrajectoryPlan(self, target):
        pathplanner = PathPlanner(self.state, target)  # 路径规划器实例化
        pathplanner.PlanCurveByNN()  # 使用神经网络规划曲线
        speed_limit, s_axis = pathplanner.GetSpeedLimit()  # 获取速度限制和轴向信息
        pathplanner.SpeedPlan(speed_limit, s_axis)  # 进行速度规划
        pathplanner.SetTrajectory()  # 设置轨迹
        self.trajectory = pathplanner.trajectory  # 将规划好的轨迹赋值给车辆

    # 输出车辆的状态数据
    def OutPutData(self):
        result = []  # 存储输出结果
        result.append(self.id)          # 车辆ID
        result.append(self.state.x)     # 车辆x坐标
        result.append(self.state.y)     # 车辆y坐标
        result.append(self.state.yaw)   # 车辆的朝向
        return result

    # 绘制车辆的形状和轨迹
    def Draw(self):
        vehicle_shape = self.GetRect()  # 获取车辆的矩形形状
        self.plot_vehicle.set_data(vehicle_shape[:, 0], vehicle_shape[:, 1])  # 绘制车辆形状
        line_x = [state.x for state in self.trajectory]  # 获取轨迹的x坐标
        line_y = [state.y for state in self.trajectory]  # 获取轨迹的y坐标
        self.plot_line.set_data(line_x, line_y)  # 绘制轨迹线
        self.plot_line.set_marker('o')  # 设置轨迹点的标记样式
        self.plot_line.set_ms(2)  # 设置标记点的大小
        self.plot_line.set_markevery(10)  # 每10个点标记一个

    # 运行仿真，更新车辆状态并返回数据
    def RunSim(self):
        if self.trajectory == []:  # 如果轨迹为空
            self.TrajectoryPlan(self.targets[-1])  # 规划新的轨迹
        if len(self.trajectory) > 1:  # 如果轨迹中有多个点
            self.state = self.trajectory[1]  # 更新车辆状态为下一个轨迹点
            self.trajectory.pop(0)  # 删除当前轨迹点
            if len(self.time_limit_to_targets) != 0:  # 如果目标的时间限制不为空
                del_list = []  # 存储需要删除的目标索引
                for i in range(len(self.time_limit_to_targets)):
                    self.time_limit_to_targets[i] -= self.frame_manager.sim_step  # 减去每帧时间
                    if self.time_limit_to_targets[i] <= 0.0:  # 如果时间限制已过
                        del_list.append(i)  # 将该目标加入删除列表
                for index in del_list:
                    self.time_limit_to_targets.pop(index)  # 删除时间限制
                    self.targets.pop(index)  # 删除目标
        else:
            self.isdead = True  # 如果轨迹只剩下一个点，表示车辆死亡
        return self.OutPutData()  # 返回当前车辆的状态数据