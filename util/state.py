import numpy as np

class State:
    def __init__(self):             # 初始化函数，构造初始状态
        self.t = 0.0                # 时间，初始为0
        self.x = 0.0                # x坐标，初始为0
        self.y = 0.0                # y坐标，初始为0
        self.speed = 0.0            # 速度，初始为0
        self.yaw = 0.0              # 偏航角，初始为0
        self.acc = 0.0              # 加速度，初始为0
        self.curvature = 0.0        # 曲率，初始为0

    def SetValue(self, state1):
        self.t = state1.t           # 设置时间为state1的时间
        self.x = state1.x           # 设置x坐标为state1的x坐标
        self.y = state1.y           # 设置y坐标为state1的y坐标
        self.speed = state1.speed   # 设置速度为state1的速度
        self.yaw = state1.yaw       # 设置偏航角为state1的偏航角
        self.acc = state1.acc       # 设置加速度为state1的加速度
        self.curvature = state1.curvature  # 设置曲率为state1的曲率

    def Rotate(self, theta):
        # 定义2D旋转矩阵
        rotation_matrix = np.array([
            [np.cos(theta), np.sin(theta)],  # 第一行：cos(theta) 和 sin(theta)
            [-np.sin(theta), np.cos(theta)]  # 第二行：-sin(theta) 和 cos(theta)
        ])
        position_matrix = np.array([self.x, self.y])  # 当前x, y坐标构成的矩阵
        rotated_postion = np.dot(position_matrix, rotation_matrix)  # 矩阵乘法，得到旋转后的坐标
        self.x = rotated_postion[0]  # 更新x坐标为旋转后的值
        self.y = rotated_postion[1]  # 更新y坐标为旋转后的值
        self.yaw += theta  # 偏航角增加theta，完成旋转后的更新

class VehicleType:
    def __init__(self, veh_type):
        self.value = veh_type                   # 车辆类型
        if veh_type == 'car':                   # 如果是'car'类型
            self.length = 4.5                   # 长度为4.5米
            self.width = 2.0                    # 宽度为2.0米
            self.max_acc = 2.0                  # 最大加速度为2.0 m/s²
            self.reaction_time = 1.0            # 反应时间为1秒
            self.desired_speed = 2.0            # 期望速度为2.0 m/s
            self.comfort_dec = -2.0             # 舒适减速度为-2.0 m/s²
            self.type = 'car'                   # 类型为'car'
        elif veh_type == 'bike':                # 如果是'bike'类型
            self.length = 1.8                   # 长度为1.8米
            self.width = 0.7                    # 宽度为0.7米
            self.max_acc = 2.0                  # 最大加速度为2.0 m/s²
            self.reaction_time = 1.0            # 反应时间为1秒
            self.desired_speed = 20.0           # 期望速度为20.0 m/s
            self.comfort_dec = -2.0             # 舒适减速度为-2.0 m/s²
            self.type = 'bike'                  # 类型为'bike'
        elif veh_type == 'truck':               # 如果是'truck'类型
            self.length = 8.5                   # 长度为8.5米
            self.width = 2.8                    # 宽度为2.8米
            self.max_acc = 3.0                  # 最大加速度为3.0 m/s²
            self.reaction_time = 1.0            # 反应时间为1秒
            self.desired_speed = 15.0           # 期望速度为15.0 m/s
            self.comfort_dec = -2.0             # 舒适减速度为-2.0 m/s²
            self.type = 'truck'                 # 类型为'truck'

    def SetParam(self, param):
        self.reaction_time = param.vehicle_type_param.reaction_time     # 设置反应时间
        self.desired_speed = param.vehicle_type_param.desired_speed     # 设置期望速度
        self.comfort_dec = param.vehicle_type_param.comfort_dec         # 设置舒适减速度

class FrameManager:
    def __init__(self):
        self.count = 0          # 初始化计数器
        self.sim_step = 0.1     # 每次模拟步长，单位为秒

    def SetParam(self, param):
        self.sim_step = param.sim_step  # 从参数中获取并设置模拟步长

    def NextStep(self):
        self.count += 1  # 步数递增

    def GetTimeStamp(self):
        return self.count * self.sim_step  # 通过步数和模拟步长计算当前时间
