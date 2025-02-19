import numpy as np
import scipy.io as sio
import math

from util.state import State
from planner.parameter import VehicleTypeParameter

class PathPlanner:
    def __init__(self, current_state, target):
        self.net_param = sio.loadmat('planner/net_param.mat')   # 加载网络参数文件

        # 设置路径规划中的权重
        self.weight = [8.979991E-01, -2.149553E-02, 1.839042E-02]
        self.plan_x = [10.0, 10.0, 10.0]                        # 计划的 x 坐标（初始化）
        self.desired_acc = 1.0                                  # 期望加速度
        self.desired_speed = 15.0                               # 期望速度
        self.current_state = current_state                      # 当前状态
        self.target = target                                    # 目标状态
        self.curve_points = []                                  # 曲线点列表
        self.speed = []                                         # 速度列表
        self.trajectory = []                                    # 轨迹列表
        self.normalize_param = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]  # 标准化参数
        self.sim_step = 0.1                                     # 仿真步长

    def SetWeight(self, weight):
        ''' 设置路径规划的权重参数 '''
        self.weight = weight[0:3]  # 权重参数取前 3 个值

    def SetNormalizeParam(self, normalize_param):
        ''' 设置标准化参数 '''
        self.normalize_param = normalize_param  # 更新标准化参数

    def RotateVector(self, position_matrix, theta):
        # 旋转矩阵
        rotation_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        return np.dot(position_matrix, rotation_matrix)  # 对位置矩阵进行旋转

    def TargetConverter(self):
        ''' 转换目标状态，使当前状态为原点 '''
        current_state_result = State()  # 创建状态对象
        current_state_result.SetValue(self.current_state)  # 设置当前状态
        target_result = State()  # 创建目标状态对象
        target_result.SetValue(self.target)  # 设置目标状态

        # 将当前状态设置为原点
        current_state_result.x = 0.0
        current_state_result.y = 0.0
        current_state_result.yaw = 0.0

        # 将目标相对于当前状态进行转换
        target_result.x -= self.current_state.x
        target_result.y -= self.current_state.y
        target_result.yaw -= self.current_state.yaw

        # 将目标位置进行旋转，调整为相对于当前状态的坐标系
        target_position = np.array([target_result.x, target_result.y])
        rotated_target_position = self.RotateVector(target_position, -self.current_state.yaw)
        target_result.x = rotated_target_position[0]
        target_result.y = rotated_target_position[1]

        # 返回当前状态和目标状态的转换结果
        return current_state_result, target_result

    def GetControlPoints(self, x):
        # 获取当前状态和目标状态的转换结果
        converted_current_state, converted_target = self.TargetConverter()

        # 定义路径的五个控制点
        P0 = np.array([converted_current_state.x, converted_current_state.y])
        P1 = np.array([x[0], 0.0])  # x[0] 是控制点的横坐标
        P2 = np.array([x[1], 4.0 / 3.0 * converted_current_state.curvature * x[0] ** 2.0])  # 曲率控制
        P3 = np.array([converted_target.x - x[2] * np.cos(converted_target.yaw),
                       converted_target.y - x[2] * np.sin(converted_target.yaw)])
        P4 = np.array([converted_target.x, converted_target.y])

        # 将控制点组成一个数组
        P = np.array([P0, P1, P2, P3, P4])

        # 计算每个控制点的导数（一阶导数）
        dP0 = 4 * (P1 - P0)
        dP1 = 4 * (P2 - P1)
        dP2 = 4 * (P3 - P2)
        dP3 = 4 * (P4 - P3)
        dP = np.array([dP0, dP1, dP2, dP3])

        # 计算每个控制点的二阶导数
        ddP0 = 3 * (dP1 - dP0)
        ddP1 = 3 * (dP2 - dP1)
        ddP2 = 3 * (dP3 - dP2)
        ddP = np.array([ddP0, ddP1, ddP2])

        # 返回控制点、导数和二阶导数
        return P, dP, ddP

    def GetCurvature(self, u, dP, ddP):
        result = 0.0
        dfx = 0.0
        dfy = 0.0
        param = [[1, 3, 3, 1], [1, 2, 1]]  # 计算曲率所需的系数

        # 计算一阶导数（对 x 和 y 分别求导）
        for i in range(4):
            dfx += param[0][i] * u ** (i) * (1.0 - u) ** (3.0 - i) * dP[i][0]
            dfy += param[0][i] * u ** (i) * (1.0 - u) ** (3.0 - i) * dP[i][1]

        # 计算二阶导数
        ddfx = 0.0
        ddfy = 0.0
        for i in range(3):
            ddfx += param[1][i] * u ** (i) * (1.0 - u) ** (2.0 - i) * ddP[i][0]
            ddfy += param[1][i] * u ** (i) * (1.0 - u) ** (2.0 - i) * ddP[i][1]

        # 根据一阶和二阶导数计算曲率
        result += (dfx * ddfy - dfy * ddfx) / ((dfx ** 2 + dfy ** 2) ** 1.5)
        return result  # 返回计算的曲率

    def BezierCurveConverter(self, x):
        P0 = np.array([self.current_state.x, self.current_state.y])                  # 当前位置 (起始点)
        P1 = np.array([x[0], 0.0])                                                   # 控制点1, 这里假设x[0]表示x坐标，y=0
        P1 = self.RotateVector(P1, self.current_state.yaw) + P0                      # 旋转P1，考虑当前偏航角 (yaw)，并平移至P0
        P2 = np.array([x[1], 4.0 / 3.0 * self.current_state.curvature * x[0] ** 2])  # 控制点2，基于曲率和x[0]计算
        P2 = self.RotateVector(P2, self.current_state.yaw) + P0          # 旋转P2，考虑当前偏航角 (yaw)，并平移至P0
        P3 = np.array([self.target.x - x[2] * np.cos(self.target.yaw),   # 控制点3，基于目标位置和偏航角进行计算
                       self.target.y - x[2] * np.sin(self.target.yaw)])  # 根据目标偏航角进行调整
        P4 = np.array([self.target.x, self.target.y])  # 目标点 (P4)
        P = np.array([P0, P1, P2, P3, P4])  # 将控制点存储在P数组中
        fx = 0.0                            # 初始化x方向的贝塞尔曲线
        fy = 0.0                            # 初始化y方向的贝塞尔曲线
        u = np.linspace(0, 1, 21, True)     # 参数u，表示从0到1均匀分布的21个点
        param = [1, 4, 6, 4, 1]             # 贝塞尔曲线的标准权重系数

        # 计算贝塞尔曲线上的每个点
        for i in range(5):
            fx += param[i] * u ** (i) * (1.0 - u) ** (4.0 - i) * P[i][0]  # x方向的贝塞尔曲线
            fy += param[i] * u ** (i) * (1.0 - u) ** (4.0 - i) * P[i][1]  # y方向的贝塞尔曲线

        self.curve_points = [list(fx), list(fy)]  # 保存计算的曲线点 (fx, fy)

    def NewSpeedProfile(self, start_list, target_list):
        if abs(start_list[0] - target_list[0]) < 1e-1:  # 如果起点和终点的x差异太小，报错
            print("A is singular!!")
        x0 = start_list[0]   # 起始点的x坐标
        y0 = start_list[1]   # 起始点的y坐标
        k0 = start_list[2]   # 起始点的曲率
        x1 = target_list[0]  # 目标点的x坐标
        y1 = target_list[1]  # 目标点的y坐标
        k1 = target_list[2]  # 目标点的曲率

        # 计算一个包含曲线起始条件的矩阵A
        A = np.mat([[1, x0, x0 ** 2, x0 ** 3],
                    [1, x1, x1 ** 2, x1 ** 3],
                    [0, 1, 2 * x0, 3 * x0 ** 2],
                    [0, 1, 2 * x1, 3 * x1 ** 2]])

        # 目标点的y坐标和曲率等条件
        B = np.mat([[y0], [y1], [k0], [k1]])

        # 通过矩阵求解得到新的速度轮廓，使用矩阵求逆和乘法
        result = np.linalg.inv(A) * B
        return np.array(result)         # 返回速度轮廓

    def DesireMoveSpeed(self, v0, s):
        plan_speed = np.sqrt(2 * self.desired_acc * s + v0 ** 2)  # 通过公式计算规划的目标速度
        if plan_speed > self.desired_speed:                       # 如果目标速度大于最大限制速度
            return self.desired_speed                             # 返回最大速度
        return plan_speed                                         # 返回计算得到的速度

    def GetSpeedLimit(self):
        # 计算曲线点之间的间距（距离）
        s_axis_intervals = np.sqrt(np.diff(self.curve_points[0]) ** 2 + np.diff(self.curve_points[1]) ** 2)
        length = sum(s_axis_intervals)                                          # 曲线的总长度
        s_axis = [0.0]                                                          # 起始点的轴坐标，初始化为0
        _, dP, ddP = self.GetControlPoints(self.plan_x)                         # 获取控制点的导数信息
        curvature = [self.GetCurvature(0, dP, ddP)]                          # 获取曲线起点的曲率
        for i in range(len(s_axis_intervals)):                                  # 遍历所有间距点
            s_axis.append(s_axis[-1] + s_axis_intervals[i])                     # 计算每个位置的轴坐标
            curvature.append(self.GetCurvature(s_axis[-1] / length, dP, ddP))   # 根据位置计算曲率

        s_axis = np.array(s_axis)
        curvature = np.array(curvature)

        # 根据曲率计算速度限制
        speed_limit = np.sqrt(1.5 / (abs(curvature) + 1e-3))
        return speed_limit, s_axis  # 返回速度限制和轴坐标

    def InsertSeparateIndex(self, temp_idx, s_axis_separate_idx):
        if s_axis_separate_idx == []:               # 如果分隔索引列表为空，直接添加索引
            s_axis_separate_idx.append(temp_idx)
            return s_axis_separate_idx
        if temp_idx not in s_axis_separate_idx:     # 如果索引不在列表中
            if temp_idx > s_axis_separate_idx[-1]:  # 如果索引大于当前列表中的最大值，添加到最后
                s_axis_separate_idx.append(temp_idx)
            else:
                # 否则按大小顺序插入
                for i in range(len(s_axis_separate_idx)):
                    if temp_idx < s_axis_separate_idx[i]:
                        s_axis_separate_idx.insert(i, temp_idx)
                        break
        return s_axis_separate_idx  # 返回更新后的分隔索引列表

    def UpdateSeparateState(self, temp_idx, s_axis_separate_idx, separate_state, speed_limit, s_axis):
        if temp_idx in s_axis_separate_idx:  # 如果索引已存在
            change_range = [s_axis_separate_idx.index(temp_idx), s_axis_separate_idx.index(temp_idx) + 1]
            return s_axis_separate_idx, separate_state, change_range
        s_axis_separate_idx = self.InsertSeparateIndex(temp_idx, s_axis_separate_idx)  # 插入新的分隔索引
        current_idx_of_idx = s_axis_separate_idx.index(temp_idx)  # 获取当前索引的位置
        current_idx = s_axis_separate_idx[current_idx_of_idx]

        # 如果是第一个元素，直接返回
        if current_idx_of_idx == 0:
            change_range = [temp_idx, temp_idx + 1]
            return s_axis_separate_idx, separate_state, change_range

        # 如果是最后一个元素，更新当前状态
        if current_idx_of_idx == len(s_axis_separate_idx) - 1:
            new_target_list = [s_axis[current_idx],
                               speed_limit[current_idx],
                               (speed_limit[current_idx] - speed_limit[current_idx - 1]) / \
                               (s_axis[current_idx] - s_axis[current_idx - 1])]
            separate_state[current_idx_of_idx] = new_target_list
            change_range = [current_idx_of_idx - 1, current_idx_of_idx]
            return s_axis_separate_idx, separate_state, change_range

        # 插入新的目标状态并进行速度规划
        new_target_list = [s_axis[current_idx],
                           speed_limit[current_idx],
                           (speed_limit[current_idx + 1] - speed_limit[current_idx - 1]) / \
                           (s_axis[current_idx + 1] - s_axis[current_idx - 1])]
        separate_state.insert(current_idx_of_idx, new_target_list)
        change_range = [current_idx_of_idx]

        # 检查并更新与后续状态相关的速度
        min_index = -1
        while min_index != 1 and current_idx_of_idx < len(s_axis_separate_idx) - 1:
            current_idx = s_axis_separate_idx[current_idx_of_idx]
            behind_idx_of_idx = current_idx_of_idx + 1
            behind_idx = s_axis_separate_idx[behind_idx_of_idx]
            start_state = separate_state[current_idx_of_idx]
            behind_state = separate_state[behind_idx_of_idx]
            if behind_idx_of_idx < len(separate_state) - 2:
                behind_behind_state = separate_state[behind_idx_of_idx]
            else:
                behind_behind_state = [100, 100, 100]
            param = VehicleTypeParameter()  # 获取车辆参数
            speed_option = [np.sqrt(2 * self.desired_acc * (behind_state[0] - start_state[0]) + start_state[1] ** 2),
                            behind_state[1],
                            np.sqrt(-1.0 * 2 * param.comfort_dec * (behind_behind_state[0] - behind_state[0]) +
                                    behind_behind_state[1] ** 2),
                            self.desired_speed]
            target_speed = 100
            min_index = -1
            for k in range(len(speed_option)):  # 选择最小速度选项
                if target_speed > speed_option[k]:
                    target_speed = speed_option[k]
                    min_index = k
            separate_state[behind_idx_of_idx][1] = speed_option[min_index]
            current_idx_of_idx = behind_idx_of_idx

        change_range.append(current_idx_of_idx)  # 更新状态并返回
        if len(s_axis_separate_idx) != len(separate_state):
            stop = 1  # 结束标志
        return s_axis_separate_idx, separate_state, change_range

    def SpeedPlan(self, speed_limit, s_axis, *target_v):
        # 获取曲线的总长度
        length = s_axis[-1]
        # 设置初始索引位置，用于分段计算
        s_axis_separate_idx = [0, len(s_axis) - 1]

        # 如果当前速度大于0.2，则设置初始状态
        if abs(self.current_state.speed) > 0.2:
            start_list = [0,  # 起始点的轴距为0
                          self.current_state.speed,  # 当前的速度
                          self.current_state.acc / self.current_state.speed]  # 当前加速度除以速度
        else:
            start_list = [0, 1.0, 0.5]  # 如果当前速度较小，则设定默认的速度和加速度

        # 计算目标速度：选择起始速度的平方加上期望加速度产生的速度，再与限速和期望速度进行比较，取最小值
        target_speed = min(np.sqrt(2 * self.desired_acc * length + start_list[1] ** 2),
                           speed_limit[-1],  # 限速的最大速度
                           self.desired_speed)  # 目标速度

        # 如果传入了目标速度数组，则进一步更新目标速度
        if len(target_v) > 0:
            target_speed = min(target_speed, target_v[0])  # 限制目标速度
            # 计算加速度
            a = (target_speed ** 2 - start_list[1] ** 2) / (2 * length) + 1e-2
            # 根据加速度生成计划的速度
            plan_speed = np.sqrt(start_list[1] ** 2 + 2 * a * s_axis)
        else:
            # 如果没有目标速度，则调用 DesireMoveSpeed 生成速度计划
            plan_speed = [self.DesireMoveSpeed(start_list[1], s) for s in s_axis]

        # 计算结束时的状态
        end_list = [length,  # 终点的轴距
                    target_speed,  # 目标速度
                    self.desired_acc / target_speed if target_speed < self.desired_speed and target_speed > 0 else 0.0]  # 目标速度时的加速度

        # 更新速度计划，比较限速和计划速度，寻找适当的点进行调整
        temp_separate_idx1 = np.argmin(speed_limit - plan_speed)
        temp_separate_idx2 = np.argmin(plan_speed)
        speed_diff = speed_limit[temp_separate_idx1] - plan_speed[temp_separate_idx1]
        min_speed = plan_speed[temp_separate_idx2]

        # 如果在最后一个限速点速度差小于0，则使用更复杂的速度配置
        if temp_separate_idx1 == len(speed_limit) - 1 and speed_diff < 0.0:
            # 使用四次多项式拟合生成新的速度计划
            curve_para = self.NewSpeedProfile(start_list, end_list)
            plan_speed = curve_para[0] + \
                         curve_para[1] * s_axis + \
                         curve_para[2] * s_axis ** 2 + \
                         curve_para[3] * s_axis ** 3
            # 再次进行限速与速度计划的比较
            temp_separate_idx1 = np.argmin(speed_limit - plan_speed)
            temp_separate_idx2 = np.argmin(plan_speed)
            speed_diff = speed_limit[temp_separate_idx1] - plan_speed[temp_separate_idx1]
            min_speed = plan_speed[temp_separate_idx2]

        count_plan = 0
        separate_state = [start_list, end_list]
        current_idx = 1

        # 进行循环，更新速度计划，直到满足条件
        while 1:
            # 判断速度差是否满足停止条件
            if speed_diff > -0.2 and min_speed >= -0.01:
                break
            count_plan += 1
            # 如果速度差小于-0.2，进行分段更新
            if speed_diff < -0.2:
                s_axis_separate_idx, separate_state, change_range1 = \
                    self.UpdateSeparateState(temp_separate_idx1, s_axis_separate_idx, separate_state, speed_limit,
                                             s_axis)
            else:
                change_range1 = []

            # 如果最小速度小于0，进行另一种分段更新
            if min_speed < 0.0:
                s_axis_separate_idx, separate_state, change_range2 = \
                    self.UpdateSeparateState(temp_separate_idx2, s_axis_separate_idx, separate_state, speed_limit,
                                             s_axis)
            else:
                change_range2 = []

            change_range = [change_range1, change_range2]

            # 对每一个分段区间进行迭代更新
            for h in range(len(change_range)):
                if change_range[h] == []:
                    continue
                for i in range(change_range[h][0], change_range[h][1]):
                    before_idx = i - 1
                    before_s_idx = s_axis_separate_idx[before_idx]
                    current_idx = i
                    current_s_idx = s_axis_separate_idx[current_idx]
                    if i == len(s_axis_separate_idx) - 1:
                        behind_idx = i
                    else:
                        behind_idx = i + 1
                    behind_s_idx = s_axis_separate_idx[behind_idx]
                    if behind_idx == len(s_axis_separate_idx) - 1:
                        behind_s_idx += 1
                    before_state = separate_state[before_idx]
                    current_state = separate_state[current_idx]

                    # 使用四次多项式对速度进行更新
                    if i == change_range[0]:
                        curve_para = self.NewSpeedProfile(before_state, current_state)
                        s_axis_piece = s_axis[before_s_idx: current_s_idx]
                        temp_plan_speed = curve_para[0] + \
                                          curve_para[1] * s_axis_piece + \
                                          curve_para[2] * s_axis_piece * s_axis_piece + \
                                          curve_para[3] * s_axis_piece * s_axis_piece * s_axis_piece
                        plan_speed[before_s_idx: current_s_idx] = temp_plan_speed
                    if i == len(s_axis_separate_idx) - 1:
                        continue
                    behind_state = separate_state[current_idx + 1]
                    s_axis_piece = s_axis[current_s_idx: behind_s_idx]
                    curve_para = self.NewSpeedProfile(current_state, behind_state)
                    temp_plan_speed = curve_para[0] + \
                                      curve_para[1] * s_axis_piece + \
                                      curve_para[2] * s_axis_piece ** 2 + \
                                      curve_para[3] * s_axis_piece ** 3
                    plan_speed[current_s_idx: behind_s_idx] = temp_plan_speed

            # 更新速度差
            temp_separate_idx1 = np.argmin(speed_limit - plan_speed)
            speed_diff = speed_limit[temp_separate_idx1] - plan_speed[temp_separate_idx1]

            if temp_separate_idx1 == 0:
                break
            temp_separate_idx2 = np.argmin(plan_speed)
            min_speed = plan_speed[temp_separate_idx2]

        # 最终生成的速度计划赋值给类的速度属性
        self.speed = plan_speed

    def SetTrajectory(self):
        # 计算曲线各个点之间的间距
        s_axis_intervals = np.sqrt(np.diff(self.curve_points[0]) ** 2 + np.diff(self.curve_points[1]) ** 2)
        length = sum(s_axis_intervals)

        # 设置起始点
        s_axis = [0.0]
        _, dP, ddP = self.GetControlPoints(self.plan_x)

        # 计算曲率
        curvature = [self.GetCurvature(0, dP, ddP)]

        # 计算每个点的曲率并更新 s 轴
        for i in range(len(s_axis_intervals)):
            s_axis.append(s_axis[-1] + s_axis_intervals[i])
            curvature.append(self.GetCurvature(s_axis[i] / length, dP, ddP))

        s_axis = np.array(s_axis)
        # 计算平均速度
        average_speed = np.diff(self.speed) / 2 + self.speed[0:-1]
        dt = []

        # 计算每段的时间
        for i in range(len(average_speed)):
            if average_speed[i] < 1.0:
                acc = (self.speed[i + 1] ** 2 - self.speed[i] ** 2) / (2 * s_axis_intervals[i])
                if acc == 0.0:
                    dt.append(self.sim_step)
                else:
                    if self.speed[i] ** 2 + 2 * acc * s_axis_intervals[i] < 0:
                        dt.append(self.sim_step)
                    else:
                        dt.append((np.sqrt(self.speed[i] ** 2 + 2 * acc * s_axis_intervals[i]) - self.speed[i]) / acc)
            else:
                dt.append(s_axis_intervals[i] / average_speed[i])

        # 生成 t 轴的时间序列
        t_axis = [0.0]
        for i in range(len(dt)):
            t_axis.append(t_axis[-1] + dt[i])

        t_sample = np.linspace(t_axis[0], t_axis[-1], int(t_axis[-1] / self.sim_step), endpoint=True)

        # 插值计算轨迹
        trj_posx = np.interp(t_sample, t_axis, self.curve_points[0])
        trj_posy = np.interp(t_sample, t_axis, self.curve_points[1])
        trj_curv = np.interp(t_sample, t_axis, curvature)
        trj_speed = np.interp(t_sample, t_axis, self.speed)

        # 更新每个时刻的状态
        for i in range(len(trj_posx)):
            new_state = State()
            new_state.x = trj_posx[i]
            new_state.y = trj_posy[i]
            new_state.speed = trj_speed[i]
            new_state.curvature = trj_curv[i]

            if i == 0:
                new_state.yaw = self.current_state.yaw
                new_state.acc = self.current_state.acc
            elif i < len(trj_posx) - 1:
                posx_diff = trj_posx[i + 1] - trj_posx[i]
                posy_diff = trj_posy[i + 1] - trj_posy[i]
                pos_diff = np.sqrt(posx_diff ** 2 + posy_diff ** 2)
                if pos_diff < 1e-3:
                    new_state.yaw = self.trajectory[i - 1].yaw
                else:
                    if posy_diff > 0:
                        new_state.yaw = math.acos(posx_diff / pos_diff)
                    else:
                        new_state.yaw = -1 * math.acos(posx_diff / pos_diff)
                new_state.acc = (trj_speed[i + 1] - new_state.speed) / self.sim_step
            else:
                new_state.yaw = self.target.yaw
                new_state.acc = self.target.acc

            # 将新的状态添加到轨迹中
            self.trajectory.append(new_state)

    def StandStill(self, t):
        # 计算停止的步骤数
        stop_step = int(t / self.sim_step) + 1

        # 如果轨迹为空，初始化轨迹并将当前状态加入轨迹
        if self.trajectory == []:
            self.trajectory.append(self.current_state)

        # 循环直到车辆停止
        for i in range(stop_step):
            # 如果当前速度小于0.5m/s，将速度和加速度设置为0
            if self.trajectory[-1].speed < 5e-1:
                self.trajectory[-1].speed = 0.0
                self.trajectory[-1].acc = 0.0
                # 将当前状态添加到轨迹
                self.trajectory.append(self.trajectory[-1])
            else:
                # 获取车辆类型的参数
                vehicle_param = VehicleTypeParameter()
                previous_state = self.trajectory[-1]
                state = State()

                # 车辆加速度不能大于舒适减速度
                state.acc = min(previous_state.acc, vehicle_param.comfort_dec)

                # 保持当前航向
                state.yaw = previous_state.yaw
                # 更新速度，考虑加速度
                state.speed = previous_state.speed + state.acc * self.sim_step
                # 如果速度小于0.5，设置为0
                if state.speed < 5e-1:
                    state.speed = 0.0
                    state.acc = 0.0
                # 更新位置信息
                state.x = previous_state.x + previous_state.speed * np.cos(previous_state.yaw) * self.sim_step
                state.y = previous_state.y + previous_state.speed * np.sin(previous_state.yaw) * self.sim_step

                # 将新的状态添加到轨迹
                self.trajectory.append(state)

    def SpeedPlanWithLimitation(self, time_limit):
        # 计算曲线段的每一小段的距离
        s_axis_intervals = np.sqrt(np.diff(self.curve_points[0]) ** 2 + np.diff(self.curve_points[1]) ** 2)
        length = sum(s_axis_intervals)

        # 如果轨迹长度小于0.2，停止车辆
        if length < 0.2:
            self.StandStill(time_limit)
            return

        # 计算期望的平均速度
        average_speed = length / time_limit
        # 获取速度限制和曲线的相关信息
        speed_limit, s_axis = self.GetSpeedLimit()

        # 如果期望的平均速度小于当前速度的一半，执行减速操作
        if average_speed < self.current_state.speed / 2.0:
            target_v = 0.0
            self.SpeedPlan(speed_limit, s_axis, target_v)
            self.SetTrajectory()

            time = len(self.trajectory) * self.sim_step
            if time < time_limit:
                # 如果时间小于限制时间，执行站立等待
                self.StandStill(time_limit - time)
                return
        else:
            # 执行正常的速度规划
            self.SpeedPlan(speed_limit, s_axis)
            self.SetTrajectory()

            upper = self.trajectory[-1].speed
            target_v = upper
            lower = 0.0
            time = len(self.trajectory) * self.sim_step
            count_plan = 0

            # 不断调整目标速度直到时间接近限制时间
            while abs(time - time_limit) > 2e-1 and abs(upper - lower) > 1e-1:
                count_plan += 1
                if time < time_limit:
                    upper = target_v
                    target_v = (upper + lower) / 2.0
                else:
                    lower = target_v
                    target_v = (upper + lower) / 2.0

                self.speed = []
                self.trajectory = []
                self.SpeedPlan(speed_limit, s_axis, target_v)

                average_speed = np.diff(self.speed) / 2 + self.speed[0:-1]
                time = 0.0
                for i in range(len(average_speed)):
                    if average_speed[i] < 1.0:
                        acc = (self.speed[i + 1] ** 2 - self.speed[i] ** 2) / (2 * s_axis_intervals[i])
                        time += ((np.sqrt(self.speed[i] ** 2 + 2 * acc * s_axis_intervals[i]) - self.speed[i]) /
                                 acc)
                    else:
                        time += (s_axis_intervals[i] / average_speed[i])

            # 设置轨迹
            self.SetTrajectory()
            if abs(upper) < 2e-1:
                # 如果最终速度很小，等待剩余时间
                wait_time = time_limit - time
                self.StandStill(wait_time)

    def SetTrajectoryWithTimeSpeed(self):
        # 初始化轨迹相关信息
        trj_x = []
        trj_y = []
        curvature = []
        new_trajectory = []

        # 提取轨迹中的各个状态值
        for state in self.trajectory:
            trj_x.append(state.x)
            trj_y.append(state.y)
            curvature.append(state.curvature)

        # 计算每个小段的距离
        s_axis_intervals = np.sqrt(np.diff(trj_x) ** 2 + np.diff(trj_y) ** 2)
        length = sum(s_axis_intervals)

        s_axis = [0.0]
        for i in range(len(s_axis_intervals)):
            s_axis.append(s_axis[-1] + s_axis_intervals[i])
        s_axis = np.array(s_axis)

        # 计算每个时刻的速度
        s_sample = [0.0]
        for i in range(len(self.speed)):
            s_sample.append(self.speed[i] * self.sim_step + s_sample[-1])
            if s_sample[i] > s_axis[-1]:
                break

        average_speed = np.diff(self.speed) / 2 + self.speed[0:-1]

        # 插值计算轨迹上的位置、速度和曲率
        trj_posx = np.interp(s_sample, s_axis, trj_x)
        trj_posy = np.interp(s_sample, s_axis, trj_y)
        trj_curv = np.interp(s_sample, s_axis, curvature)
        trj_speed = self.speed[0:i + 2]

        # 根据速度重新构建轨迹
        for i in range(len(trj_speed) - 1):
            new_state = State()
            new_state.x = trj_posx[i]
            new_state.y = trj_posy[i]
            new_state.speed = trj_speed[i]
            new_state.curvature = trj_curv[i]

            # 设置初始状态
            if i == 0:
                new_state.yaw = self.current_state.yaw
                new_state.acc = self.current_state.acc
            # 更新状态
            elif i < len(trj_posx) - 1:
                posx_diff = trj_posx[i + 1] - trj_posx[i]
                posy_diff = trj_posy[i + 1] - trj_posy[i]
                pos_diff = np.sqrt(posx_diff ** 2 + posy_diff ** 2)
                if pos_diff < 1e-3:
                    new_state.yaw = new_trajectory[i - 1].yaw
                else:
                    if posy_diff > 0:
                        new_state.yaw = math.acos(posx_diff / pos_diff)
                    else:
                        new_state.yaw = -1 * math.acos(posx_diff / pos_diff)
                # 更新加速度
                new_state.acc = (trj_speed[i + 1] - new_state.speed) / self.sim_step
            else:
                new_state.yaw = self.target.yaw
                new_state.acc = self.target.acc

            # 添加新的状态到轨迹
            new_trajectory.append(new_state)

        # 更新轨迹
        self.trajectory = new_trajectory

    def GetVectorAngle(self, x1, y1, x2, y2):
        # 计算两点之间的距离
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if abs(distance) < 1e-2:
            return None

        # 计算向量的夹角
        if y2 - y1 > 0:
            return math.acos((x2 - x1) / distance)
        else:
            return -math.acos((x2 - x1) / distance)

    def StateNormalize(self):
        length = 10
        # 转换目标状态
        current_state_result, target_result = self.TargetConverter()

        # 计算当前状态和目标状态之间的距离
        distance = np.sqrt((current_state_result.x - target_result.x) ** 2 +
                           (current_state_result.y - target_result.y) ** 2)

        # 计算起始点曲率
        start_curvature = current_state_result.curvature / (length / distance)

        # 获取终点夹角
        end_angle = self.GetVectorAngle(current_state_result.x,
                                        current_state_result.y,
                                        target_result.x,
                                        target_result.y)

        # 如果太接近目标，不能规划
        if end_angle is None:
            print("too close to plan\n")
            return None

        end_yaw = target_result.yaw
        return np.array([start_curvature, end_angle, end_yaw]), length / distance

    def PlanCurveByNN(self):
        # 定义tansig激活函数，用于神经网络的激活层
        def tansig(x):
            return 2 / (1 + np.exp(-2 * x)) - 1  # 双曲正切函数，输出范围在[-1, 1]

        # 获取归一化后的输入状态 x1 和 k
        x1, k = self.StateNormalize()
        # 对x1进行转置处理，确保输入的形状适合神经网络
        x1 = np.transpose(np.matrix(x1))

        # 获取神经网络的参数
        net_param = self.net_param

        # 获取输入层的偏移量、增益和最小值
        input_xoffset = net_param['input_xoffset']
        input_gain = net_param['input_gain']
        input_ymin = net_param['input_ymin']

        # 对输入数据进行归一化处理
        xp1 = np.multiply((x1 - input_xoffset), input_gain) + input_ymin

        # 神经网络第一层的计算
        IW = net_param['IW'][0]  # 输入层到第一层的权重
        LW = net_param['LW']  # 后续层之间的权重
        b = net_param['b']  # 偏置项

        # 计算第一层输出，使用tansig激活函数
        a = tansig(b[0][0] + np.dot(IW[0], xp1))

        # 逐层计算神经网络的输出
        for i in range(len(LW)):
            if i < len(LW) - 2:
                # 对每一层应用激活函数
                a = tansig(b[i + 1][0] + np.dot(LW[i + 1, i], a))
            if i == len(LW) - 2:
                # 对倒数第二层应用线性激活函数，获取最终的网络输出
                y1 = b[i + 1][0] + np.dot(LW[i + 1, i], a)

        # 输出层的反归一化
        output_xoffset = net_param['output_xoffset']
        output_gain = net_param['output_gain']
        output_ymin = net_param['output_ymin']

        # 通过输出层的归一化参数，反向转换网络输出
        output = (y1 - output_ymin) / output_gain + output_xoffset

        # 将输出转置并转换为一维数组
        output = np.array(np.transpose(output))

        # 将输出结果除以k后存储在self.plan_x中
        self.plan_x = list(output[0] / k)

        # 调用Bezier曲线转换函数，将规划好的点转化为Bezier曲线
        self.BezierCurveConverter(self.plan_x)


