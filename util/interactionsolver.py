import numpy as np
import math

from util.state import VehicleType
from util.state import State
from planner.parameter import IDMParameter
from planner.planner import PathPlanner


class InteractionSolver:
    def __init__(self):
        self.obj_list = {}              # 存储对象的字典
        self.road = None                # 路径对象
        self.connector_car_dict = None  # 连接器-车辆字典
        self.section_car_dict = None    # 路段-车辆字典
        self.current_obj = None         # 当前对象
        self.physical_graph = {}        # 物理图，用于描述车辆间的空间关系
        self.logical_graph = {}         # 逻辑图，用于描述交通规则关系
        self.group = {}                 # 分组字典
        self.simulation_step = 0.1      # 仿真步长
        self.path_planner = PathPlanner(State(), State())  # 路径规划器，假设它接受初始状态和目标状态作为输入

    def GetConnectorCrossPoint(self, connector_list, direction):
        # 获取连接器交叉点，根据给定的连接器列表和方向
        if len(connector_list) < 2:
            return 0.0, 0.0  # 如果连接器数量小于2，则返回默认值
        conn_list = {'left turn': [], 'right turn': [], 'straight': []}  # 用于存储不同类型的连接器
        for conn_id in connector_list:
            conn = self.road.connector_list[conn_id]  # 获取连接器
            conn_list[conn.direction] = conn  # 根据连接器方向将其存入字典
        if direction == 'left':
            if conn_list['left turn'] == []:
                return 0.0, 0.0  # 如果没有左转连接器，返回默认值
        if direction == 'right':
            if conn_list['right turn'] == []:
                return 0.0, 0.0  # 如果没有右转连接器，返回默认值
        if direction == 'left':
            # 获取左转和直行的边界，并计算它们的交点
            left_boundary = conn_list['left turn'].boundaries[1]
            straight_boundary = conn_list['straight'].boundaries[0]
            cross_point = self.GetCrossPoint(left_boundary, straight_boundary)  # 获取交点
            if cross_point == []:
                return 0.0, 0.0  # 如果没有交点，返回默认值
            left_s = self.GetSPosition(left_boundary, cross_point[0])  # 获取左转的s位置
            straight_s = self.GetSPosition(straight_boundary, cross_point[0])  # 获取直行的s位置
            return left_s, straight_s  # 返回左转和直行的交点位置
        if direction == 'right':
            # 获取右转和直行的边界，并计算它们的交点
            right_boundary = conn_list['right turn'].boundaries[0]
            straight_boundary = conn_list['straight'].boundaries[1]
            cross_point = self.GetCrossPoint(right_boundary, straight_boundary)  # 获取交点
            if cross_point == []:
                return 0.0, 0.0  # 如果没有交点，返回默认值
            right_s = self.GetSPosition(right_boundary, cross_point[0])  # 获取右转的s位置
            straight_s = self.GetSPosition(straight_boundary, cross_point[0])  # 获取直行的s位置
            return right_s, straight_s  # 返回右转和直行的交点位置

    def FindCommonVehicles(self, lane_veh, s, s_threshold):
        # 查找当前车道上，位置s小于阈值s_threshold的车辆
        if lane_veh == []:
            return [], [], [], []   # 如果车道上没有车辆，返回空列表
        common_veh_list = []        # 存储公共车辆列表
        common_veh_s = []           # 存储公共车辆的位置列表
        non_common_veh_list = []    # 存储非公共车辆列表
        non_common_veh_s = []       # 存储非公共车辆的位置列表
        for i, veh in enumerate(lane_veh):
            if s[i] < s_threshold:  # 如果车辆的位置小于阈值
                common_veh_list.append(veh)  # 将该车辆加入公共车辆列表
                common_veh_s.append(s[i])  # 将该车辆的位置加入公共车辆位置列表
            else:
                non_common_veh_list.append(veh)  # 否则加入非公共车辆列表
                non_common_veh_s.append(s[i])  # 将该车辆的位置加入非公共车辆位置列表
        return common_veh_list, common_veh_s, non_common_veh_list, non_common_veh_s  # 返回结果

    def SetPhysicalRelation(self, veh_list, s):
        # 设置车辆之间的物理关系（前后关系）
        if veh_list == []:
            return None  # 如果车辆列表为空，返回None
        sort_list = sorted(enumerate(s), key=lambda x: x[1], reverse=True)  # 根据车辆的位置排序
        sort_veh_index = [x[0] for x in sort_list]  # 获取车辆排序后的索引
        for i in range(len(sort_veh_index)):
            if i > 0:
                front_idx = sort_veh_index[i - 1]  # 获取前车的索引
                behind_idx = sort_veh_index[i]  # 获取后车的索引
                self.physical_graph[veh_list[front_idx]] = []  # 在物理图中为前车创建条目
                self.physical_graph[veh_list[front_idx]].append(veh_list[behind_idx])  # 在前车的条目中加入后车
            else:
                self.physical_graph[veh_list[sort_veh_index[i]]] = []  # 第一个车辆没有前车，只有空列表
        head_tail_vehs = [[veh_list[sort_veh_index[0]], s[sort_veh_index[0]]],
                          [veh_list[sort_veh_index[-1]], s[sort_veh_index[-1]]]]  # 获取前车和后车
        return head_tail_vehs  # 返回前后车和它们的位置

    def SetBikeGroup(self, veh_list, s):
        # 获取自行车的长度和汽车的宽度
        bike_len = VehicleType('bike').length
        car_width = VehicleType('car').width

        # 如果车辆列表为空，返回 None
        if veh_list == []:
            return None

        # 根据 s 进行排序，获取排序后的车辆索引和排序值
        sort_list = sorted(enumerate(s), key=lambda x: x[1], reverse=True)
        sort_veh_index = [x[0] for x in sort_list]
        sort_s = [x[1] for x in sort_list]

        # 初始化组编号
        group_num = 0
        self.group[group_num] = [veh_list[sort_veh_index[0]]]

        # 遍历所有的车辆
        for i in range(len(sort_s)):
            if i > 0:
                # 计算相邻车辆的 s 值差距
                s_diff = sort_s[i] - sort_s[i - 1]
                # 如果差距大于自行车长度加汽车宽度，说明需要分组
                if s_diff > bike_len + car_width:
                    group_num += 1
                    self.group[group_num] = []
            # 将车辆加入当前组
            self.group[group_num].append(veh_list[sort_veh_index[i]])

    def FirstAnalyze(self):
        # 遍历每个车道段的车辆列表
        for section_id, veh_list in self.section_car_dict.items():
            if veh_list == []:
                continue

            section = self.road.section_list[section_id]
            connector_id_list = section.connector_id_list

            # 遍历每个车道，分析车辆的物理关系
            for i in range(1, section.lane_number + 1):
                start_lane_id = i
                current_lane_veh = {}
                s = {}

                # 遍历每个连接器，处理每个连接器上的车辆
                for connector_id in connector_id_list:
                    current_lane_veh[connector_id] = []
                    s[connector_id] = []
                    connector = self.road.connector_list[connector_id]
                    veh_list_in_conn = self.connector_car_dict[connector_id]

                    if veh_list_in_conn == []:
                        continue

                    # 根据车辆的起始车道，分类到相应的连接器上
                    for veh_id in veh_list_in_conn:
                        vehicle = self.obj_list[veh_id]
                        if vehicle.start_lane_id == start_lane_id:
                            current_lane_veh[connector_id].append(veh_id)
                            s[connector_id].append(connector.GetSPosition(vehicle))

                # 获取左转、直行、右转的交点
                left_s, straight_s = self.GetConnectorCrossPoint(list(current_lane_veh.keys()), 'left')
                right_s, straight_s = self.GetConnectorCrossPoint(list(current_lane_veh.keys()), 'right')

                # 遍历连接器，分析不同方向的车辆
                for connector_id in current_lane_veh.keys():
                    if self.road.connector_list[connector_id].direction == 'left turn':
                        # 查找左转方向的公共车辆和非公共车辆
                        common_left_veh_list, common_left_veh_s, non_common_left_veh_list, non_common_left_veh_s = \
                            self.FindCommonVehicles(current_lane_veh[connector_id], s[connector_id], left_s)
                    if self.road.connector_list[connector_id].direction == 'straight':
                        # 查找直行方向的公共车辆和非公共车辆
                        common_straight_veh_list, common_straight_veh_s, non_common_straight_veh_list, non_common_straight_veh_s = \
                            self.FindCommonVehicles(current_lane_veh[connector_id], s[connector_id], straight_s)
                    if self.road.connector_list[connector_id].direction == 'right turn':
                        # 查找右转方向的公共车辆和非公共车辆
                        common_right_veh_list, common_right_veh_s, non_common_right_veh_list, non_common_right_veh_s = \
                            self.FindCommonVehicles(current_lane_veh[connector_id], s[connector_id], right_s)

                # 汇总所有方向上的公共车辆
                common_veh_list = []
                common_veh_s = []

                # 将左转、直行、右转方向的公共车辆合并
                if 'common_left_veh_list' in locals():
                    common_veh_list.extend(common_left_veh_list)
                    common_left_veh_list = []
                if 'common_straight_veh_list' in locals():
                    common_veh_list.extend(common_straight_veh_list)
                    common_straight_veh_list = []
                if 'common_right_veh_list' in locals():
                    common_veh_list.extend(common_right_veh_list)
                    common_right_veh_list = []

                # 汇总公共车辆的 s 值
                if 'common_left_veh_s' in locals():
                    common_veh_s.extend(common_left_veh_s)
                    common_left_veh_s = []
                if 'common_straight_veh_s' in locals():
                    common_veh_s.extend(common_straight_veh_s)
                    common_straight_veh_s = []
                if 'common_right_veh_s' in locals():
                    common_veh_s.extend(common_right_veh_s)
                    common_right_veh_s = []

                # 如果连接器是汽车类型，处理非公共车辆并建立物理关系
                if connector.car_type == 'car':
                    non_common_veh_max_s = 0
                    non_common_tail_veh = int(0)

                    # 处理公共车辆
                    if 'common_veh_list' in locals():
                        common_head_tail = self.SetPhysicalRelation(common_veh_list, common_veh_s)

                    # 处理左转、直行、右转方向的非公共车辆
                    if 'non_common_left_veh_list' in locals():
                        left_head_tail = self.SetPhysicalRelation(non_common_left_veh_list, non_common_left_veh_s)
                        if left_head_tail is not None:
                            if left_head_tail[1][1] > non_common_veh_max_s:
                                non_common_veh_max_s = left_head_tail[1][1]
                                non_common_tail_veh = left_head_tail[1][0]

                    if 'non_common_straight_veh_list' in locals():
                        straight_head_tail = self.SetPhysicalRelation(non_common_straight_veh_list,
                                                                      non_common_straight_veh_s)
                        if straight_head_tail is not None:
                            if straight_head_tail[1][1] > non_common_veh_max_s:
                                non_common_veh_max_s = straight_head_tail[1][1]
                                non_common_tail_veh = straight_head_tail[1][0]

                    if 'non_common_right_veh_list' in locals():
                        right_head_tail = self.SetPhysicalRelation(non_common_right_veh_list, non_common_right_veh_s)
                        if right_head_tail is not None:
                            if right_head_tail[1][1] > non_common_veh_max_s:
                                non_common_veh_max_s = right_head_tail[1][1]
                                non_common_tail_veh = right_head_tail[1][0]

                    # 如果有公共车辆头尾和非公共车辆尾，建立物理关系
                    if common_head_tail is not None and non_common_tail_veh != 0:
                        self.physical_graph[non_common_tail_veh] = [common_head_tail[0][0]]

                # 如果连接器是自行车类型，直接处理物理关系
                elif connector.car_type == 'bike':
                    if 'common_veh_list' in locals():
                        self.SetPhysicalRelation(common_veh_list, common_veh_s)
                    if 'non_common_left_veh_list' in locals():
                        self.SetPhysicalRelation(non_common_left_veh_list, non_common_left_veh_s)
                    if 'non_common_straight_veh_list' in locals():
                        self.SetPhysicalRelation(non_common_straight_veh_list, non_common_straight_veh_s)
                    if 'non_common_right_veh_list' in locals():
                        self.SetPhysicalRelation(non_common_right_veh_list, non_common_right_veh_s)

                    # 处理自行车组
                    self.SetBikeGroup(non_common_straight_veh_list, non_common_straight_veh_s)

    # 获取图中某个节点的头节点
    def GetHeadNode(self, node):
        head = node  # 初始化头节点为当前节点
        not_in_graph = False  # 用于标记是否找到图中的头节点
        while not_in_graph == False:
            not_in_graph = True  # 默认假设当前节点是头节点
            # 遍历物理图中的每个边
            for k, v in self.physical_graph.items():
                # 如果当前节点在某个边的连接中，则说明当前节点不是头节点
                if head in v:
                    head = k  # 将当前节点的前驱节点设为新的头节点
                    not_in_graph = False  # 重新标记为未找到头节点
                    break
        return head  # 返回图中找到的头节点

    # 求解物理关系
    def SolvePhysicalRelation(self):
        non_visited_nodes = list(self.physical_graph.keys())  # 获取所有未访问的节点
        while non_visited_nodes != []:
            node = non_visited_nodes[0]  # 从未访问节点列表中取第一个节点
            head = self.GetHeadNode(node)  # 获取该节点的头节点
            front_id = head  # 设置头节点为当前节点
            non_visited_nodes.remove(front_id)  # 从未访问节点列表中移除头节点
            behind_id = self.physical_graph[front_id]  # 获取当前节点的后继节点
            if behind_id == []:  # 如果当前节点没有后继节点
                continue  # 跳过本次循环
            else:
                behind_id = behind_id[0]  # 获取后继节点的第一个元素（假设只有一个后继）

            while behind_id is not None:
                affected_logically = 0  # 标记是否受到逻辑影响
                # 检查当前后继节点是否在逻辑图中有影响
                for values in self.logical_graph.values():
                    if behind_id in values:
                        affected_logically = 1  # 受逻辑影响
                        self.physical_graph[front_id].remove(behind_id)  # 从物理图中删除当前边
                        break
                if affected_logically == 0:  # 如果没有逻辑影响
                    # 获取前后车辆的对象
                    front_veh = self.obj_list[front_id]
                    behind_veh = self.obj_list[behind_id]
                    # 更新后续车辆的轨迹
                    self.obj_list[behind_id].trajectory = self.CarFollow(behind_veh, front_veh)
                    front_id = behind_id  # 设置当前车辆为后继车辆
                    if front_id in self.physical_graph:  # 如果前车辆在物理图中
                        non_visited_nodes.remove(front_id)  # 从未访问节点列表中移除该节点
                    if front_id in self.physical_graph.keys():  # 检查是否仍有后继节点
                        behind_id = self.physical_graph[front_id]
                        if behind_id == []:
                            behind_id = None  # 如果没有后继节点，置为None
                        else:
                            behind_id = behind_id[0]  # 获取新的后继节点
                    else:
                        behind_id = None  # 如果没有后继节点，置为None
                else:
                    behind_id = None  # 如果受到了逻辑影响，停止遍历

    # 计算两个向量之间的角度
    def GetVectorAngle(self, x1, y1, x2, y2):
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # 计算两个点之间的距离
        if abs(distance) < 1e-2:  # 如果距离很小，返回None
            return None
        if y2 - y1 > 0:  # 如果目标点的y坐标大于源点的y坐标
            return math.acos((x2 - x1) / distance)  # 计算并返回角度
        else:
            return -math.acos((x2 - x1) / distance)  # 如果目标点的y坐标小于源点，返回负的角度

    # 旋转轨迹
    def RotateTrj(self, theta, trj):
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # 构建旋转矩阵
        return np.dot(rotation_matrix, trj)  # 将轨迹与旋转矩阵相乘，得到旋转后的轨迹

    # 将轨迹按不同段分开
    def SeparateTrj(self, trj):
        trj = np.array(trj)     # 将轨迹转为numpy数组
        x = trj[0]              # 取出x坐标
        separate_trj = []       # 存储分段后的轨迹
        diff_x = np.diff(x)     # 计算x坐标的差值
        separate_idx_start = 0  # 分段的起始索引
        separate_idx_end = 0    # 分段的结束索引
        for i in range(len(diff_x) - 1):
            # 如果相邻两个差值符号不同，则认为轨迹发生了变化，进行分段
            if diff_x[i] * diff_x[i + 1] < 0:
                separate_idx_end = i + 2  # 设置分段结束索引
                separate_trj.append(trj[:, separate_idx_start: separate_idx_end])  # 保存当前分段轨迹
                separate_idx_start = separate_idx_end - 1  # 更新分段起始索引
        separate_trj.append(trj[:, (separate_idx_start)::])  # 保存最后一段轨迹
        return separate_trj  # 返回分段后的轨迹

    # 将两段轨迹对齐
    def UnitizeTrj(self, trj1, trj2):
        x1 = trj1[0]
        y1 = trj1[1]
        x2 = trj2[0]
        y2 = trj2[1]
        min_sample_interval = 0.01  # 设置最小采样间隔为0.01米

        # 获取去重后的x坐标和对应的y坐标
        unique_x1, idx = np.unique(x1, return_index=True)
        unique_y1 = y1[idx]
        unique_x2, idx = np.unique(x2, return_index=True)
        unique_y2 = y2[idx]

        # 如果任一段轨迹的有效点少于2个，则无法对齐
        if min([len(unique_x1), len(unique_x2)]) < 2:
            return None
        # 如果两段轨迹的x坐标区间不重叠，无法对齐
        if (min(unique_x1) > max(unique_x2)) or (min(unique_x2) > max(unique_x1)):
            return None

        # 计算两段轨迹x坐标的重叠区间
        x_min = max([min(unique_x1), min(unique_x2)])
        x_max = min([max(unique_x1), max(unique_x2)])

        # 如果重叠区间过小，无法进行有效对齐
        if np.abs(x_min - x_max) < min_sample_interval:
            return None

        # 根据重叠区间计算新的采样点
        steps = np.ceil((x_max - x_min) / min_sample_interval)
        x_vals = np.linspace(x_min, x_max, num=int(steps), endpoint=True)
        # 插值计算新的y值
        y1_vals = np.interp(x_vals, unique_x1, unique_y1)
        y2_vals = np.interp(x_vals, unique_x2, unique_y2)
        return x_vals, y1_vals, y2_vals  # 返回对齐后的x、y1、y2坐标

    def CrossOnePiece(self, trj1, trj2):
        y_val = self.UnitizeTrj(trj1, trj2)  # 将轨迹单位化处理
        if y_val == None:
            return None
        x_vals = y_val[0]           # 获取x坐标值
        y1_vals = y_val[1]          # 获取轨迹1的y坐标值
        y2_vals = y_val[2]          # 获取轨迹2的y坐标值
        dy = y1_vals - y2_vals      # 计算y值的差异

        # 如果dy的两个端点符号相同，则说明轨迹没有交点
        if dy[0] * dy[-1] > 0:
            return None
        else:
            # 找到dy值绝对值最小的索引位置，即交点位置
            cross_idx = np.argmin(abs(dy))
            return x_vals[cross_idx], y1_vals[cross_idx]  # 返回交点的x和y坐标
        return None

    def GetCrossPoint(self, trj1, trj2):
        trj1 = np.array(trj1)  # 将轨迹1转化为numpy数组
        trj2 = np.array(trj2)  # 将轨迹2转化为numpy数组
        move_vector = np.array([[-trj1[0, 0]], [-trj1[1, 0]]])  # 计算移动向量
        trj1 = trj1 + np.tile(move_vector, trj1.shape[1])  # 移动轨迹1
        trj2 = trj2 + np.tile(move_vector, trj2.shape[1])  # 移动轨迹2

        # 计算轨迹1的旋转角度
        rotate_theta = self.GetVectorAngle(trj1[0][0], trj1[1][0], trj1[0][-1], trj1[1][-1])
        trj1 = self.RotateTrj(-1 * rotate_theta, trj1)  # 旋转轨迹1
        trj2 = self.RotateTrj(-1 * rotate_theta, trj2)  # 旋转轨迹2

        cross_point = []  # 存储交点的列表
        separate_trj2 = self.SeparateTrj(trj2)  # 将轨迹2分段处理
        for sub_trj in separate_trj2:
            point = self.CrossOnePiece(trj1, sub_trj)  # 获取轨迹1和子轨迹的交点
            if point is not None:
                # 如果交点存在，则进行旋转和移动回原始坐标系
                point = self.RotateTrj(rotate_theta, point)
                point[0] -= move_vector[0]
                point[1] -= move_vector[1]
                cross_point.append(point)  # 添加交点
        return cross_point

    def GetSPosition(self, trj, point):
        x = point[0]
        y = point[1]
        dis = (x - trj[0]) ** 2 + (y - trj[1]) ** 2  # 计算点到轨迹起始点的距离
        s_axis_interval = np.sqrt(np.diff(trj[0]) ** 2 + np.diff(trj[1]) ** 2)  # 计算每段轨迹的弧长
        s_axis = [0.0]
        for i in range(len(s_axis_interval)):
            s_axis.append(s_axis[-1] + s_axis_interval[i])  # 计算整个轨迹的s轴（累积弧长）
        return s_axis[np.argmin(dis)]  # 返回距离最近的s值

    def GetXYPosition(self, trj, s):
        s_axis_interval = np.sqrt(np.diff(trj[0]) ** 2 + np.diff(trj[1]) ** 2)  # 计算每段轨迹的弧长
        s_axis = [0.0]
        for i in range(len(s_axis_interval)):
            s_axis.append(s_axis[-1] + s_axis_interval[i])  # 计算整个轨迹的s轴
        index = np.argmin((s_axis - s) ** 2)  # 找到最接近s的索引
        return trj[0][index], trj[1][index]  # 返回对应位置的x和y坐标

    def GetStateByS(self, trajectory, s):
        trj = [[state.x for state in trajectory],
               [state.y for state in trajectory]]
        s_axis_interval = np.sqrt(np.diff(trj[0]) ** 2 + np.diff(trj[1]) ** 2)  # 计算每段轨迹的弧长
        s_axis = [0.0]
        for i in range(len(s_axis_interval)):
            s_axis.append(s_axis[-1] + s_axis_interval[i])  # 计算整个轨迹的s轴
        s_axis = np.array(s_axis)
        index = np.argmin((s_axis - s) ** 2)  # 找到最接近s的索引
        return trajectory[index]  # 返回对应状态

    def GetStateByPos(self, trajectory, point):
        x = point[0]
        y = point[1]
        trj = [[state.x for state in trajectory],
               [state.y for state in trajectory]]
        dis = (x - trj[0]) ** 2 + (y - trj[1]) ** 2  # 计算点到轨迹所有点的距离
        index = np.argmin(dis)  # 找到最接近的点
        return trajectory[index]  # 返回对应的状态

    def GetSimilarState(self, trajectory, point):
        state = State()
        x = point[0]
        y = point[1]
        trj = [[state.x for state in trajectory],
               [state.y for state in trajectory]]
        dis = (x - trj[0]) ** 2 + (y - trj[1]) ** 2  # 计算点到轨迹所有点的距离
        index = np.argmin(dis)  # 找到最接近的点
        state.SetValue(trajectory[index])  # 设置状态值
        state.x = x
        state.y = y
        return state  # 返回最相似的状态

    def GetTimeToPoint(self, trj, trj_idx, point):
        x = point[0]
        y = point[1]
        dis = (x - trj[0]) ** 2 + (y - trj[1]) ** 2  # 计算点到轨迹的距离
        return trj_idx[np.argmin(dis)] * self.simulation_step  # 返回到该点的时间

    def GetInteractConnector(self, vehicle):
        connector = self.road.connector_list[vehicle.connector_id]
        overlap_connector_ids = self.road.overlap[connector.id]
        interact_connector_ids = []
        for connector_id in overlap_connector_ids:
            # if self.road.signal_list[self.road.connector_list[connector_id].signal_id].color == 'green':
            interact_connector_ids.append(connector_id)
        return interact_connector_ids

    def XYToFrenet(self, trajectory, behind_state):
        # 检查轨迹是否是二维列表，如果不是，则构建一个参考线
        if len(trajectory) != 2:
            reference_line = [[], []]
            for state in trajectory:
                reference_line[0].append(state.x)
                reference_line[1].append(state.y)
        else:
            reference_line = trajectory

        reference_line = np.array(reference_line)  # 将参考线转换为NumPy数组
        pos = [behind_state.x, behind_state.y]  # 获取后车位置
        dis = np.sqrt((reference_line[0] - pos[0]) ** 2 + (reference_line[1] - pos[1]) ** 2)  # 计算每个参考点到后车的距离
        index = np.argmin(dis)  # 找到距离后车最近的参考点
        nearest_state = trajectory[index]  # 获取最近参考点的状态
        offset = (pos[0] - nearest_state.x) * np.cos(nearest_state.yaw) + \
                 (pos[1] - nearest_state.y) * np.sin(nearest_state.yaw)  # 计算偏移量

        # 计算沿参考线方向的最近点坐标
        nearest_x = nearest_state.x + offset * np.cos(nearest_state.yaw)
        nearest_y = nearest_state.y + offset * np.sin(nearest_state.yaw)

        # 计算参考线的s轴间距
        s_axis_interval = np.sqrt(np.diff(reference_line[0]) ** 2 + np.diff(reference_line[1]) ** 2)
        s_axis = [0.0]
        for i in range(len(s_axis_interval)):
            s_axis.append(s_axis[-1] + s_axis_interval[i])  # 计算s轴上的每个点
        s_axis = np.array(s_axis)

        # 计算当前位置在s轴上的坐标
        s = s_axis[index] + offset

        # 计算法向距离n
        n = np.sqrt((pos[0] - nearest_x) ** 2 + (pos[1] - nearest_y) ** 2)

        # 判断偏移方向来确定n的符号
        if (pos[0] - nearest_state.x) * np.sin(nearest_state.yaw) - \
                (pos[1] - nearest_state.y) * np.cos(nearest_state.yaw) > 0:
            n = -1.0 * n
        return s, n  # 返回s和n值

    def GetReferenceLine(self, front_veh, behind_veh):
        # 根据前车和后车的行驶方向，获取参考线
        if front_veh.direction == 'straight':
            if behind_veh.direction == 'left turn':
                front_reference_line, _, front_trj_idx = front_veh.GetRearEndTrj()  # 获取前车后端轨迹
                behind_reference_line, _, behind_trj_idx = behind_veh.GetFrontEndTrj()  # 获取后车前端轨迹
            else:
                _, front_reference_line, front_trj_idx = front_veh.GetRearEndTrj()  # 获取前车后端轨迹
                behind_reference_line, _, behind_trj_idx = behind_veh.GetFrontEndTrj()  # 获取后车前端轨迹
        elif front_veh.direction == 'left turn':
            if behind_veh.direction == 'straight':
                _, front_reference_line, front_trj_idx = front_veh.GetRearEndTrj()  # 获取前车后端轨迹
                _, behind_reference_line, behind_trj_idx = behind_veh.GetFrontEndTrj()  # 获取后车前端轨迹
            else:
                _, front_reference_line, front_trj_idx = front_veh.GetRearEndTrj()  # 获取前车后端轨迹
                behind_reference_line, _, behind_trj_idx = behind_veh.GetFrontEndTrj()  # 获取后车前端轨迹
        elif front_veh.direction == 'right turn':
            if behind_veh.direction == 'straight':
                front_reference_line, _, front_trj_idx = front_veh.GetRearEndTrj()  # 获取前车后端轨迹
                _, behind_reference_line, behind_trj_idx = behind_veh.GetFrontEndTrj()  # 获取后车前端轨迹
            else:
                front_reference_line, _, front_trj_idx = front_veh.GetRearEndTrj()  # 获取前车后端轨迹
                _, behind_reference_line, behind_trj_idx = behind_veh.GetFrontEndTrj()  # 获取后车前端轨迹
        return front_reference_line, behind_reference_line, front_trj_idx, behind_trj_idx  # 返回前后车的参考线和轨迹索引

    def LimitPlan(self, front_veh, behind_veh):
        # 获取前后车的参考线和轨迹索引
        front_reference_line, behind_reference_line, front_trj_idx, behind_trj_idx = self.GetReferenceLine(front_veh,
                                                                                                           behind_veh)

        # 如果参考线的点数不足，直接返回后车对象
        if len(front_reference_line[0]) < 2 or len(behind_reference_line[0]) < 2:
            return behind_veh

        # 计算前车和后车的冲突点
        conflict_point = self.GetCrossPoint(front_reference_line, behind_reference_line)
        if conflict_point == []:  # 如果没有冲突点，则返回后车对象
            return behind_veh

        # 计算前车和后车到冲突点的时间
        front_time = self.GetTimeToPoint(front_reference_line, front_trj_idx, conflict_point[0])
        behind_time = self.GetTimeToPoint(behind_reference_line, behind_trj_idx, conflict_point[0])

        # 如果后车的时间大于前车的时间，则返回后车对象
        if behind_time > front_time:
            return behind_veh

        # 否则，调整后车的目标点以避免碰撞
        temp = behind_veh.targets.pop()  # 临时保存后车的最后一个目标
        behind_s = self.GetSPosition(behind_reference_line, conflict_point[0])  # 计算后车在s轴上的位置
        behind_veh.targets.append(self.GetStateByS(behind_veh.trajectory, max((behind_s - 5.0), 0.0)))  # 更新后车的目标位置
        behind_veh.targets.append(temp)  # 恢复后车的原始目标
        behind_veh.time_limit_to_targets.append(front_time)  # 添加时间限制

        if front_time is not None:
            if front_time > 20:  # 如果前车时间大于20，则做停靠处理
                stop = 1

        behind_veh.trajectory = []  # 清空后车的轨迹
        return behind_veh  # 返回更新后的后车对象

    def LimitPlanTrj(self, behind_veh):
        # 如果没有时间限制目标
        if len(behind_veh.time_limit_to_targets) == 0:
            # 清空速度和轨迹
            self.path_planner.speed = []
            self.path_planner.trajectory = []
            # 设置当前状态为后方车辆轨迹的最后一个状态
            self.path_planner.current_state = behind_veh.trajectory[-1]
            # 设置目标为后方车辆目标的最后一个目标
            self.path_planner.target = behind_veh.targets[-1]
            # 如果当前状态与目标非常接近，返回当前轨迹
            if (behind_veh.trajectory[-1].x - behind_veh.targets[-1].x) ** 2 + \
                    (behind_veh.trajectory[-1].y - behind_veh.targets[-1].y) ** 2 < 5e-1:
                return behind_veh.trajectory
            # 使用神经网络规划曲线
            self.path_planner.PlanCurveByNN()
            # 获取速度限制和s轴
            speed_limit, s_axis = self.path_planner.GetSpeedLimit()
            # 基于速度限制进行速度规划
            self.path_planner.SpeedPlan(speed_limit, s_axis)
            # 设置轨迹
            self.path_planner.SetTrajectory()
            return self.path_planner.trajectory
        # 如果有时间限制目标
        behind_veh.trajectory = []
        # 获取当前连接器
        current_connector = self.road.connector_list[behind_veh.connector_id]
        s = []
        # 弹出最后一个目标作为最终目标
        final_target = behind_veh.targets.pop()
        # 如果目标数量大于等于2，设定停止标志
        if len(behind_veh.targets) >= 2:
            stop = 1
        trj = current_connector.centerline[0]
        # 将后方车辆的轨迹添加到中心线
        for state in behind_veh.trajectory:
            trj[0].append(state.x)
            trj[1].append(state.y)
        # 为每个目标计算s位置
        for target in behind_veh.targets:
            target_point = [target.x, target.y]
            s.append(self.GetSPosition(trj, target_point))
        # 根据s值对目标进行排序
        sort_list = sorted(enumerate(s), key=lambda x: x[1], reverse=False)
        # 如果有多个目标，合并接近的目标
        if len(sort_list) > 1:
            i = 1
            while i < len(sort_list):
                if abs(sort_list[i][1] - sort_list[i - 1][1]) < 1:
                    # 如果两个目标接近，则合并它们的时间限制
                    temp_time_limit_to_target = max(behind_veh.time_limit_to_targets[sort_list[i][0]],
                                                    behind_veh.time_limit_to_targets[sort_list[i - 1][0]])
                    behind_veh.time_limit_to_targets[sort_list[i - 1][0]] = temp_time_limit_to_target
                    behind_veh.time_limit_to_targets.pop(sort_list[i][0])
                    sort_list.pop(i)
                    i -= 1
                i += 1
        # 根据排序后的索引生成新的目标列表和时间限制列表
        sort_veh_index = [x[0] for x in sort_list]
        sorted_targets = []
        sorted_time = []
        for i in range(len(sort_veh_index)):
            sorted_targets.append(behind_veh.targets[sort_veh_index[i]])
            sorted_time.append(behind_veh.time_limit_to_targets[sort_veh_index[i]])
        behind_veh.targets = sorted_targets
        behind_veh.targets.append(final_target)  # 将最终目标加到目标列表中
        behind_veh.time_limit_to_targets = sorted_time
        # 对每个目标进行轨迹规划
        for i in range(len(behind_veh.targets)):
            # 如果是第一个目标
            if i == 0:
                self.path_planner.speed = []
                self.path_planner.trajectory = [behind_veh.state]
                self.path_planner.current_state = behind_veh.state
                self.path_planner.target = behind_veh.targets[i]
                # 如果当前状态和目标非常接近，停车
                if (behind_veh.state.x - behind_veh.targets[i].x) ** 2 + \
                        (behind_veh.state.y - behind_veh.targets[i].y) ** 2 < 1e-2:
                    self.path_planner.StandStill(behind_veh.time_limit_to_targets[i])
                    behind_veh.trajectory.extend(self.path_planner.trajectory[1:])
                    continue
                # 使用神经网络规划曲线
                self.path_planner.PlanCurveByNN()
                # 进行速度规划，考虑时间限制
                self.path_planner.SpeedPlanWithLimitation(behind_veh.time_limit_to_targets[i])
            # 如果是最后一个目标
            elif i == len(behind_veh.time_limit_to_targets):
                self.path_planner.speed = []
                self.path_planner.trajectory = []
                self.path_planner.current_state = behind_veh.trajectory[-1]
                self.path_planner.target = behind_veh.targets[i]
                # 如果当前状态和目标非常接近，跳过
                if (behind_veh.trajectory[-1].x - behind_veh.targets[i].x) ** 2 + \
                        (behind_veh.trajectory[-1].y - behind_veh.targets[i].y) ** 2 < 5e-1:
                    continue
                # 使用神经网络规划曲线
                self.path_planner.PlanCurveByNN()
                # 获取速度限制和s轴
                speed_limit, s_axis = self.path_planner.GetSpeedLimit()
                # 进行速度规划
                self.path_planner.SpeedPlan(speed_limit, s_axis)
                self.path_planner.SetTrajectory()
            # 如果是中间目标
            else:
                self.path_planner.speed = []
                self.path_planner.trajectory = []
                self.path_planner.current_state = behind_veh.trajectory[-1]
                self.path_planner.target = behind_veh.targets[i]
                # 如果当前状态和目标非常接近，处理停车
                if (behind_veh.trajectory[-1].x - behind_veh.targets[i].x) ** 2 + \
                        (behind_veh.trajectory[-1].y - behind_veh.targets[i].y) ** 2 < 1e-2:
                    if behind_veh.time_limit_to_targets[i] > behind_veh.time_limit_to_targets[i - 1]:
                        self.path_planner.StandStill(behind_veh.time_limit_to_targets[i] -
                                                     behind_veh.time_limit_to_targets[i - 1])
                        behind_veh.trajectory.extend(self.path_planner.trajectory[1:])
                    continue
                # 使用神经网络规划曲线
                self.path_planner.PlanCurveByNN()
                # 如果当前目标的时间限制大于前一个目标，进行速度规划
                if behind_veh.time_limit_to_targets[i] > behind_veh.time_limit_to_targets[i - 1]:
                    self.path_planner.SpeedPlanWithLimitation(behind_veh.time_limit_to_targets[i] -
                                                              behind_veh.time_limit_to_targets[i - 1])
                behind_veh.trajectory.extend(self.path_planner.trajectory[1:])
        # 返回最终的轨迹
        return behind_veh.trajectory

    def CarFollow(self, ego_veh, front_veh):
        # 获取自车和前车的连接器
        ego_connector = self.road.connector_list[ego_veh.connector_id]
        front_connector = self.road.connector_list[front_veh.connector_id]

        # 获取前车的轨迹
        front_trj = front_veh.trajectory

        # 初始化IDM参数
        IDM_param = IDMParameter()

        # 用于存储自车的速度
        new_speed = []
        new_speed.append(ego_veh.state.speed)

        # 计算自车与前车在道路上距离的起始位置
        ego_s = self.GetSPosition(ego_connector.centerline[0],
                                  [ego_veh.state.x, ego_veh.state.y])
        front_s = self.GetSPosition(ego_connector.centerline[0],
                                    [front_veh.state.x, front_veh.state.y])

        # 计算自车和前车之间的间距
        s = front_s - ego_s - ego_veh.vehicle_type.length

        # 判断自车与前车是否在同一连接器上
        if ego_veh.connector_id == front_veh.connector_id:
            overlap_range = len(front_trj)  # 如果在同一连接器上，重叠范围为前车轨迹的长度
        else:
            for i in range(len(front_trj)):
                state = front_trj[i]
                _, n = self.XYToFrenet(ego_veh.trajectory, state)
                if abs(n) > ego_veh.vehicle_type.width:
                    overlap_range = i  # 找到自车与前车轨迹的重叠范围
                    break

        # 使用车-follow模型进行车辆跟随控制
        for i in range(overlap_range):
            front_state = front_trj[i]

            # 计算自车和前车速度差
            dv = new_speed[i] - front_state.speed

            # 计算期望的安全间距
            desired_space = IDM_param.min_gap + max(0, new_speed[i] * IDM_param.time_headway +
                                                    new_speed[i] * dv / (2 * np.sqrt(IDM_param.a * IDM_param.b)))

            # 更新间距
            s = s - dv * self.simulation_step

            # 计算加速度
            acc = IDM_param.a * (1 -(new_speed[i] / IDM_param.desired_speed) ** IDM_param -
                                 (desired_space / s) ** 2)

            # 计算新速度
            temp_speed = new_speed[i] + acc * self.simulation_step

            # 如果加速度小于阈值，则将速度设为0
            if acc < 2e-1:
                temp_speed = 0.0

            # 将新速度添加到速度列表中
            new_speed.append(temp_speed)

        # 更新自车速度和路径规划
        ego_veh.speed = new_speed
        self.path_planner.speed = []
        self.path_planner.trajectory = []
        self.path_planner.current_state = ego_veh.state
        self.path_planner.target = ego_veh.targets[-1]
        self.path_planner.trajectory = ego_veh.trajectory
        self.path_planner.speed = new_speed
        self.path_planner.SetTrajectoryWithTimeSpeed()
        ego_veh.trajectory = self.path_planner.trajectory

        # 如果轨迹为空，则重新规划路径
        if ego_veh.trajectory == []:
            self.path_planner.current_state = ego_veh.state
        else:
            self.path_planner.current_state = ego_veh.trajectory[-1]

        # 判断是否接近目标，如果接近目标则重新规划路径
        if ((self.path_planner.current_state.x - ego_veh.targets[-1].x) ** 2 +
            (self.path_planner.current_state.y - ego_veh.targets[-1].y) ** 2) > 5e-1:
            self.path_planner.speed = []
            self.path_planner.trajectory = []
            self.path_planner.target = ego_veh.targets[-1]
            self.path_planner.PlanCurveByNN()  # 使用神经网络规划曲线
            speed_limit, s_axis = self.path_planner.GetSpeedLimit()  # 获取速度限制和轴向
            self.path_planner.SpeedPlan(speed_limit, s_axis)  # 速度规划
            self.path_planner.SetTrajectory()  # 设置新的轨迹
            ego_veh.trajectory.extend(self.path_planner.trajectory)  # 扩展自车轨迹

        return ego_veh.trajectory  # 返回自车的轨迹

    def GetDecisionResult(self, ego_vehicle, behind_obj):
        # 获取自车和后方物体的参考线
        ego_reference_line, behind_reference_line, ego_trj_idx, behind_trj_idx \
            = self.GetReferenceLine(ego_vehicle, behind_obj)

        # 如果参考线的长度小于2，返回True，表示没有冲突
        if len(ego_reference_line[0]) < 2 or len(behind_reference_line[0]) < 2:
            return True

        # 获取自车和后方物体的交点
        conflict_point = self.GetCrossPoint(ego_reference_line, behind_reference_line)

        # 如果没有交点，判断是否存在冲突
        if conflict_point == []:
            if behind_obj.direction == 'straight':
                s, _ = self.XYToFrenet(behind_obj.trajectory, ego_vehicle.state)
                if s > 0:
                    return True, 'No conflict'  # 无冲突
                else:
                    return False  # 存在冲突
            else:
                return True, 'No conflict'  # 无冲突

        # 获取自车和后方物体到交点的时间
        ego_veh_time = self.GetTimeToPoint(ego_reference_line, ego_trj_idx, conflict_point[0])
        behind_veh_time = self.GetTimeToPoint(behind_reference_line, behind_trj_idx, conflict_point[0])

        # 如果自车到交点的时间为0，表示已经到达交点，返回True
        if ego_veh_time == 0:
            return True

        # 获取交点的s坐标
        behind_s = self.GetSPosition(behind_reference_line, conflict_point[0])
        ego_s = self.GetSPosition(ego_reference_line, conflict_point[0])

        # 判断自车和后方物体是否为左转或右转，如果是，判断谁先到达交点
        if (ego_vehicle.direction == 'right turn' and behind_obj.direction == 'left turn') or \
                (ego_vehicle.direction == 'left turn' and behind_obj.direction == 'right turn'):
            if ego_veh_time < behind_veh_time:
                return True
            else:
                return False

        # 根据速度和距离计算后车的加速度和时间差
        behind_speed = behind_obj.state.speed
        if behind_s < 0.5 * behind_speed * ego_veh_time:
            D = ego_veh_time + \
                behind_speed / behind_obj.vehicle_type.max_acc - \
                behind_s / behind_speed
            A = -1 * behind_speed ** 2 / behind_s / 2
        elif behind_s < behind_speed * ego_veh_time:
            D = 0.5 * ego_veh_time + \
                behind_speed / behind_obj.vehicle_type.max_acc - \
                behind_s / behind_speed
            A = -2 * (behind_speed * ego_veh_time - behind_s) / ego_veh_time ** 2
        else:
            D = 0.0
            A = 2 * (behind_s / ego_veh_time ** 2 - behind_speed / ego_veh_time)

        # 判断是否存在冲突并返回结果
        if ego_vehicle.direction == 'left turn':
            if behind_obj.vehicle_type.value == 'car':
                if A > -0.15 and D < 3.39:
                    return True
                else:
                    return False
            else:
                if A > -0.5 and D < 4.0:
                    return True
                else:
                    return False
        else:
            if behind_obj.vehicle_type.value == 'car':
                if A > -0.15 and D < 3.39:
                    return True
                else:
                    return False
            else:
                if A > 0.0 and D < 2.0:
                    return True
                else:
                    return False

    # 更新函数，用于递归更新交互对象的索引
    def Update(self, i, interact_obj_idxs, vehs):
        # 更新当前交互对象的索引
        interact_obj_idxs[i] += 1
        # 如果索引大于车辆列表长度，检查是否是第一个索引
        if interact_obj_idxs[i] > len(vehs[i]):
            if i == 0:
                print("wrong index")  # 如果是第一个索引，输出错误信息
            else:
                # 如果不是第一个索引，递归更新前一个交互对象的索引
                self.Update(i - 1, interact_obj_idxs, vehs)
        # 如果不是最后一个车辆，重置下一个交互对象的索引
        elif i < len(vehs) - 1:
            interact_obj_idxs[i + 1:-1] = 0
        return interact_obj_idxs

    # 添加新的对象到交互图中
    def AddObj(self, vehicle):
        # 获取与车辆交互的连接器ID列表
        interact_connector_ids = self.GetInteractConnector(vehicle)
        interact_lane_ids = []
        # 根据连接器ID找到所有的交互车道ID
        for connector_id in interact_connector_ids:
            connector = self.road.connector_list[connector_id]
            # 如果连接器的起始车道ID大于1，遍历多个车道ID
            if len(connector.start_lane_id) > 1:
                for i in range(len(connector.start_lane_id)):
                    interact_lane_ids.append(connector.id * 10000 + connector.start_lane_id[i])
            else:
                # 否则只添加一个车道ID
                interact_lane_ids.append(connector.id * 10000 + connector.start_lane_id[0])

        # 初始化交互对象的索引数组，默认为0
        interact_obj_idxs = np.zeros(len(interact_lane_ids), dtype=int)
        front_objs = []
        behind_objs = []
        decision_result = np.tile(False, len(interact_obj_idxs))  # 初始化决策结果为False
        vehs = []

        # 获取每条车道的车辆
        for interact_lane_id in interact_lane_ids:
            lane_vehs = []
            connector_id = round(interact_lane_id / 10000)
            lane_id = interact_lane_id % 10000
            connector_vehs = self.connector_car_dict[connector_id]
            for veh in connector_vehs:
                veh_obj = self.obj_list[veh]
                if veh_obj.start_lane_id == lane_id:
                    lane_vehs.append(veh)
            vehs.append(lane_vehs)

        front_in_graph = []
        behind_in_graph = []

        # 获取车辆与其他车辆在逻辑图中的相对位置
        for k, v in self.logical_graph.items():
            if k == vehicle.id:
                behind_in_graph.extend(v)
            if vehicle.id in v:
                front_in_graph.append(k)

        # 更新交互对象的索引
        for i in range(len(vehs)):
            for j in range(len(behind_in_graph)):
                if behind_in_graph[j] in vehs[i]:
                    interact_obj_idxs[i] = vehs[i].index(behind_in_graph[j])
            for j in range(len(front_in_graph)):
                if front_in_graph[j] in vehs[i]:
                    if interact_obj_idxs[i] == 0:
                        interact_obj_idxs[i] = vehs[i].index(front_in_graph[j]) + 1

        # 分析决策结果
        first_decision = True
        while all(decision_result) == False:
            front_objs = []
            behind_objs = []
            # 找到前方和后方的对象
            for i in range(len(interact_lane_ids)):
                if vehs[i] == []:
                    decision_result[i] = True
                    front_objs.append(None)
                    behind_objs.append(None)
                    continue
                if interact_obj_idxs[i] == 0:
                    front_objs.append(None)
                else:
                    front_objs.append(vehs[i][int(interact_obj_idxs[i]) - 1])
                if interact_obj_idxs[i] == len(vehs[i]):
                    behind_objs.append(None)
                else:
                    behind_objs.append(vehs[i][interact_obj_idxs[i]])

            # 获取决策结果
            conflict_state = np.ones(len(interact_lane_ids), dtype=bool)
            for i in range(len(behind_objs)):
                if behind_objs[i] is None:
                    decision_result[i] = True
                    continue
                behind_obj = self.obj_list[behind_objs[i]]
                result = self.GetDecisionResult(vehicle, behind_obj)
                if isinstance(result, bool):
                    decision_result[i] = result
                else:
                    decision_result[i] = result
                    conflict_state[i] = False
                if decision_result[i] == False:
                    # 如果决策结果为False，更新逻辑图
                    if vehicle.id in self.logical_graph.keys():
                        if behind_objs[i] in self.logical_graph[vehicle.id]:
                            self.logical_graph[vehicle.id].remove(behind_objs[i])
                    interact_obj_idxs = self.Update(i, interact_obj_idxs, vehs)
                    temp = vehicle.targets.pop()
                    vehicle.targets = [temp]
                    vehicle.time_limit_to_targets = []
                    break

            # 如果所有决策结果都为True，且是第一次决策，跳出循环
            if all(decision_result) == True and first_decision == True:
                break
            first_decision = False

            # 生成让行轨迹
            for i in range(len(front_objs)):
                if front_objs[i] is not None:
                    if front_objs[i] in self.logical_graph.keys():
                        if vehicle.id in self.logical_graph[front_objs[i]]:
                            continue
                    front_obj = self.obj_list[front_objs[i]]
                    print("Yielding to Vehicle: ", front_obj.id, "by LimitPlan")
                    vehicle = self.LimitPlan(front_obj, vehicle)
                    vehicle.trajectory = self.LimitPlanTrj(vehicle)

        # 调整后方车辆的轨迹
        for i in range(len(behind_objs)):
            if conflict_state[i] == True:
                front_veh = front_objs[i]
                if front_veh is not None:
                    if front_veh not in self.logical_graph.keys():
                        self.logical_graph[front_veh] = []
                    if vehicle.id not in self.logical_graph[front_veh]:
                        self.logical_graph[front_veh].append(vehicle.id)
            if behind_objs[i] is None:
                continue
            if vehicle.id in self.logical_graph.keys():
                if behind_objs[i] in self.logical_graph[vehicle.id]:
                    continue
            if conflict_state[i] == True:
                print("Behind Vehicle Yielding: ", behind_objs[i], "by LimitPlan")
                # 调整后方车辆的轨迹
                self.obj_list[behind_objs[i]] = self.LimitPlan(vehicle, self.obj_list[behind_objs[i]])
                self.obj_list[behind_objs[i]].trajectory = self.LimitPlanTrj(self.obj_list[behind_objs[i]])
                if vehicle.id not in self.logical_graph:
                    self.logical_graph[vehicle.id] = []
                self.logical_graph[vehicle.id].append(behind_objs[i])
        '''
        # 清空目标和时间限制
        for veh in self.obj_list.values():
            veh.targets = [veh.targets[-1]]
            veh.time_limit_to_targets = []
        '''

