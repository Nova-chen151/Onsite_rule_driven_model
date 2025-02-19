import csv
import numpy as np
import math

from planner.planner import PathPlanner
from util.state import State
from util.interface import Field
from util.interface import SignalSetting
from util.interface import SignalToConnectorSetting
from util.interface import ConnectorSetting
from util.interface import SectionSetting
from util.interactionsolver import InteractionSolver

class Position:
    def __init__(self):
        self.x = 0.0        # x坐标
        self.y = 0.0        # y坐标
        self.yaw = 0.0      # 朝向角度，单位：弧度


class Section:
    def __init__(self):
        self.id = 0                                 # 道路段的ID
        self.end_list = [Position(), Position()]    # 两个端点的位置列表
        self.lane_number = 1                        # 车道数量，默认为1
        self.connector_id_list = []                 # 连接此段道路的所有连接器ID列表
        self.car_type = None                        # 适用的车辆类型
        self.born_interval = []                     # 生成间隔

    def SetData(self, field):
        self.id = field.section_id                  # 设置ID
        self.lane_number = field.lane_number        # 设置车道数量
        self.end_list[0].x = field.end1_x           # 设置左端点的x坐标
        self.end_list[0].y = field.end1_y           # 设置左端点的y坐标
        self.end_list[0].yaw = field.end1_yaw       # 设置左端点的朝向
        self.end_list[1].x = field.end2_x           # 设置右端点的x坐标
        self.end_list[1].y = field.end2_y           # 设置右端点的y坐标
        self.end_list[1].yaw = field.end2_yaw       # 设置右端点的朝向
        self.car_type = field.car_type              # 设置车辆类型

    def GetDividePos(self, divide, *args):
        # 处理偏移量，如果没有传递，使用默认值0.0
        if len(args) > 0:
            offset = args[0]
        else:
            offset = 0.0

        # 计算道路的长度
        section_dist = np.sqrt(
            (self.end_list[0].x - self.end_list[1].x) ** 2 + (self.end_list[0].y - self.end_list[1].y) ** 2)

        # 计算偏移权重
        offset_weight = offset / section_dist

        # 计算分割位置的权重miu，miu是根据分割比例和偏移量计算的
        miu = offset_weight + (1.0 - 2.0 * offset_weight) * divide

        # 计算分割位置的坐标和朝向
        result = Position()
        result.x = (1.0 - miu) * self.end_list[0].x + miu * self.end_list[1].x
        result.y = (1.0 - miu) * self.end_list[0].y + miu * self.end_list[1].y
        result.yaw = (1.0 - miu) * self.end_list[0].yaw + miu * self.end_list[1].yaw

        return result


class Connector:
    def __init__(self):
        self.id = 0                     # 连接器的ID
        self.start_section = None       # 起始段道路
        self.start_lane_id = []         # 起始车道ID列表
        self.end_section = None         # 结束段道路
        self.end_lane_id = []           # 结束车道ID列表
        self.boundaries = []            # 边界列表
        self.car_type = None            # 适用的车辆类型
        self.signal_id = 0              # 交通信号灯ID
        self.direction = 'turn'         # 连接器的方向（例如“turn”表示转弯）
        self.priority = 0               # 优先级，数值越大，优先级越高
        self.centerline = []            # 连接器的中心线，可能是一个坐标点的列表
        flow = 400                      # 默认流量
        self.arrive_rate = flow / 3600  # 计算车辆到达率，单位：辆/秒

    def SetData(self, field):
        self.id = field.connector_id  # 设置连接器的ID
        self.car_type = field.car_type  # 设置车辆类型
        self.start_section = field.start_section_id  # 设置起始路段ID
        if isinstance(field.start_lane_id, list):
            # 如果起始车道ID是列表形式，则逐个添加
            for i in range(len(field.start_lane_id)):
                self.start_lane_id.append(field.start_lane_id[i])
        else:
            # 如果起始车道ID是字符串，则将其拆分并转换为整数
            self.start_lane_id = [int(lane_id) for lane_id in field.start_lane_id.split()]

        self.end_section = field.end_section_id  # 设置结束路段ID
        if isinstance(field.end_lane_id, list):
            # 如果结束车道ID是列表形式，则逐个添加
            for i in range(len(field.end_lane_id)):
                self.end_lane_id.append(field.end_lane_id[i])
        else:
            # 如果结束车道ID是字符串，则将其拆分并转换为整数
            self.end_lane_id = [int(lane_id) for lane_id in field.end_lane_id.split()]
        self.direction = field.direction  # 设置方向

    def SetLine(self, start_divide, end_divide):
        start_state = State()   # 创建起始状态对象
        target = State()        # 创建目标状态对象
        # 获取起始和目标位置
        start_position = self.start_section.GetDividePos(start_divide)
        target_position = self.end_section.GetDividePos(end_divide)

        # 设置起始状态
        start_state.x = start_position.x
        start_state.y = start_position.y
        start_state.yaw = start_position.yaw

        # 设置目标状态
        target.x = target_position.x
        target.y = target_position.y
        target.yaw = target_position.yaw

        # 创建路径规划对象并执行规划
        planner = PathPlanner(start_state, target)
        planner.PlanCurveByNN()

        # 返回规划得到的曲线点
        return planner.curve_points

    def SetBoundary(self):
        # 设置边界线，计算起始和结束车道的分割比例
        self.boundaries.append(self.SetLine((self.start_lane_id[0] - 1) / self.start_section.lane_number,
                                            (self.end_lane_id[0] - 1) / self.end_section.lane_number))
        self.boundaries.append(self.SetLine(self.start_lane_id[-1] / self.start_section.lane_number,
                                            self.end_lane_id[-1] / self.end_section.lane_number))

    def SetCenterLine(self, start_lane_id, end_lane_id):
        # 计算中心线的分割比例，并调用 SetLine 函数设置中心线
        self.centerline.append(self.SetLine((start_lane_id - 0.5) / self.start_section.lane_number,
                                            (end_lane_id - 0.5) / self.end_section.lane_number))

    def GetLength(self):
        # 计算路径上的每个点之间的欧几里得距离，并求和得到总长度
        return sum(np.sqrt(np.diff(self.centerline[0][0]) ** 2 + \
                           np.diff(self.centerline[0][1]) ** 2))

    def GetSPosition(self, vehicle):
        # 计算车辆当前点与路径中心线点的距离
        dis = (vehicle.state.x - self.centerline[0][0]) ** 2 + \
              (vehicle.state.y - self.centerline[0][1]) ** 2
        # 计算路径中心线的每段的距离
        s_axis_interval = np.sqrt(np.diff(self.centerline[0][0]) ** 2 + \
                                  np.diff(self.centerline[0][1]) ** 2)

        # 计算S轴位置
        s_axis = [0.0]
        for i in range(len(s_axis_interval)):
            s_axis.append(s_axis[-1] + s_axis_interval[i])

        # 返回与车辆当前位置最近的S位置
        return s_axis[np.argmin(dis)]

class Signal:
    def __init__(self):
        # 信号灯初始设置
        self.timestamp = 0.0        # 当前时间戳
        self.id = 0                 # 信号灯ID
        self.connector_id = []      # 连接器ID列表
        self.position = 0.0         # 信号灯位置
        self.color = 'red'          # 初始信号灯颜色为红色
        self.offset = 0.0           # 偏移量
        # 默认信号灯周期表（红、绿、黄灯的持续时间）
        self.schedule = {'red': list(range(30)),
                         'green': list(range(30, 60)),
                         'yellow': list(range(60, 63))}

    def SetSchedule(self, schedule):
        # 设置信号灯的周期
        self.schedule['red'] = list(range(schedule[0]))                                 # 红灯持续时间
        self.schedule['green'] = list(range(schedule[0], schedule[0] + schedule[1]))    # 绿灯持续时间
        self.schedule['yellow'] = list(
            range(schedule[0] + schedule[1], schedule[0] + schedule[1] + schedule[2]))  # 黄灯持续时间

    def SetData(self, signal_setting):
        # 设置信号灯的基本数据
        self.id = signal_setting.id                     # 设置信号灯ID
        self.connector_id.append(signal_setting.id)     # 添加连接器ID
        self.SetSchedule(signal_setting.schedule)       # 设置周期
        self.offset = signal_setting.offset             # 设置偏移量

    def GetCycleLen(self):
        # 获取一个周期的总长度
        result = 0
        # 遍历信号灯的周期表，计算所有状态的总长度
        for status in self.schedule.values():
            result += len(status)
        return result

    def UpdateColor(self):
        # 更新信号灯颜色
        cycle_len = self.GetCycleLen()  # 获取周期长度
        cycle_time = (self.timestamp + self.offset) % cycle_len  # 计算当前时间点的周期时间
        # 判断当前时间点对应的信号灯颜色
        if cycle_time >= self.schedule['red'][0] and cycle_time < self.schedule['red'][-1]:
            self.color = 'red'
        elif cycle_time >= self.schedule['green'][0] and cycle_time < self.schedule['green'][-1]:
            self.color = 'green'
        elif cycle_time > self.schedule['yellow'][0] and cycle_time < self.schedule['yellow'][-1]:
            self.color = 'yellow'

class Road:
    def __init__(self):
        self.connector_list = {}    # 连接器列表
        self.signal_list = {}       # 信号灯列表
        self.signal_to_lane = {}    # 信号灯与车道的映射
        self.section_list = {}      # 路段列表
        self.overlap = {}           # 重叠连接器对

    def SetOverlap(self):
        for connector1 in self.connector_list.values():
            s = []  # 存储交叉点的相对位置
            self.overlap[connector1.id] = []  # 初始化该连接器的重叠列表
            for connector2 in self.connector_list.values():
                if connector1.id == connector2.id:
                    continue  # 跳过自己与自己的重叠
                # 判断连接器1的方向，并选择适当的轨迹（边界或中心线）
                if connector1.direction == 'straight':
                    if connector2.direction == 'right turn':
                        trj1 = np.array(connector1.boundaries[0])  # 直行轨迹
                    else:
                        trj1 = np.array(connector1.boundaries[1])  # 左转或其它方向轨迹
                else:
                    trj1 = np.array(connector1.centerline[0])  # 右转轨迹

                # 判断连接器2的方向，并选择适当的轨迹（边界或中心线）
                if connector2.direction == 'straight':
                    if connector1.direction == 'right turn':
                        trj2 = np.array(connector2.boundaries[0])  # 直行轨迹
                    else:
                        trj2 = np.array(connector2.boundaries[1])  # 左转或其它方向轨迹
                else:
                    trj2 = np.array(connector2.centerline[0])  # 右转轨迹

                # 使用InteractionSolver求解交叉点
                solver = InteractionSolver()
                cross_point = solver.GetCrossPoint(trj1, trj2)  # 获取交叉点
                if cross_point != []:
                    s.append(solver.GetSPosition(trj1, cross_point[0]))  # 获取交叉点的相对位置
                    self.overlap[connector1.id].append(connector2.id)  # 记录连接器间的重叠关系
            # 对重叠连接器按相对位置进行排序
            sorted_connector = sorted(zip(self.overlap[connector1.id], s), key=lambda x: x[1])
            self.overlap[connector1.id] = [x[0] for x in sorted_connector]  # 更新排序后的重叠连接器列表

    def ReadRoadData(self, file_path):
        file_path.encode('utf-8')       # 文件路径编码
        file = open(file_path)          # 打开文件
        file_reader = csv.reader(file)  # 创建CSV读取器
        for data_record in file_reader:
            yield Field(data_record)    # 按行返回字段数据

    # 读取Section数据的函数
    def ReadSectionData(self, file_path):
        # 编码文件路径为utf-8
        file_path.encode('utf-8')
        # 打开文件
        file = open(file_path)
        # 使用csv.reader读取文件内容
        file_reader = csv.reader(file)
        # 遍历文件中的每一行数据
        for data_record in file_reader:
            # 返回一个SectionSetting对象（使用yield生成器）
            yield SectionSetting(data_record)

    # 设置Section数据的函数
    def SetSection(self, data_file):
        # 遍历读取到的Section数据
        for f in self.ReadSectionData(data_file):
            # 创建一个Section对象
            section = Section()
            # 设置Section数据
            section.SetData(f)
            # 为每个车道初始化出生间隔（假设车道数量为lane_number）
            for lane_id in range(section.lane_number):
                section.born_interval.append(10.0)
            # 将Section对象添加到section_list字典中，以section.id为键
            self.section_list[section.id] = section

    # 读取Connector数据的函数
    def ReadConnectorData(self, file_path):
        # 编码文件路径为utf-8
        file_path.encode('utf-8')
        # 打开文件
        file = open(file_path)
        # 使用csv.reader读取文件内容
        file_reader = csv.reader(file)
        # 遍历文件中的每一行数据
        for data_record in file_reader:
            # 返回一个ConnectorSetting对象（使用yield生成器）
            yield ConnectorSetting(data_record)

    # 设置Connector数据的函数
    def SetConnector(self, file_path):
        # 遍历读取到的Connector数据
        for f in self.ReadConnectorData(file_path):
            # 创建一个Connector对象
            connector = Connector()
            # 设置Connector数据
            connector.SetData(f)
            # 将Connector的start_section添加到对应的section的connector_id_list中
            self.section_list[connector.start_section].connector_id_list.append(connector.id)
            # 更新connector的start_section和end_section对象
            connector.start_section = self.section_list[connector.start_section]
            connector.end_section = self.section_list[connector.end_section]
            # 设置连接器的边界
            connector.SetBoundary()
            # 设置连接器的中心线
            connector.SetCenterLine(connector.start_lane_id[0], connector.end_lane_id[0])
            # 将Connector对象添加到connector_list字典中，以connector.id为键
            self.connector_list[f.connector_id] = connector

    # 读取Signal数据的函数
    def ReadSignalData(self, file_path):
        # 编码文件路径为utf-8
        file_path.encode('utf-8')
        # 打开文件
        file = open(file_path)
        # 使用csv.reader读取文件内容
        file_reader = csv.reader(file)
        # 遍历文件中的每一行数据
        for data_record in file_reader:
            # 返回一个SignalSetting对象（使用yield生成器）
            yield SignalSetting(data_record)

    # 设置Signal数据的函数
    def SetSignal(self, file_path):
        # 遍历读取到的Signal数据
        for f in self.ReadSignalData(file_path):
            # 创建一个Signal对象
            signal = Signal()
            # 设置Signal数据
            signal.SetData(f)
            # 将Signal对象添加到signal_list字典中，以signal.id为键
            self.signal_list[f.id] = signal

    # 读取Signal到Connector的映射数据的函数
    def ReadSignalToConnector(self, file_path):
        # 编码文件路径为utf-8
        file_path.encode('utf-8')
        # 打开文件
        file = open(file_path)
        # 使用csv.reader读取文件内容
        file_reader = csv.reader(file)
        # 遍历文件中的每一行数据
        for data_record in file_reader:
            # 返回一个SignalToConnectorSetting对象（使用yield生成器）
            yield SignalToConnectorSetting(data_record)

    # 设置Signal到Connector的映射数据的函数
    def SetSignalToConnector(self, file_path):
        # 遍历读取到的Signal到Connector的映射数据
        for f in self.ReadSignalToConnector(file_path):
            # 将Signal与Connector建立关系（connector_id -> signal_id）
            self.signal_to_lane[f.connector_id] = f.signal_id
            # 将Signal与对应的Connector ID进行绑定
            self.signal_list[f.signal_id].connector_id.append(f.connector_id)
            # 设置Connector的Signal ID
            self.connector_list[f.connector_id].signal_id = f.signal_id

    # 绘制所有连接器（Connectors）的函数
    def DrawConnectors(self, ax):
        # 遍历所有Connector对象
        for c in self.connector_list.values():
            # 绘制Connector的所有边界（boundary）
            for boundary in c.boundaries:
                # 使用点线样式绘制边界
                ax.plot(boundary[0], boundary[1], color='gray', linestyle=':', marker='>', markersize=0.1)

    # 绘制绿色信号的连接器（Connectors）的函数
    def DrawGreenConnectors(self, ax):
        # 遍历所有Connector对象
        for c in self.connector_list.values():
            # 如果信号是绿色的
            if self.signal_list[c.signal_id].color == 'green':
                # 绘制绿色信号的连接器边界
                for boundary in c.boundaries:
                    # 每隔200个点绘制一个绿色圆点
                    ax.plot(boundary[0][0::200], boundary[1][0::200], 'go', markersize=0.5)

    def graph2road(self, graph, shp_flag):
        # 遍历图中的每一个连接点（link）
        for link in graph.link_map.values():
            conn = Connector()  # 创建一个新的连接器对象
            conn.id = link.id  # 设置连接器的ID为当前link的ID

            # 如果link的junction_id为-1，说明它是直行连接
            if link.junction_id == -1:
                conn.direction = 'straight'
            else:
                # 如果有交叉口，计算起始角度和结束角度之间的夹角
                end_yaw = link.lane_lst[0].direct[-1]
                start_yaw = link.lane_lst[0].direct[0]
                rela_yaw = (end_yaw - start_yaw) % (2 * math.pi)

                # 根据夹角的大小决定方向
                if rela_yaw < math.pi / 4.0 or rela_yaw > math.pi * 7 / 4.0:
                    conn.direction = 'straight'  # 直行
                elif rela_yaw >= math.pi / 4.0 and rela_yaw < math.pi:
                    conn.direction = 'left turn'  # 左转
                else:
                    conn.direction = 'right turn'  # 右转
                    # 如果是连接器，还需要补充一段下游的section
                    # outlink = link.out_link_lst[0]
                    # 这部分代码被注释掉了，暂时没有使用

            # 处理入链路的情况
            if link.in_link_lst:
                # 获取所有入链路
                link_lst = link.in_link_lst + [link]
                try:
                    # 获取每个link的车道数量
                    lanenum_lst = [len(x.lane_lst) for x in link_lst]
                except:
                    a = 1  # 这里没有进行任何操作，可能是为了防止错误

                # 获取最大车道数
                section_lanenum = max(lanenum_lst)
                if len(link.lane_lst) == section_lanenum:
                    conn.start_lane_id = [x + 1 for x in range(0, section_lanenum)]
                    # 设置车道起始ID
                    for lane in link.lane_lst:
                        lane.start_lane_id = lane.index
                else:
                    try:
                        # 找到车道数最多的link
                        linkx = link_lst[lanenum_lst.index(section_lanenum)]
                    except:
                        a = 1  # 这里同样没有进行操作
                    for lane in link.lane_lst:
                        for inlane in lane.in_lane_lst:
                            if inlane in linkx.lane_lst:
                                # 设置每条车道的起始车道ID
                                conn.start_lane_id.append(inlane.index)
                                lane.start_lane_id = inlane.index
                                break
                # 对车道ID进行排序
                conn.start_lane_id.sort()

            # 处理出链路的情况
            if link.out_link_lst:
                link_lst = link.out_link_lst + [link]
                lanenum_lst = [len(x.lane_lst) for x in link_lst]
                section_lanenum = max(lanenum_lst)
                if len(link.lane_lst) == section_lanenum:
                    conn.end_lane_id = [x + 1 for x in range(0, section_lanenum)]
                    for lane in link.lane_lst:
                        lane.end_lane_id = lane.index
                else:
                    linkx = link_lst[lanenum_lst.index(section_lanenum)]
                    for lane in link.lane_lst:
                        for outlane in lane.out_lane_lst:
                            if outlane in linkx.lane_lst:
                                conn.end_lane_id.append(outlane.index)
                                lane.end_lane_id = outlane.index
                                break
                conn.end_lane_id.sort()

            # 添加连接器到连接器列表
            self.add_conn(conn)

            # 创建一个新的section
            section = Section()
            fromlink = link.in_link_lst

            # 如果没有入链路，跳过
            if len(fromlink) == 0:
                continue

            if len(fromlink) > 1:  # 从多个link连接到1个link
                tolink = link
                section.conn_id_list_from = [link.id for link in fromlink]
                section.id = sum(section.conn_id_list_from, tolink.id)
                # 如果该section已经存在，跳过
                if section.id in self.section_list.keys():
                    continue
                section.lane_number = len(tolink.lane_lst)
                section.connector_id_list.append(tolink.id)
                objlink = tolink
                objrank = 0
            else:
                fromlink = fromlink[0]
                tolinks = fromlink.out_link_lst
                if len(tolinks) == 0:
                    continue
                if len(tolinks) == 1:  # 从1个link连接到1个link
                    tolink = tolinks[0]
                    section.id = fromlink.id + tolink.id
                    if section.id in self.section_list.keys():
                        continue
                    section.lane_number = max(len(fromlink.lane_lst), len(tolink.lane_lst))
                    section.conn_id_list_from = [fromlink.id]
                    section.connector_id_list.append(tolink.id)
                else:  # 从1个link连接到多个link
                    section.connector_id_list = [link.id for link in tolinks]
                    section.id = sum(section.connector_id_list, fromlink.id)
                    if section.id in self.section_list.keys():
                        continue
                    section.conn_id_list_from = [fromlink.id]
                    section.lane_number = len(fromlink.lane_lst)
                objlink = fromlink
                objrank = -1

            # 如果fromlink的ID是413或415，做特殊处理
            if objlink.id == 413 or objlink.id == 415:
                a = 1

            # 获取section的结束位置并添加到section列表
            section.end_list = self.get_section_pos(objlink, objrank, shp_flag)
            self.add_sec(section)

        # 更新每个section的连接关系
        for section in self.section_list.values():
            try:
                # 处理section的入链路
                for fromlink_id in section.conn_id_list_from:
                    fromlink = self.connector_list[fromlink_id]
                    # fromlink.end_section = section  # 注释掉了没有使用的代码

                    # 处理section的结束位置
                    section_id = section.connector_id_list[0]
                    if section_id not in self.section_list.keys():
                        new_section = Section()
                        new_section.id = section_id
                        new_section.lane_number = len(graph.get_link(new_section.id).lane_lst)
                        s_threshold = 4
                        new_direct = graph.get_link(new_section.id).lane_lst[0].direct[0]
                        pos1 = section.end_list[0]
                        pos2 = section.end_list[1]
                        new_p1 = Position()
                        new_p2 = Position()
                        new_p1.yaw = new_direct
                        new_p1.x = pos1.x + s_threshold * math.cos(new_direct)
                        new_p1.y = pos1.y + s_threshold * math.sin(new_direct)
                        new_p2.yaw = new_direct
                        new_p2.x = pos2.x + s_threshold * math.cos(new_direct)
                        new_p2.y = pos2.y + s_threshold * math.sin(new_direct)
                        new_section.end_list = [new_p1, new_p2]
                    else:
                        new_section = self.section_list[section_id]

                    fromlink.end_section = new_section

                # 处理section的出链路
                for tolink_id in section.connector_id_list:
                    tolink = self.connector_list[tolink_id]
                    tolink.start_section = section
            except:
                pass  # 某些section没有conn_id_list_from/connector_id_list，跳过

        # 最后清理不完整的连接器
        for link_id in list(self.connector_list):
            link = self.connector_list[link_id]
            if link.end_section is None or link.start_section is None:
                self.connector_list.pop(link_id)
            else:
                link.SetBoundary()  # 设置边界
                link.SetCenterLine(link.start_lane_id[0], link.end_lane_id[0])  # 设置中心线

    def add_conn(self, conn):
        # 添加连接对象，如果该连接id已存在，则抛出异常
        if conn.id not in self.connector_list.keys():
            self.connector_list[conn.id] = conn
        else:
            raise Exception("Connection is existed ?")  # 连接已存在，抛出异常

    def add_sec(self, section):
        # 添加路段对象，如果该路段id已存在，则抛出异常
        if section.id not in self.section_list.keys():
            self.section_list[section.id] = section
        else:
            raise Exception("Section is existed ?")  # 路段已存在，抛出异常

    def get_section_pos(self, objlink, objrank, shp_flag):
        # 获取指定车道位置
        if shp_flag == 1:  # shp道路车道是从外侧往内侧编号, 而xodr是从内往外编号
            p1 = Position()  # 创建位置对象p1
            p2 = Position()  # 创建位置对象p2

            # 获取第一个车道的坐标和车道宽度
            p1_lanex = objlink.lane_lst[0].xy[0][objrank]
            p1_laney = objlink.lane_lst[0].xy[1][objrank]
            p1_w = objlink.lane_lst[0].width

            # 如果车道宽度数据不存在，尝试计算宽度，若计算失败，默认宽度为3.5
            if not p1_w:
                if len(objlink.lane_lst) > 1:  # 如果有多个车道数据，尝试用前一个车道数据来推算
                    p0_lanex = objlink.lane_lst[1].xy[0][objrank]
                    p0_laney = objlink.lane_lst[1].xy[1][objrank]
                    p1_w = math.sqrt((p1_lanex - p0_lanex) ** 2 + (p1_laney - p0_laney) ** 2)
                else:
                    p1_w = 3.5  # 如果没有前车道数据，默认车道宽度为3.5

            # 计算第一个车道的yaw角度并根据车道宽度计算位置
            p1.yaw = objlink.lane_lst[0].direct[min(len(objlink.lane_lst[0].direct) - 1, objrank)]
            p1.x = p1_lanex + math.cos(p1.yaw - math.pi / 2.0) * p1_w / 2.0
            p1.y = p1_laney + math.sin(p1.yaw - math.pi / 2.0) * p1_w / 2.0

            # 获取最后一个车道的坐标和车道宽度
            p2_lanex = objlink.lane_lst[-1].xy[0][objrank]
            p2_laney = objlink.lane_lst[-1].xy[1][objrank]
            p2_w = objlink.lane_lst[-1].width

            # 如果车道宽度数据不存在，尝试计算宽度，若计算失败，默认宽度为3.5
            if not p2_w:
                if len(objlink.lane_lst) > 1:  # 如果有多个车道数据，尝试用前一个车道数据来推算
                    p0_lanex = objlink.lane_lst[-2].xy[0][objrank]
                    p0_laney = objlink.lane_lst[-2].xy[1][objrank]
                    p2_w = math.sqrt((p2_lanex - p0_lanex) ** 2 + (p2_laney - p0_laney) ** 2)
                else:
                    p2_w = 3.5  # 如果没有前车道数据，默认车道宽度为3.5

            # 计算最后一个车道的yaw角度并根据车道宽度计算位置
            p2.yaw = objlink.lane_lst[-1].direct[min(len(objlink.lane_lst[0].direct) - 1, objrank)]
            p2.x = p2_lanex + math.cos(p2.yaw + math.pi / 2.0) * p2_w / 2.0
            p2.y = p2_laney + math.sin(p2.yaw + math.pi / 2.0) * p2_w / 2.0

            return [p1, p2]  # 返回车道的两个位置对象p1和p2
        else:
            p1 = Position()  # 创建位置对象p1
            p2 = Position()  # 创建位置对象p2

            # 获取最后一个车道的坐标和车道宽度
            p1_lanex = objlink.lane_lst[-1].xy[0][objrank]
            p1_laney = objlink.lane_lst[-1].xy[1][objrank]
            p1_w = objlink.lane_lst[-1].width

            # 如果车道宽度数据不存在，尝试计算宽度，若计算失败，默认宽度为3.5
            if not p1_w:
                if len(objlink.lane_lst) > 1:  # 如果有多个车道数据，尝试用前一个车道数据来推算
                    p0_lanex = objlink.lane_lst[-2].xy[0][objrank]
                    p0_laney = objlink.lane_lst[-2].xy[1][objrank]
                    p1_w = math.sqrt((p1_lanex - p0_lanex) ** 2 + (p1_laney - p0_laney) ** 2)
                else:
                    p1_w = 3.5  # 如果没有前车道数据，默认车道宽度为3.5

            # 计算最后一个车道的yaw角度并根据车道宽度计算位置
            p1.yaw = objlink.lane_lst[0].direct[min(len(objlink.lane_lst[0].direct) - 1, objrank)]
            p1.x = p1_lanex + math.cos(p1.yaw - math.pi / 2.0) * p1_w / 2.0
            p1.y = p1_laney + math.sin(p1.yaw - math.pi / 2.0) * p1_w / 2.0

            # 获取第一个车道的坐标和车道宽度
            p2_lanex = objlink.lane_lst[0].xy[0][objrank]
            p2_laney = objlink.lane_lst[0].xy[1][objrank]
            p2_w = objlink.lane_lst[0].width

            # 如果车道宽度数据不存在，尝试计算宽度，若计算失败，默认宽度为3.5
            if not p2_w:
                if len(objlink.lane_lst) > 1:  # 如果有多个车道数据，尝试用前一个车道数据来推算
                    p0_lanex = objlink.lane_lst[1].xy[0][objrank]
                    p0_laney = objlink.lane_lst[1].xy[1][objrank]
                    p2_w = math.sqrt((p2_lanex - p0_lanex) ** 2 + (p2_laney - p0_laney) ** 2)
                else:
                    p2_w = 3.5  # 如果没有前车道数据，默认车道宽度为3.5

            # 计算第一个车道的yaw角度并根据车道宽度计算位置
            p2.yaw = objlink.lane_lst[-1].direct[min(len(objlink.lane_lst[0].direct) - 1, objrank)]
            p2.x = p2_lanex + math.cos(p2.yaw + math.pi / 2.0) * p2_w / 2.0
            p2.y = p2_laney + math.sin(p2.yaw + math.pi / 2.0) * p2_w / 2.0

            return [p1, p2]  # 返回车道的两个位置对象p1和p2
