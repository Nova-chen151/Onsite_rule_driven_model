#coding=utf-8
import random
import math
import numpy as np
from roads import D1xodrRoads as Roads
from functools import reduce
import pandas as pd
import random


class Vehicle():
    def __init__(self, graph, vehicles, veh_type, plot_vehicle, veh_params, sim_step, all_path_list):
        # 初始化车辆对象，传入图形、车辆属性、车辆类型、绘图对象、车辆参数等
        self.graph = graph
        self.id = vehicles[1]       # 车辆ID
        self.length = vehicles[-4]  # 车辆长度
        self.width = vehicles[-3]   # 车辆宽度
        # 车辆的净车距（加上一些随机扰动）
        self.net_dist = veh_params.net_dist + random.uniform(-0.1, 0.1)
        # 车辆的反应时间（加上一些随机扰动）
        self.reac_time = veh_params.reac_time + random.uniform(-0.3, 0.3)
        # 车辆的期望速度（加上一些随机扰动）
        self.desired_speed = veh_params.desired_speed + random.uniform(-3, 3)
        # 车辆的最大加速度（加上一些随机扰动）
        self.max_acc = veh_params.max_acc + random.uniform(-0.5, 0.5)
        # 车辆的舒适减速度（加上一些随机扰动）
        self.comfort_dec = veh_params.comfort_dec + random.uniform(-0.5, 0.5)
        self.acc = 0  # 初始加速度为零

        # 通过车辆位置数据获得当前所在的车道和位置
        [current_lane_index, self.position] = graph.worldxy2_link_lanepos(vehicles[2], vehicles[3])
        self.current_lane = self.graph.get_lane(current_lane_index)  # 获取当前车道对象
        self.current_link = self.current_lane.ownner  # 获取当前路段对象

        path_id_lst = []  # 初始化路径ID列表
        # 筛选出包含当前路段ID的路径
        valid_paths = [path for path in all_path_list if self.current_link.id in path]
        if valid_paths:
            # 随机选择一个有效路径
            path_id_lst = random.choice(valid_paths)

        self.speed = vehicles[5]  # 初始化车辆的当前速度
        self.next_lane = None  # 初始化目标车道为空
        self.path_id = path_id_lst  # 路径ID列表
        self.path = [graph.get_link(i) for i in self.path_id]  # 获取路径对应的路段对象列表

        # 筛选出可用车道列表
        in_lane_lst = [lane for lane in self.current_link.lane_lst]

        # 检查是否有其他车辆在当前车道中并影响可用车道
        for car in graph.vehicles.values():
            if car.current_link.id != self.current_link.id:
                continue
            if car.id != 0 and car.current_lane in in_lane_lst and car.position - car.length < 0 + self.length + self.net_dist:
                in_lane_lst.remove(car.current_lane)

        # 如果没有可用车道，车辆进入死亡状态
        if not in_lane_lst:
            self.status = 0  # 状态为0：死亡
            return

        # 获取当前车辆在世界坐标系中的位置和朝向
        [self.world_x, self.world_y, self.heading] = Roads.lanepos2_worldxy(self.current_lane, self.position)

        # 初始化车辆的一些标志和时间变量
        self.path_order = 0                 # 路径顺序
        self.lc_flag = 0                    # 换道标志
        self.lc_phase = 0                   # 换道阶段
        self.motiv_time = 0                 # 动机时间
        self.static_time = 0                # 静态时间
        self.lc_traj = []                   # 换道轨迹
        self.type = veh_type                # 车辆类型
        self.status = 1                     # 状态为1：正常
        self.sim_step = sim_step            # 仿真步长
        self.plot_vehicle = plot_vehicle    # 绘图对象
        # 初始化计时器
        self.timee0 = 0
        self.timee1 = 0
        self.timee2 = 0
        self.timee3 = 0
        self.times0 = 0
        self.times1 = 0
        self.times2 = 0
        self.times3 = 0

        # 将车辆对象添加到图形中的车辆字典
        graph.vehicles[self.id] = self

    @property
    def on_link(self):
        # 判断车辆是否在普通路段还是虚拟连接段上
        return self.current_link.junction_id == -1 and 1 or 0

    ########################################
    # 用于一维和二维模型转换的绘图方法
    def draw(self):
        # 根据车辆的位置、尺寸和朝向获取车辆的矩形轮廓
        p = Roads.get_rect(self.world_x, self.world_y, self.width, self.length, self.heading)
        # 提取轮廓点的x和y坐标
        xdata = [e[0] for e in p]
        ydata = [e[1] for e in p]
        # 使用绘图对象设置车辆的显示数据
        self.plot_vehicle.set_data(xdata, ydata)

    ########################################

    def IDM(self, front_speed, gap):
        # IDM跟驰模型，计算车辆的加速度
        # 输入变量：front_speed（前车速度），gap（前车尾部与本车车头之间的车距）
        net_dist = self.net_dist        # 净车距
        v = max(0, self.speed)          # 本车速度
        fv = front_speed                # 前车速度
        dv = self.desired_speed         # 期望速度
        rt = self.reac_time             # 反应时间
        ma = self.max_acc               # 最大加速度
        cd = self.comfort_dec           # 舒适减速度

        if gap < net_dist:
            # 如果车距小于净车距，立即减速
            acc = (0 - self.speed) / self.sim_step
        else:
            # 计算期望的跟车距离，并根据车距调整加速度
            desired_dist = net_dist + 2 * np.math.sqrt(v / dv) + rt * v + v * (v - fv) / 2.0 / np.math.sqrt(ma * cd)
            acc = ma * (1 - (v / dv) ** 4 - (desired_dist / gap) ** 2)

        return acc

    def follow_in_lc(self):
        # 换道影响模型：车辆在换道阶段2时，要对目标车道的前车进行跟驰
        if self.lc_phase == 2:
            if self.lc_flag == 1:
                front_speed = self.leftfront.speed
                gap = self.leftfront.gap
            else:
                front_speed = self.rightfront.speed
                gap = self.rightfront.gap
            vehicle_vir_acc = self.IDM(front_speed, gap)
            self.acc = min(self.acc, vehicle_vir_acc)

    def merge_follow(self):
        # 合流下的跟驰模型
        if self.merge_front.is_movable():
            fv = self.merge_front.speed  # 合流前车的速度
            fd = self.merge_front.gap  # 合流前车的车距
            if self.merge_front.dist > 10:
                # 如果与合流前车距离大于10米，则加速到接近最大加速度
                self.acc = min(self.acc, max(self.comfort_dec, (5 - self.speed) / self.sim_step))
            else:
                if self.merge_front.gap < self.merge_front.vehicle.length:
                    # 如果合流前车车距小于其车长，减速至停车状态
                    self.acc = min(self.acc, self.IDM(0, 0))
                else:
                    self.acc = min(self.IDM(fv, fd), self.acc)

    def diverge_follow(self):
        # 分流下的跟驰模型
        if self.diverge_front.is_movable() and self.rest_length() < 20:
            fv = self.diverge_front.speed  # 分流前车的速度
            fd = self.diverge_front.gap  # 分流前车的车距
            self.acc = min(self.IDM(fv, fd), self.acc)

    def follow_lc_veh(self):  # 对应于文档中的6.1.7换道影响模型，处于换道阶段2的车辆还要对目标车道后车产生影响
        # 判断左侧前车是否可移动并且处于换道阶段2且换道标志为2
        if self.leftfront.is_movable() and self.leftfront.vehicle.lc_phase == 2 and self.leftfront.vehicle.lc_flag == 2:
            # 使用IDM模型计算车辆的加速度（基于左侧前车的速度和车距）
            follow_lc_acc = self.IDM(self.leftfront.speed, self.leftfront.gap - 2)
            self.acc = min(self.acc, follow_lc_acc)  # 保证加速度不超过计算的最大值

        # 判断右侧前车是否可移动并且处于换道阶段2且换道标志为1
        if self.rightfront.is_movable() and self.rightfront.vehicle.lc_phase == 2 and self.rightfront.vehicle.lc_flag == 1:
            # 使用IDM模型计算车辆的加速度（基于右侧前车的速度和车距）
            follow_lc_acc = self.IDM(self.rightfront.speed, self.rightfront.gap - 2)
            self.acc = min(self.acc, follow_lc_acc)  # 保证加速度不超过计算的最大值

    def signal_reaction(self):  # 对应于文档中的6.1.6信号灯反应模型
        # 如果当前车道不是期望车道列表中的一部分，直接返回
        if not self.current_lane in self.desired_lane_lst:
            return

        # 获取到信号灯的距离和是否存在信号灯
        [light, dist_to_signal] = self.dist2_signal(self.current_lane, self.path_order, self.rest_length())

        # 如果存在信号灯并且距离小于100米
        if light is not None and dist_to_signal < 100:
            # 如果是红灯且前车不存在，或者当前车是离红灯最近的一辆车，则减速
            if light.is_red():
                fv = 0  # 前车速度为0（假设前方停着的车）
                fd = dist_to_signal  # 距离信号灯的距离
                self.acc = min(self.acc, self.IDM(fv, fd))  # 使用IDM模型计算加速度

            # 如果是黄灯并且剩余时间不足以通过信号灯，则减速
            elif light.is_yellow():
                if light.remain_time * self.speed < self.current_lane.length - self.position:
                    fv = 0  # 前车速度为0
                    fd = dist_to_signal  # 距离信号灯的距离
                    self.acc = min(self.acc, self.IDM(fv, fd))  # 使用IDM模型计算加速度

    def dist2_signal(self, lane, path_order, sum_dist):  # 计算到信号灯的距离
        signal_view_dist = 100  # 信号灯视距（假设为100米）

        # 如果剩余行驶距离超过信号灯视距，返回当前路段和总距离
        if sum_dist > signal_view_dist:
            return [None, sum_dist]

        # 如果已经走到路径的末尾，返回信号灯视距
        if path_order == len(self.path):
            return [None, signal_view_dist]

        # 如果当前车道有信号灯，返回信号灯和当前行驶距离
        if lane.has_light():
            return [lane.light, sum_dist]

        # 获取当前车道到下一路段的连接车道
        next_lane_lst = self.graph.conn_lanes_of_nextlink(lane, self.path[path_order])

        # 如果路径顺序不在最后，筛选符合条件的车道
        if path_order < len(self.path) - 1:
            for lane0 in next_lane_lst[::-1]:
                if not self.graph.conn_lanes_of_nextlink(lane0, self.path[path_order + 1]):
                    next_lane_lst.remove(lane0)

        # 如果没有找到符合条件的连接车道，返回信号灯视距
        if not next_lane_lst:
            return [None, signal_view_dist]

        # 获取第一个连接车道，并递归调用dist2_signal方法继续查找
        one1 = next_lane_lst[0]
        sum_dist = sum_dist + one1.length
        if sum_dist > signal_view_dist:
            return [None, sum_dist]

        # 递归调用获取信号灯距离
        return self.dist2_signal(one1, path_order + 1, sum_dist)

    def find_nextlane(self):  # 对应于文档中的4.2，查找本车的下一车道，即根据路径确定下游将要走的车道
        # 如果已经是路径的最后一段，或者当前车道不在期望车道列表中，则返回None
        if self.at_last_link() or self.current_lane not in self.desired_lane_lst:
            self.next_lane = None
            return

        # 获取当前车道的所有出路车道
        nextlane_lst = []
        for lane in self.current_lane.out_lane_lst:
            if lane.ownner.id in self.path_id:
                nextlane_lst.append(lane)

        # 如果没有找到出路车道，说明当前车道不在期望车道中，且距离换道点较远，返回None
        if not nextlane_lst:
            self.next_lane = None
            return

        # 随机选择一个出路车道作为下一车道
        self.next_lane = nextlane_lst[random.randint(0, len(nextlane_lst) - 1)]

    def follow_by_mlc(self):  # 对应于文档中的6.1.5强制性换道条件下的跟驰
        # 获取需要换道的车道数（正负值）
        lanes2change0 = self.get_lanes2change()
        if lanes2change0 == 0:
            return

        # 根据换道方向选择后方的车速和车距
        if lanes2change0 < 0:
            subj_bv = self.rightbehind.speed  # 右侧后车的速度
            subj_fdist = self.rightfront.gap  # 右侧前车的车距
        else:
            subj_bv = self.leftbehind.speed  # 左侧后车的速度
            subj_fdist = self.leftfront.gap  # 左侧前车的车距

        # 获取本车前方的车距
        self_fdist = self.front.gap

        # 如果本车前方的车距大于后方车距，且后方车距大于10米，且后车速度接近零，返回
        if self_fdist > subj_fdist and subj_fdist > 10 and subj_bv < 0.1:
            return

        # 获取需要换道的车道数
        lanes2change = abs(lanes2change0)

        # 计算停车所需的距离（包括换道所需的额外距离）
        dist2stop = self.rest_length() - lanes2change * 30 - self.current_link.lane_lst.index(self.current_lane) * 20
        front_speed = 0  # 假设前车速度为0
        gap = dist2stop  # 停车所需的车距

        # 使用IDM模型计算加速度
        vehicle_mlc_acc = self.IDM(front_speed, gap)
        self.acc = min(self.acc, vehicle_mlc_acc)  # 更新加速度

    def cross_through(self):  # 对应于文档中的5.5/6.1.4，交叉口内的冲突避让模型
        # 判断当前车道或下一车道是否有交叉点
        if self.current_lane.has_cross_point():
            self_cross = self.current_lane.cross_lst  # 当前车道的交叉点列表
        elif self.next_lane is not None and self.next_lane.has_cross_point():
            self_cross = self.next_lane.cross_lst  # 下一车道的交叉点列表
        else:
            return  # 如果当前和下一车道都没有交叉点，直接返回

        cirit_hd = 2  # 临界时距，用于判断是否存在冲突
        cross_acc = self.max_acc  # 初始化加速度为最大加速度

        # 遍历所有交叉点，判断是否存在冲突车辆
        for cross_one in self_cross:
            if cross_one in self.current_lane.cross_lst:
                self_dist2_cross = cross_one.this_position - self.position - self.length / 2.0
            else:
                self_dist2_cross = cross_one.this_position + self.rest_length()  # 计算当前车辆到交叉点的距离

            if self_dist2_cross < 0 or self_dist2_cross > 50:
                continue  # 如果交叉点距离超出有效范围，跳过

            # 获取交叉点所在车道上的所有车辆
            v = self.graph.get_vehicles_in_lane(cross_one.cross_lane)
            if v:
                for one in v:
                    one_dist2_cross = cross_one.cross_position - one.position - one.length / 2.0  # 计算冲突车辆到交叉点的距离
                    self_hd = self_dist2_cross / max(self.speed, 0.01)  # 当前车辆的头时距
                    one_hd = one_dist2_cross / max(one.speed, 0.01)  # 冲突车辆的头时距

                    # 判断是否有车辆停在交叉点上，需要避让
                    if one_dist2_cross < self.width / 2.0 + 1 and one_dist2_cross > -one.length - self.width / 2.0 - 1:
                        if self_dist2_cross < one.width / 2.0 + 1 and self_dist2_cross > -self.length - one.width / 2.0 - 1 and self_dist2_cross < one_dist2_cross:
                            break  # 如果当前车辆已经接近冲突车辆，则停止并避让
                        cross_acc = min(cross_acc, self.IDM(0, self_dist2_cross - 3))  # 根据IDM模型计算加速度
                        break

                    # 判断车辆是否存在冲突并需要减速
                    if one_dist2_cross > 0 and one_dist2_cross < 50 and self_hd > one_hd and self_hd < one_hd + cirit_hd:
                        cross_acc = min(cross_acc, self.IDM(0, self_dist2_cross - 3))  # 根据IDM模型计算加速度
                        break
            elif not self.at_last_link():
                # 如果当前车道没有冲突点，往下游的车道查找潜在的冲突车辆
                for lane2 in cross_one.cross_lane.in_lane_lst:
                    if lane2 is self.next_lane:
                        continue  # 跳过当前车道
                    v2 = self.graph.get_vehicles_in_lane(lane2)
                    for one2 in v2:
                        one_dist2_cross = cross_one.cross_position + one2.current_lane.length - one2.position - one2.length / 2.0  # 计算后方车辆到交叉点的距离
                        self_hd = self_dist2_cross / max(self.speed, 0.01)
                        one_hd = one_dist2_cross / max(one2.speed, 0.01)

                        # 判断是否有冲突车辆需要避让
                        if one_dist2_cross > 0 and one_dist2_cross < 50 and self_hd > one_hd and self_hd < one_hd + cirit_hd:
                            cross_acc = min(cross_acc, self.IDM(0, self_dist2_cross - 3))  # 根据IDM模型计算加速度
                            break

        # 更新车辆加速度，保证不会超过计算的最大加速度
        self.acc = min(self.acc, cross_acc)

    def find_mergefront2nd(self):  # 查找第二个合并点前方的车辆
        merge_front = surr_vehicle(self.desired_speed)  # 初始化一个合并前车辆的对象
        dist2_merge = 50  # 初始化合并点距离为50米
        outlane = self.graph.conn_lanes_of_nextlink(self.next_lane, self.path[self.path_order + 2])  # 获取下一车道到下游路径的连接车道
        if not outlane:
            return merge_front  # 如果没有合适的连接车道，直接返回

        all_vehicles = []
        # 获取出路车道上所有车辆
        for inlane in outlane[0].in_lane_lst:
            if inlane is self.next_lane:
                continue  # 跳过当前车道
            v = self.graph.get_vehicles_in_lane(inlane)
            all_vehicles.extend(v)  # 将车辆添加到列表中

        # 遍历所有车辆，计算车辆到合并点的距离
        for v in all_vehicles:
            merge_dist = self.rest_length() + self.next_lane.length - v.rest_length()  # 计算当前车辆到合并点的距离
            if merge_dist > 0 and merge_dist < merge_front.dist:  # 如果合并距离小于已有的最短距离
                merge_gap = merge_dist - v.length  # 计算合并车距
                merge_front.update(v, merge_dist, merge_gap)  # 更新合并前车辆的状态
                dist2_merge = self.rest_length() + self.next_lane.length  # 更新合并点的距离
        merge_front.dist = dist2_merge  # 将merge dist更新为车辆距离合并点的距离
        return merge_front  # 返回合并前方的车辆信息

    def find_mergefront1st(self):
        merge_front = surr_vehicle(self.desired_speed)
        all_vehicles = []
        for inlane in self.next_lane.in_lane_lst:
            if inlane is self.current_lane:
                continue
            v = self.graph.get_vehicles_in_lane(inlane)
            all_vehicles.extend(v)

        for v in all_vehicles:
            merge_dist = self.rest_length() - v.rest_length()
            if merge_dist > 0 and merge_dist < merge_front.dist: #将距离合流点更近的车辆作为合流前车
                merge_gap = merge_dist - v.length
                merge_front.update(v, merge_dist, merge_gap)

        merge_front.dist = self.rest_length() #将merge dist更新为车辆距离冲突点距离
        return merge_front

    def find_mergefront(self):  # 对应于文档中的5.3合流下的跟驰模型
        merge_front = surr_vehicle(self.desired_speed)  # 初始化合流前车辆的对象
        merge_view_dist = 30  # 合流视距（用于判断合流前车辆的有效范围）

        # 如果当前车道已经到达最后一个连接点，或者车辆的剩余长度超过合流视距，或者下一车道为空，则直接返回
        if self.at_last_link() or self.rest_length() > merge_view_dist or self.next_lane is None:
            self.merge_front = merge_front  # 设置合流前车辆信息
            return

        # 如果当前车道有合流前车，则直接计算合流加速度
        merge_front = self.find_mergefront1st()
        # 如果没有合流前车，且当前路径还没有到达倒数第二个位置，则继续往下游查找
        if not merge_front.is_movable() and self.path_order < len(self.path) - 2:
            merge_front = self.find_mergefront2nd()  # 查找下游车道的合流前车
        self.merge_front = merge_front  # 更新合流前车辆信息

    def is_sim(self):  # 判断当前车辆是否在模拟状态
        return self.status == 1  # 如果状态为1，则表示当前为模拟状态

    def get_frontlink(self, k):  # 获取前方第k个link（路径点）
        if self.path_order + k < len(self.path):
            return self.path[self.path_order + k]  # 返回路径中的第k个连接点
        else:
            return None  # 如果超出路径范围，返回None

    def get_vehicles_on_front_link(self, graph_pre):  # 查找下游link上的车辆
        front_view_dist = 100  # 下游车辆的视距为100米
        # 获取当前车道前方三个link上的所有车辆
        front1_vehicles, front2_vehicles, front3_vehicles = graph_pre.get_vehicles_in_front_link(self.get_frontlink(0),
                                                                                                 self.get_frontlink(1),
                                                                                                 self.get_frontlink(2))

        # 遍历前方第一个link上的车辆，删除不符合条件的车辆
        for one in front1_vehicles[::-1]:
            if one.position <= self.position or not self.at_same_lane(one):  # 判断车辆是否在同一车道且位置是否在当前车辆前方
                front1_vehicles.remove(one)
        if front1_vehicles:  # 如果第一个link上仍然有符合条件的车辆，则返回这些车辆
            return front1_vehicles, [], []

        # 遍历前方第二个link上的车辆，删除不符合条件的车辆
        for one in front2_vehicles[::-1]:
            if one.current_lane not in self.current_lane.out_lane_lst or self.rest_length() + one.position > front_view_dist:
                front2_vehicles.remove(one)
        if front2_vehicles:  # 如果第二个link上仍然有符合条件的车辆，则返回这些车辆
            return front1_vehicles, front2_vehicles, []

        # 如果前两个link都没有合适的车辆，检查第三个link
        front_link1 = self.get_frontlink(1)
        if not front_link1 or front_link1.lane_lst[0].length > front_view_dist:  # 如果第三个link不存在或超出视距范围，直接返回
            return front1_vehicles, front2_vehicles, []

        # 遍历第三个link上的车辆，删除不符合条件的车辆
        for one in front3_vehicles[::-1]:
            lane_inbetween = self.graph.get_lane_inbetween(self.current_lane, one.current_lane)
            if lane_inbetween is None or self.rest_length() + front_link1.lane_lst[
                0].length + one.position > front_view_dist:
                front3_vehicles.remove(one)
        # 返回三个link上的车辆信息
        return front1_vehicles, front2_vehicles, front3_vehicles

    def find_front(self):  # 查找前车
        f1, f2, f3 = self.get_vehicles_on_front_link(self.graph)  # 根据道路拓扑找到前方车辆

        self.front = surr_vehicle(self.desired_speed)  # 初始化前方车辆的对象
        # TODO: 自动驾驶车辆的关系需要进一步修改
        # if abs(av.world_y - car.world_y) < car.width/2 and av.world_x > car.world_x:
        #     d = math.sqrt((av.world_x - car.world_x) ** 2 + (av.world_y - car.world_y) ** 2)
        #     self.front.update(av, d, d - car.length)

        if f1:  # 如果当前车道前方有车辆
            one = reduce(lambda x, y: x.position < y.position and x or y, f1)  # 找到前方最近的车辆
            if one.position - self.position < self.front.dist:  # 如果前车在有效距离范围内
                front_dist = one.position - self.position  # 计算前车与当前车的距离
                front_gap = front_dist - one.length  # 计算前车与当前车的间隙
                self.front.update(one, front_dist, front_gap)  # 更新前车信息
            return

        if f2:  # 如果第二个link有车辆
            lanes = {}  # 记录不同车道上最靠近的车辆
            for one in f2:
                if one.current_lane.id not in lanes:
                    lanes[one.current_lane.id] = surr_vehicle(self.desired_speed)

                if one.position < lanes[one.current_lane.id].dist:  # 如果当前车辆更靠近
                    front_dist = one.position + self.rest_length()  # 计算当前车辆与前车的距离
                    front_gap = front_dist - one.length  # 计算前车与当前车的间隙
                    lanes[one.current_lane.id].update(one, front_dist, front_gap)  # 更新最靠近的车辆信息
            # 找到距离当前车辆最近的车道上的前车
            self.front = reduce(lambda e1, e2: e1.dist > e2.dist and e1 or e2, lanes.values())
            return

        if f3:  # 如果第三个link上有车辆
            lanes = {}
            for one in f3:
                if one.current_lane.id not in lanes:
                    lanes[one.current_lane.id] = surr_vehicle(self.desired_speed)

                    # 计算当前车辆与第三个link上车辆之间的距离
                lane_inbetween = self.graph.get_lane_inbetween(self.current_lane, one.current_lane)
                if self.rest_length() + lane_inbetween.length + one.position < lanes[one.current_lane.id].dist:
                    front_dist = self.rest_length() + lane_inbetween.length + one.position  # 计算当前车辆与第三个link上车辆的距离
                    front_gap = front_dist - one.length  # 计算前车与当前车的间隙
                    lanes[one.current_lane.id].update(one, front_dist, front_gap)  # 更新最靠近的车辆信息
                # 找到距离当前车辆最近的车道上的前车
            self.front = reduce(lambda e1, e2: e1.dist > e2.dist and e1 or e2, lanes.values())
        return

    def find_nearbyveh(self):  # 对应于文档5.2，找周围车辆：左右车道前后车
        # 初始化四个方向的周围车辆：左前、左后、右前、右后
        rightfront = surr_vehicle(self.desired_speed)
        rightbehind = surr_vehicle()
        leftfront = surr_vehicle(self.desired_speed)
        leftbehind = surr_vehicle()

        # 获取当前车道左右相邻车道的所有车辆
        [lvehs, rvehs] = self.graph.get_vehicles_in_lanes(self.current_lane.llane, self.current_lane.rlane)

        # 检查当前车道的左侧车道，如果有车辆则进一步处理
        if self.current_lane.llane:
            for one in lvehs:
                # 如果左侧车辆在当前车辆前方，且距离小于左前车的最大可视距离
                if one.position > self.position and one.position - self.position < leftfront.dist:
                    front_dist = one.position - self.position  # 计算前车距离
                    front_gap = front_dist - one.length / 2.0 - self.length / 2.0  # 计算前车与当前车的间隙
                    leftfront.update(one, front_dist, front_gap)  # 更新左前车的信息
                # 如果左侧车辆在当前车辆后方，且距离小于左后车的最大可视距离
                if one.position < self.position and self.position - one.position < leftbehind.dist:
                    behind_dist = self.position - one.position  # 计算后车距离
                    behind_gap = behind_dist - self.length / 2.0 - one.length / 2.0  # 计算后车与当前车的间隙
                    leftbehind.update(one, behind_dist, behind_gap)  # 更新左后车的信息

        # 检查当前车道的右侧车道，如果有车辆则进一步处理
        if self.current_lane.rlane:
            for one in rvehs:
                # 如果右侧车辆在当前车辆前方，且距离小于右前车的最大可视距离
                if one.position > self.position and one.position - self.position < rightfront.dist:
                    front_dist = one.position - self.position  # 计算前车距离
                    front_gap = front_dist - one.length / 2.0 - self.length / 2.0  # 计算前车与当前车的间隙
                    rightfront.update(one, front_dist, front_gap)  # 更新右前车的信息
                # 如果右侧车辆在当前车辆后方，且距离小于右后车的最大可视距离
                if one.position < self.position and self.position - one.position < rightbehind.dist:
                    behind_dist = self.position - one.position  # 计算后车距离
                    behind_gap = behind_dist - self.length / 2.0 - one.length / 2.0  # 计算后车与当前车的间隙
                    rightbehind.update(one, behind_dist, behind_gap)  # 更新右后车的信息

        # 如果左侧车道没有前后车辆，则向上游/下游车道查找左后/前车
        if self.current_lane.llane:
            if not leftbehind.is_movable() and self.position < self.speed * 2:  # 如果左后车不可移动，且当前位置较靠前
                all_vehicles = []
                # 查找当前车道左侧车道的所有车辆
                for inlane in self.current_lane.llane.in_lane_lst:
                    v = self.graph.get_vehicles_in_lane(inlane)
                    all_vehicles.extend(v)
                for one in all_vehicles:
                    # 判断这些车辆是否在可视距离范围内，如果是则更新左后车信息
                    if one.current_lane.length - one.position + self.position < leftbehind.dist:
                        behind_dist = one.current_lane.length - one.position + self.position  # 计算后车与当前车的距离
                        behind_gap = behind_dist - self.length / 2.0 - one.length / 2.0  # 计算后车与当前车的间隙
                        leftbehind.update(one, behind_dist, behind_gap)

            # 如果左前车不可移动，且当前车还未到达路径的最后一段，且剩余长度小于当前速度的两倍，则查找下游车道的前车
            if not leftfront.is_movable() and not self.at_last_link() and self.rest_length() < self.speed * 2:
                all_vehicles = []
                lanes = {}
                # 查找当前路径的下游车道的所有车辆
                for outlane in self.graph.conn_lanes_of_nextlink(self.current_lane.llane,
                                                                 self.path[self.path_order + 1]):
                    v = self.graph.get_vehicles_in_lane(outlane)
                    all_vehicles.extend(v)
                    lanes[outlane.id] = surr_vehicle(self.desired_speed)
                # 遍历下游车道的所有车辆，找到距离当前车辆最近的前车并更新
                for one in all_vehicles:
                    if one.position + self.rest_length() < lanes[one.current_lane.id].dist:
                        front_dist = one.position + self.rest_length()
                        front_gap = front_dist - one.length / 2.0 - self.length / 2.0
                        lanes[one.current_lane.id].update(one, front_dist, front_gap)
                # 更新左前车信息
                front_dist = 0
                for one in lanes:
                    if lanes[one].dist > front_dist:
                        front_dist = lanes[one].dist
                        leftfront = lanes[one]

        # 同理，如果右侧车道没有前后车辆，则向上游/下游车道查找右后/前车
        if self.current_lane.rlane:
            if not rightbehind.is_movable() and self.position < self.speed * 2:
                all_vehicles = []
                for inlane in self.current_lane.rlane.in_lane_lst:
                    v = self.graph.get_vehicles_in_lane(inlane)
                    all_vehicles.extend(v)
                for one in all_vehicles:
                    if one.current_lane.length - one.position + self.position < rightbehind.dist:
                        behind_dist = one.current_lane.length - one.position + self.position
                        behind_gap = behind_dist - self.length / 2.0 - one.length / 2.0
                        rightbehind.update(one, behind_dist, behind_gap)

            if not rightfront.is_movable() and not self.at_last_link() and self.rest_length() < self.speed * 2:
                all_vehicles = []
                lanes = {}
                for outlane in self.graph.conn_lanes_of_nextlink(self.current_lane.rlane,
                                                                 self.path[self.path_order + 1]):
                    v = self.graph.get_vehicles_in_lane(outlane)
                    all_vehicles.extend(v)
                    lanes[outlane.id] = surr_vehicle(self.desired_speed)
                for one in all_vehicles:
                    if one.position + self.rest_length() < lanes[one.current_lane.id].dist:
                        front_dist = one.position + self.rest_length()
                        front_gap = front_dist - one.length / 2.0 - self.length / 2.0
                        lanes[one.current_lane.id].update(one, front_dist, front_gap)
                front_dist = 0
                for one in lanes:
                    if lanes[one].dist > front_dist:
                        front_dist = lanes[one].dist
                        rightfront = lanes[one]

        # 最终更新四个方向的车辆信息
        self.rightfront = rightfront
        self.rightbehind = rightbehind
        self.leftfront = leftfront
        self.leftbehind = leftbehind

    def find_divergefront(self):  # 找分流前车
        diverge_front = surr_vehicle(self.desired_speed)  # 初始化分流前车对象，使用目标速度
        if self.rest_length() < self.speed * 2 and len(
                self.current_lane.out_lane_lst) > 1:  # 判断当前车道剩余长度小于两倍车速，并且车道有多个出口
            diverge_front.dist = 5  # 设置分流前车的车尾距离车道起点的默认距离为 5
            all_vehicles = []  # 初始化一个空的车辆列表，用于存储所有车辆
            # 遍历当前车道的所有出口车道，获取这些车道上的所有车辆
            for outlane in self.current_lane.out_lane_lst:
                v = self.graph.get_vehicles_in_lane(outlane)  # 获取当前出口车道上的车辆
                all_vehicles.extend(v)  # 将获取的车辆添加到 all_vehicles 列表中
            # 遍历所有收集到的车辆，查找距离车道起点小于设定距离的车辆
            for one in all_vehicles:
                if one.position - one.length < diverge_front.dist:  # 如果车辆的尾部距离车道起点小于设定的分流前车距离
                    front_dist = one.position - one.length  # 获取前车的尾部距离车道起点
                    front_gap = self.rest_length()  # 当前车道剩余的长度作为前车的间距
                    diverge_front.update(one, front_dist, front_gap)  # 更新分流前车的信息
        self.diverge_front = diverge_front  # 将分流前车的最终信息赋值给实例的 diverge_front 属性

    def find_surr_vehs(self):
        # self.times0 = time.time()
        self.find_front()  # 查找本车道的前车，对应于文档 5.1
        # self.timee0 = time.time()
        # self.times1 = time.time()
        self.find_nearbyveh()  # 查找相邻车道的前后车，对应于文档 5.2
        # self.timee1 = time.time()
        # self.times2 = time.time()
        self.find_mergefront()  # 查找合流前车，对应于文档 5.3
        # self.timee2 = time.time()
        # self.times3 = time.time()
        self.find_divergefront()  # 查找分流前车，对应于文档 5.4
        # self.timee3 = time.time()
        # print('sum time: ', self.timee3 - self.times0, 'time0: ', self.timee0 - self.times0, ' time1: ', self.timee1 - self.times1, ' time2: ', self.timee2 - self.times2, ' time3: ', self.timee3 - self.times3)

    def rest_length(self):  # 当前车道剩余长度
        return self.current_lane.length - self.position  # 返回当前车道的剩余长度，即车道总长度减去车辆当前位置

    def at_same_lane(self, v):  # 判断当前车辆是否与目标车辆在同一车道
        return self.current_lane.id == v.current_lane.id  # 如果车辆所在车道的 id 与目标车辆所在车道的 id 相同，返回 True

    def at_last_link(self):  # 判断是否在本车路径中的最后一个路段
        if self.path_order == len(self.path) - 1:  # 如果路径索引等于路径总长度减一，说明在最后一个路段
            return True
        else:
            return False  # 否则不在最后一个路段

    def calc_link_sequence(self):  # 向路径下游找一段距离 mlc_view_dist，在该距离内车辆将要行驶的路段
        mlc_view_dist = max(400, 3 * self.speed + 10)  # 计算视距，最小为 400，或者为车辆速度的三倍加 10
        link_lst = []  # 存储将要经过的路段列表

        # 确保 path_order 是合法的索引
        assert 0 <= self.path_order < len(self.path), "Invalid path_order"  # 如果路径索引不合法，抛出异常

        order = self.path_order  # 当前路径索引
        link = self.current_link  # 当前路段
        link_lst.append(link)  # 将当前路段添加到路段列表
        mlc_view_dist -= (self.current_lane.length - self.position)  # 减去当前车道剩余长度

        # 遍历路径下游的路段，直到视距用尽
        while mlc_view_dist > 0:
            if order >= len(self.path) - 1:  # 如果已经遍历到路径的最后一个路段，退出
                break
            order += 1  # 移动到下一个路径索引
            link = self.path[order]  # 获取下一个路段
            mlc_view_dist -= link.lane_lst[0].length  # 减去下一个路段的长度
            link_lst.append(link)  # 将下一个路段添加到路段列表

        # 如果只有一个路段，并且当前路径不是最后一个路段，检查是否添加下一个路段
        if len(link_lst) == 1 and not self.at_last_link() and self.path_order + 1 < len(self.path):
            link_lst.append(self.path[self.path_order + 1])  # 如果路径顺序没有越界，添加下一个路段

        return link_lst  # 返回下游路段列表

    def get_pass_lane(self, link_lst):  # 遍历当前路段的车道，判断哪些车道的下游车道属于 link_lst
        if not link_lst:  # 如果传入的路段列表为空，抛出异常
            raise ()
        link_lst = link_lst[::-1]  # 逆序遍历传入的路段列表
        lanes = list(link_lst[0].iter_lane())  # 获取第一个路段的车道列表

        # 遍历后续的路段，获取与这些路段相关的车道
        while link_lst[1:]:
            prev_link = link_lst[1]  # 获取下一个路段
            lanes = self.graph.get_sub_lane_to_outlane(prev_link, lanes) or list(prev_link.iter_lane())  # 获取子车道
            link_lst = link_lst[1:]  # 更新路段列表，移除已处理的路段
        return lanes  # 返回所有相关车道

    def get_desiredlanes_by_network(self):  # 确定当前 link 的期望车道
        if self.on_link:  # 如果当前车辆在某个路段上
            links = self.calc_link_sequence()  # 计算车辆将要行驶的下游路段序列
            lane_lst = self.get_pass_lane(links)  # 获取这些路段的车道列表
            if not lane_lst:  # 如果没有找到合适的车道
                lane_lst = [self.current_lane]  # 保持当前车道
            self.desired_lane_lst = lane_lst  # 设置期望车道列表
        else:
            self.desired_lane_lst = [self.current_lane]  # 如果不在路段上，期望车道即为当前车道

    def get_lanes2change(self):  # 计算如果要到达期望车道，至少需要进行几次换道
        lanes2change = 0  # 初始化换道次数
        if self.current_lane in self.desired_lane_lst:  # 如果当前车道已经在期望车道列表中，则不需要换道
            return lanes2change

        # 尝试向左换道，逐步查找是否能到达期望车道
        lane = self.current_lane.llane  # 获取当前车道的左侧车道
        while lane:  # 只要还有左侧车道
            lanes2change += 1  # 计数换道次数
            if lane in self.desired_lane_lst:  # 如果找到了期望车道
                return lanes2change
            lane = lane.llane  # 继续向左查找

        # 如果左侧没有找到，尝试向右换道
        lanes2change = 0  # 重新初始化换道次数
        lane = self.current_lane.rlane  # 获取当前车道的右侧车道
        while lane:  # 只要还有右侧车道
            lanes2change -= 1  # 计数换道次数（负值表示向右换道）
            if lane in self.desired_lane_lst:  # 如果找到了期望车道
                return lanes2change
            lane = lane.rlane  # 继续向右查找

        print('NOT FOUND', lanes2change)  # 如果无法找到合适的换道路径，打印错误信息
        return 0  # 返回 0，表示无法换道到期望车道

    def get_mangap(self, lanes2change):  # 计算强制换道的最小安全间隙
        """
        计算强制换道所需的最小安全间隙，受车辆速度、相对速度、剩余距离等因素影响
        """
        gap_lf = 1 + 0.15 * 2.237 * max(0, self.speed - self.leftfront.speed) \
                 + 0.3 * 2.237 * min(0, self.speed - self.leftfront.speed) \
                 + 0.2 * 2.237 * self.speed \
                 + 0.1 * 2.237 * (1 - np.exp(
            -0.008 * (self.current_lane.length - self.position - 20 * (lanes2change - 1)))) + random.random()

        gap_rf = 1 + 0.15 * 2.237 * max(0, self.speed - self.rightfront.speed) \
                 + 0.3 * 2.237 * min(0, self.speed - self.rightfront.speed) \
                 + 0.2 * 2.237 * self.speed \
                 + 0.1 * 2.237 * (1 - np.exp(
            -0.008 * (self.current_lane.length - self.position - 20 * (lanes2change - 1)))) + random.random()

        gap_lb = 0.5 + 0.1 * 2.237 * max(0, self.leftbehind.speed - self.speed) \
                 + 0.35 * 2.237 * min(0, self.leftbehind.speed - self.speed) \
                 + 0.25 * 2.237 * self.leftbehind.speed \
                 + 0.1 * 2.237 * (1 - np.exp(
            -0.008 * (self.current_lane.length - self.position - 20 * (lanes2change - 1)))) + 1.5 * random.random()

        gap_rb = 0.5 + 0.1 * 2.237 * max(0, self.rightbehind.speed - self.speed) \
                 + 0.35 * 2.237 * min(0, self.rightbehind.speed - self.speed) \
                 + 0.25 * 2.237 * self.rightbehind.speed \
                 + 0.1 * 2.237 * (1 - np.exp(
            -0.008 * (self.current_lane.length - self.position - 20 * (lanes2change - 1)))) + 1.5 * random.random()

        # 确保间隙值不会小于 0
        gap_lf = max(0, gap_lf)
        gap_rf = max(0, gap_rf)
        gap_lb = max(0, gap_lb)
        gap_rb = max(0, gap_rb)

        return gap_lf, gap_rf, gap_lb, gap_rb  # 返回计算出的四个间隙

    def get_disgap(self):  # 计算自由换道的最小安全间隙
        """
        计算在非强制情况下，自由换道所需的最小安全间隙
        """
        dis_gap_lf = 1 + 0.2 * 2.237 * max(0, self.speed - self.leftfront.speed) \
                     + 0.35 * 2.237 * min(0, self.speed - self.leftfront.speed) \
                     + 0.25 * 2.237 * self.speed + random.random()

        dis_gap_rf = 1 + 0.2 * 2.237 * max(0, self.speed - self.rightfront.speed) \
                     + 0.35 * 2.237 * min(0, self.speed - self.rightfront.speed) \
                     + 0.25 * 2.237 * self.speed + random.random()

        dis_gap_lb = 1.5 + 0.15 * 2.237 * max(0, self.leftbehind.speed - self.speed) \
                     + 0.45 * 2.237 * min(0, self.leftbehind.speed - self.speed) \
                     + 0.30 * 2.237 * self.leftbehind.speed + 1.5 * random.random()

        dis_gap_rb = 1.5 + 0.15 * 2.237 * max(0, self.rightbehind.speed - self.speed) \
                     + 0.45 * 2.237 * min(0, self.rightbehind.speed - self.speed) \
                     + 0.30 * 2.237 * self.rightbehind.speed + 1.5 * random.random()

        # 确保间隙值不会小于 0
        dis_gap_lf = max(0, dis_gap_lf)
        dis_gap_rf = max(0, dis_gap_rf)
        dis_gap_lb = max(0, dis_gap_lb)
        dis_gap_rb = max(0, dis_gap_rb)

        return dis_gap_lf, dis_gap_rf, dis_gap_lb, dis_gap_rb  # 返回计算出的自由换道间隙

    def get_lc_direction(self):  # 确定换道方向
        """
        确定车辆应该向左换道、向右换道，还是无法换道
        """
        if self.current_lane.llane in self.desired_lane_lst and self.current_lane.rlane in self.desired_lane_lst:
            lc_direction = 0  # 左右都可以换道
        elif self.current_lane.rlane in self.desired_lane_lst:
            lc_direction = -1  # 只能向右换道
        elif self.current_lane.llane in self.desired_lane_lst:
            lc_direction = 1  # 只能向左换道
        else:
            lc_direction = 2  # 无法换道

        return lc_direction  # 返回换道方向

    def get_mlc_flag(self, lanes2change): #自由换道换道动机计算
        man_gap_lf, man_gap_rf, man_gap_lb, man_gap_rb = self.get_mangap(abs(lanes2change))
        # 分别针对leftfront（lf),rightfront(rf),leftbehind(lb),rightbehind(rb)
        lc_direction = np.sign(lanes2change)
        self.motiv_time += np.sign(lanes2change)
        # if abs(self.motiv_time) > 200:
        #     self.die('wait too long... motive time:', self.motiv_time)
        #     return
        # # if self.motiv_time == 0:#记录换道动机产生的时间，
        # # # #motiv_time<0为向右，每过1步长motiv_time -= 1
        # # # #motiv_time>0为向左，每过1步长，motiv_time += 1
        if self.current_lane.lmark == 'dashed' and lc_direction > 0 and self.leftfront.gap > man_gap_lf and self.leftbehind.gap > man_gap_lb:
            self.lc_flag = 1  # 向左换道
        elif self.current_lane.rmark == 'dashed' and lc_direction < 0 and self.rightfront.gap > man_gap_rf and self.rightbehind.gap > man_gap_rb:
            self.lc_flag = 2  # 向右换道

    def get_dlc_flag(self): #强制换道换道动机计算
        dis_gap_lf, dis_gap_rf, dis_gap_lb, dis_gap_rb = self.get_disgap()
        lc_direction = self.get_lc_direction()
        if lc_direction < 2:
            if self.front.is_movable():
                self.front.vehicle.find_front()
                front2_veh = self.front.vehicle.front# 前方排队
                if front2_veh.is_movable():
                    if self.front.speed <= self.speed and front2_veh.speed <= self.front.speed:
                        if self.current_lane.lmark == 'dashed' and (lc_direction >= 0 and self.leftfront.gap > self.front.gap + 2 * self.length
                                                                    and self.leftfront.gap > dis_gap_lf and self.leftbehind.gap > dis_gap_lb):
                            self.lc_flag = 1
                        elif self.current_lane.rmark == 'dashed' and (lc_direction <= 0 and self.rightfront.gap > self.front.gap + 2 * self.length
                                                                      and self.rightfront.gap > dis_gap_rf and self.rightbehind.gap > dis_gap_rb):
                            self.lc_flag = 2
                if (self.front.speed < self.desired_speed - 20 / 3.6 and self.front.dist < 3 * self.speed):  # 前方低速车
                    if self.current_lane.lmark == 'dashed' and lc_direction >= 0 and \
                            (self.leftfront.speed > self.front.speed + 10 / 3.6 and self.leftfront.gap > dis_gap_lf) and \
                            (self.leftbehind.gap > dis_gap_lb and self.leftfront.gap > self.front.gap + 1 * self.length):
                        self.lc_flag = 1
                    elif self.current_lane.rmark == 'dashed' and (lc_direction <= 0 and
                            self.rightfront.speed > self.front.speed + 10 / 3.6 and self.rightfront.gap > dis_gap_rf and \
                            self.rightbehind.gap > dis_gap_rb and self.rightfront.gap > self.front.gap + 1 * self.length):
                        self.lc_flag = 2
                if (self.front.length > 8 and self.front.gap < 2 * self.speed):  # 前方大型车
                    if self.current_lane.lmark == 'dashed' and (lc_direction >= 0 and self.leftfront.speed > self.front.speed and self.leftfront.gap > dis_gap_lf and \
                            self.leftbehind.gap > dis_gap_lb and self.leftfront.gap > self.front.gap + 1 * self.length):
                        self.lc_flag = 1
                    elif self.current_lane.rmark == 'dashed' and (lc_direction <= 0 and self.rightfront.speed > self.front.speed and self.rightfront.gap > dis_gap_rf and \
                          self.rightbehind.gap > dis_gap_rb and self.rightfront.gap > self.front.gap + 1 * self.length):
                        self.lc_flag = 2

    def get_adjacent_lane(self):
        if self.lc_flag == 1:
            return self.current_lane.llane
        elif self.lc_flag == 2:
            return self.current_lane.rlane

    def lc(self):#前提：前车不处于换道状态，本车位于link上，且距离link末端大于10m
        if (self.front.is_movable() and self.front.vehicle.lc_flag != 0) or (self.path_order == 0 and self.position < self.length): # or self.rest_length() < self.length * 2:
            return
        lanes2change = self.get_lanes2change()
            # MLC一共需要跨越lanes2change条车道
        #判断换道的关键间隙
        if lanes2change:#如果有强制换道需求
            # MLC一共需要跨越lanes2change条车道
            self.get_mlc_flag(lanes2change)

        else: #如果没有强制换道需求
            if not self.front.is_movable():
                return
            self.get_dlc_flag()
        if self.lc_flag != 0:
            self.adj_lane = self.get_adjacent_lane()
            self.lc_phase = 1
            self.motiv_time = 0
            self.get_lc_traj()

    def get_lc_traj(self):  # 换道轨迹计算
        lc_dist = self.cal_lc_dist()  # 计算换道所需的距离
        if lc_dist < self.length:  # 如果换道距离小于车辆长度，认为换道失败
            self.die()  # 车辆“死亡”表示换道无法进行
            self.clear_lc()  # 清除换道标志
            return
        end_position = self.position + lc_dist  # 计算换道结束的位置
        end_current_lane = self.adj_lane  # 获取相邻车道
        # 获取相邻车道的目标位置（x, y, heading）
        [end_x, end_y, end_heading] = Roads.lanepos2_worldxy(end_current_lane, end_position)
        if not end_x:  # 如果目标位置无效，表示换道失败
            self.die()
            return
        # 计算换道轨迹（贝塞尔曲线）
        self.lc_traj = self.BezierCurve(self.world_x, self.world_y, end_x, end_y, self.heading, end_heading)

    def cal_lc_dist_with_speed(self):
        # 根据当前车速计算换道轨迹的长度
        if self.speed < 20 / 3.6:  # 车速低于20 km/h
            lc_dist = 2 * self.length  # 换道距离为车辆长度的两倍
        elif self.speed < 30 / 3.6:  # 车速低于30 km/h
            lc_dist = 5 * self.length  # 换道距离为车辆长度的五倍
        elif self.speed < 40 / 3.6:  # 车速低于40 km/h
            lc_dist = 6 * self.length  # 换道距离为车辆长度的六倍
        else:  # 车速高于40 km/h
            lc_dist = 8 * self.length  # 换道距离为车辆长度的八倍
        return lc_dist  # 返回换道距离

    def cal_lc_dist(self):  # 计算换道距离
        lc_dist = self.cal_lc_dist_with_speed()  # 根据车速计算换道距离
        self.adj_lane = self.get_adjacent_lane()  # 获取相邻车道
        lanes2change = self.get_lanes2change()  # 获取需要换道的车道数
        if abs(lanes2change) > 1:  # 如果需要换道的车道数大于1
            # 如果车道剩余空间有限，计算换道所需的最大距离
            lc_dist = min(lc_dist, (self.adj_lane.length - 2 - self.position) / abs(lanes2change))
        else:
            # 如果需要换道的车道数为1，则使用车道剩余距离来限制换道距离
            lc_dist = min(lc_dist, self.adj_lane.length - 2 - self.position)
        return lc_dist  # 返回计算后的换道距离

    def get_adjacent_lane(self):
        # 获取当前车道的相邻车道（左侧或右侧）
        if self.lc_flag == 1:
            return self.current_lane.llane  # 返回左侧车道
        elif self.lc_flag == 2:
            return self.current_lane.rlane  # 返回右侧车道

    def BezierCurve(self, start_px, start_py, end_px, end_py, start_heading, end_heading):
        """
        根据起始和结束位置及角度，计算出一条贝塞尔曲线作为换道轨迹
        """
        fragment_count = 50  # 轨迹分段数
        t = np.linspace(0, 1, num=int(fragment_count))  # 生成均匀分布的t值
        x1 = start_px * 2.0 / 3 + end_px * 1.0 / 3  # 贝塞尔曲线控制点1的x坐标
        x2 = start_px * 1.0 / 3 + end_px * 2.0 / 3  # 贝塞尔曲线控制点2的x坐标
        y1 = start_py * 2.0 / 3 + end_py * 1.0 / 3  # 贝塞尔曲线控制点1的y坐标
        y2 = start_py * 1.0 / 3 + end_py * 2.0 / 3  # 贝塞尔曲线控制点2的y坐标
        # 计算贝塞尔曲线的控制点坐标
        p1_x = (y1 - start_py - math.tan(start_heading + math.pi / 2) * x1 + math.tan(start_heading) * start_px) / \
               (math.tan(start_heading) - math.tan(start_heading + math.pi / 2))
        p1_y = math.tan(start_heading) * (p1_x - start_px) + start_py
        p2_x = (y2 - end_py - math.tan(end_heading + math.pi / 2) * x2 + math.tan(end_heading) * end_px) / \
               (math.tan(end_heading) - math.tan(end_heading + math.pi / 2))
        p2_y = math.tan(end_heading) * (p2_x - end_px) + end_py
        # 计算贝塞尔曲线的坐标
        Bx = start_px * (1 - t) ** 3 + 3 * p1_x * t * (1 - t) ** 2 + 3 * p2_x * t ** 2 * (1 - t) + end_px * t ** 3
        By = start_py * (1 - t) ** 3 + 3 * p1_y * t * (1 - t) ** 2 + 3 * p2_y * t ** 2 * (1 - t) + end_py * t ** 3
        return (Bx.tolist(), By.tolist())  # 返回计算出的贝塞尔曲线的x和y坐标

    def normal_cf(self):  # 对应于文档 6.1.1 的常规换道策略
        self.acc = min(self.acc, self.IDM(self.front.speed, self.front.gap))  # 限制加速度

    def update(self):
        # 更新车辆状态，进行车辆行为决策和动作
        self.get_desiredlanes_by_network()  # 获取期望车道
        self.find_nextlane()  # 查找下一车道
        self.find_surr_vehs()  # 查找周围的车辆
        self.acc = self.max_acc  # 设置最大加速度

        self.normal_cf()  # 执行常规换道策略
        self.merge_follow()  # 执行合流跟随策略
        self.diverge_follow()  # 执行发散跟随策略
        if self.current_lane.has_cross_point():  # 如果当前车道有交叉点
            self.cross_through()  # 通过交叉点
        if not self.lc_flag:  # 如果没有进行换道
            self.follow_by_mlc()  # 进行基于多车道跟随的行为
        self.signal_reaction()  # 根据交通信号做出反应
        self.follow_in_lc()  # 在换道过程中保持跟随
        self.follow_lc_veh()  # 根据换道车辆进行跟随
        if self.lc_flag == 0 and self.on_link:  # 如果还没有换道且在链路上
            self.lc()  # 执行换道操作

        self.update_pos()  # 更新车辆位置

    def acc_limit(self):
        # 限制加速度，避免过大负加速度
        if self.acc < -8:  # 如果加速度小于 -8
            self.acc = -8  # 限制加速度为 -8

    def update_pos(self):
        # 限制加速度，防止过大的加速度值
        self.acc_limit()

        # 更新速度，根据加速度和仿真步长计算新的速度
        self.speed += self.sim_step * self.acc

        # 如果速度接近零，认为车辆停滞
        if self.speed < 1e-6:
            self.speed = 0  # 将速度设为零
            self.static_time += self.sim_step  # 增加静止时间
        else:
            self.static_time = 0  # 重置静止时间

        # 如果静止时间超过60秒，认为车辆已经停止，进入死亡状态
        if self.static_time > 60:
            self.die()

        # 根据当前速度更新位置
        self.position += self.sim_step * self.speed

        # 如果车辆位置超过当前车道的长度，尝试进入下一个车道
        if self.position > self.current_lane.length:
            if self.at_last_link():  # 如果已经到达最后一个链路
                self.die()  # 车辆死亡
                return
            if self.next_lane is None:
                a = 1  # 用于调试，表示下一车道为空
            self.position -= self.current_lane.length  # 计算到达下一车道的相对位置
            self.current_lane = self.next_lane  # 切换到下一个车道
            try:
                self.current_link = self.current_lane.ownner  # 更新当前链路
            except:
                a = 1  # 错误处理
            self.path_order += 1  # 更新路径顺序

        # 初始化变量，确保在任何情况下都有值
        x, y, heading = 0.0, 0.0, 0.0

        # 如果当前车道为空，表示车辆死亡
        if self.current_lane is None:
            self.die()
        else:
            # 获取当前车道的世界坐标
            pos = Roads.lanepos2_worldxy(self.current_lane, self.position)
            if pos is None or len(pos) < 3:  # 如果获取到的坐标无效
                # 可选: 记录日志或采取其他措施
                pass
            else:
                x, y, heading = pos  # 更新车辆的坐标和朝向

        # 如果车辆处于换道状态，更新换道过程中的坐标
        if self.lc_flag:
            self.update_lc_phasexy(x, y, heading)
        else:
            # 如果不处于换道状态，直接更新车辆的世界坐标和朝向
            self.world_x = x
            self.world_y = y
            self.heading = heading

    def die(self):
        # 车辆死亡，状态设为0
        self.status = 0

    def update_lc_phasexy(self, x, y, heading):
        # 车辆处于换道状态时，更新车辆坐标是在贝塞尔曲线上更新坐标点的
        [traj_rank, traj_x, traj_y, traj_heading] = self.get_closest_point(x, y, self.lc_traj)

        # 如果找不到换道过程中的轨迹点，表示换道已结束
        if traj_rank is None:
            if self.lc_phase != 3:  # 如果换道阶段不是3，说明换道发生错误
                self.die()  # 车辆死亡
                print('lc error, lc finished when lc phase is:', self.lc_phase)
                return
            self.clear_lc()  # 清除换道标志
            self.world_x = x  # 更新世界坐标
            self.world_y = y
            self.heading = heading
            return

        # 更新车辆的世界坐标和朝向
        self.world_x = traj_x
        self.world_y = traj_y
        self.heading = traj_heading

        # 根据当前轨迹点的顺序更新换道阶段
        if traj_rank < len(self.lc_traj[0]) / 5.0:
            new_lc_phase = 1
        elif traj_rank < len(self.lc_traj[0]) * 3 / 5.0:
            new_lc_phase = 2
        else:
            new_lc_phase = 3
        new_lc_phase = max(self.lc_phase, new_lc_phase)  # 更新换道阶段

        # 如果换道从第2阶段转到第3阶段，表示车辆已经完成换道，移到目标车道
        if self.lc_phase == 2 and new_lc_phase == 3:
            objective_lane = self.get_adjacent_lane()  # 获取目标车道
            if objective_lane is None:
                a = 1  # 用于调试，表示目标车道为空

            # 精细化车道坐标的处理
            if objective_lane.add_length[1] - objective_lane.add_length[0] > 0.2:  # 路网坐标精细化
                [lane_xy, lane_direct, lane_add_length] = Roads.detail_xy(objective_lane.xy)
            else:
                lane_add_length = objective_lane.add_length
                lane_xy = objective_lane.xy
                lane_direct = objective_lane.direct

            # 更新换道过程中车辆在目标车道上的位置
            new_addlen = lane_add_length + [self.position]
            new_addlen.sort()
            new_idx = new_addlen.index(self.position)
            idxs = max(0, new_idx - 50)  # 获取一定范围内的车道坐标
            idxe = min(new_idx + 50, len(lane_add_length) - 1)

            # 计算车辆在目标车道上的虚拟位置
            [sublane_rank, sub_point] = self.get_cross_segment(traj_x, traj_y, lane_direct[idxs:idxe],
                                                               [lane_xy[0][idxs:idxe], lane_xy[1][idxs:idxe]])
            sublane_rank += idxs  # 更新车道段序号

            # 更新车辆的虚拟位置
            vehicle_vir_position = lane_add_length[sublane_rank] + \
                                   math.sqrt((sub_point[0] - lane_xy[0][sublane_rank]) ** 2 + (
                                               sub_point[1] - lane_xy[1][sublane_rank]) ** 2)
            self.current_lane = objective_lane  # 更新当前车道
            self.position = vehicle_vir_position  # 更新车辆位置
        self.lc_phase = new_lc_phase  # 更新换道阶段

    # 计算经过某一点(x0, y0)且角度为angle的直线与多段线xy_lst的交点
    def get_cross_segment(self, x0, y0, angle, xy_lst):
        temp_len = 10  # 用来计算交点的一个临时长度
        for i in range(len(xy_lst[0]) - 1):  # 遍历多段线中的每一段
            # 如果angle是一个列表，则选取每段对应的角度，否则直接使用单一角度
            if isinstance(angle, list) == 0:
                temp_angle = angle
            else:
                temp_angle = angle[i]

            # 计算当前点附近的两条线段，分别为直线与当前段线段
            inter_point = is_intersect(x0 + temp_len * math.cos(temp_angle - math.pi / 2),
                                       y0 + temp_len * math.sin(temp_angle - math.pi / 2),
                                       x0 + temp_len * math.cos(temp_angle + math.pi / 2),
                                       y0 + temp_len * math.sin(temp_angle + math.pi / 2),
                                       xy_lst[0][i], xy_lst[1][i], xy_lst[0][i + 1], xy_lst[1][i + 1])

            # 如果找到了交点，返回交点所在的段号及交点
            if inter_point != []:
                return [i, inter_point]

            # 如果遍历到最后一段线时还未找到交点，尝试使用最接近的点进行计算
            if i == len(xy_lst[0]) - 2:
                # 如果第一次找不到交点，递归尝试获取最近的点
                [sublane_rank, sub_pointx, sub_pointy, sublane_heading] = self.get_closest_point(x0, y0, xy_lst)
                return [sublane_rank, [sub_pointx, sub_pointy]]

    # 找到xy_lst中距离(x0, y0)最近的点
    def get_closest_point(self, x0, y0, xy_lst):
        if len(xy_lst) < 2:
            print('xy_lst', xy_lst)  # 打印错误信息，如果xy_lst不符合要求
        # 计算(x0, y0)与xy_lst中所有点的距离
        dist = list(map(lambda dx, dy: np.sqrt((dx - x0) ** 2 + (dy - y0) ** 2), xy_lst[0], xy_lst[1]))

        # 找到最小的距离，并获取对应的索引
        min = reduce(lambda d1, d2: d1 < d2 and d1 or d2, dist)
        rank = dist.index(min)  # 得到最近点的索引

        pointx = xy_lst[0][rank]  # 获取最近点的x坐标
        pointy = xy_lst[1][rank]  # 获取最近点的y坐标

        # 如果最近点是xy_lst中的最后一个点，计算其与前一个点的角度（heading）
        if rank == len(xy_lst[0]) - 1:
            heading = math.atan((xy_lst[1][rank] - xy_lst[1][rank - 1]) / (xy_lst[0][rank] - xy_lst[0][rank - 1])) - \
                      math.pi * ((xy_lst[0][rank] - xy_lst[0][rank - 1]) < 0)
            return [None, pointx, pointy, heading]

        # 如果最近点不是最后一个，计算它与下一个点的角度
        heading = math.atan((xy_lst[1][rank + 1] - xy_lst[1][rank]) / (xy_lst[0][rank + 1] - xy_lst[0][rank])) - \
                  math.pi * ((xy_lst[0][rank + 1] - xy_lst[0][rank]) < 0)

        # 返回最近点的索引、坐标及角度
        return [rank, pointx, pointy, heading]

    # 清除车辆换道状态
    def clear_lc(self):
        self.lc_traj = []       # 清空换道轨迹
        self.lc_flag = 0        # 重置换道标志
        self.lc_phase = 0       # 重置换道阶段
        self.adj_lane = None    # 重置相邻车道

# 车辆类型类，用于定义不同类型的车辆（如小车、卡车等）的属性
class VehicleType:
    def __init__(self, veh_type):
        self.value = veh_type  # 车辆类型（'car' 或 'truck'）
        # 如果是小车
        if veh_type == 'car':
            self.length = 4.5  # 小车长度
            self.width = 2.0  # 小车宽度
            self.max_acc = 4.0  # 小车最大加速度
            self.desired_speed = random.randint(20, 30)  # 小车的期望速度（随机在20到30之间）
            self.comfort_dec = 2.0  # 小车的舒适减速度
        # 如果是卡车
        elif veh_type == 'truck':
            self.length = 10.5  # 卡车长度
            self.width = 2.6  # 卡车宽度
            self.max_acc = 2.5  # 卡车最大加速度
            self.desired_speed = random.randint(15, 20)  # 卡车的期望速度（随机在15到20之间）
            self.comfort_dec = 1.8  # 卡车的舒适减速度


# 周围车辆类，表示与当前车辆相关的周围车辆信息
class surr_vehicle():
    def __init__(self, speed=0, length=0):
        self.dist = 200             # 默认的距离
        self.gap = 200              # 默认的间隙
        self.vehicle = None         # 当前周围车辆的实例
        self.speed = speed          # 当前周围车辆的速度
        self.length = length        # 当前周围车辆的长度
        self.id = 0                 # 当前周围车辆的ID

    def is_movable(self):
        # 判断当前车辆是否存在
        return self.vehicle is not None

    def update(self, v, d, g):
        # 更新周围车辆的信息
        self.vehicle = v            # 周围车辆实例
        self.gap = g                # 周围车辆的间隙
        self.dist = d               # 周围车辆的距离
        self.speed = v.speed        # 周围车辆的速度
        self.length = v.length      # 周围车辆的长度
        self.id = self.vehicle.id   # 周围车辆的ID


# 判断两条线段是否相交，并返回交点
def is_intersect(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y):
    # 计算交点的几何参数
    m = (p2_x - p1_x) * (p3_y - p1_y) - (p3_x - p1_x) * (p2_y - p1_y)
    n = (p2_x - p1_x) * (p4_y - p1_y) - (p4_x - p1_x) * (p2_y - p1_y)
    p = (p4_x - p3_x) * (p1_y - p3_y) - (p1_x - p3_x) * (p4_y - p3_y)
    q = (p4_x - p3_x) * (p2_y - p3_y) - (p2_x - p3_x) * (p4_y - p3_y)

    # 判断两条线段是否相交
    if m * n <= 0 and p * q <= 0:
        if abs(p2_x - p1_x) < 1e-3:  # 如果第一条线段接近垂直
            x = p1_x
            y = (p4_y - p3_y) / (p4_x - p3_x) * (x - p3_x) + p3_y
        elif abs(p4_x - p3_x) < 1e-3:  # 如果第二条线段接近垂直
            x = p3_x
            y = (p2_y - p1_y) / (p2_x - p1_x) * (x - p1_x) + p1_y
        else:
            # 计算交点的坐标
            x = ((p4_y - p3_y) / (p4_x - p3_x) * p3_x - (p2_y - p1_y) / (p2_x - p1_x) * p1_x + p1_y - p3_y) / \
                ((p4_y - p3_y) / (p4_x - p3_x) - (p2_y - p1_y) / (p2_x - p1_x))
            y = (p2_y - p1_y) / (p2_x - p1_x) * (x - p1_x) + p1_y
        return [x, y]  # 返回交点的坐标
    else:
        return []  # 如果没有交点，返回空列表


# 找到离指定点(x, y)最近的车道
def find_closest_lane(x, y, lane_list):
    min_distance = float('inf')  # 初始化最小距离为无限大
    closest_lane = None  # 初始化最接近的车道为None
    for lane in lane_list:
        # 假设车道有x0, y0, heading属性，分别表示车道起点的坐标和车道的朝向
        dx = x - lane.x0
        dy = y - lane.y0
        # 计算点到车道的距离（直线到点的最短距离）
        distance = abs(dx * math.sin(lane.heading) - dy * math.cos(lane.heading))
        if distance < min_distance:
            min_distance = distance  # 更新最小距离
            closest_lane = lane  # 更新最接近的车道
    return closest_lane  # 返回最接近的车道


# 计算车辆在车道上的相对位置
def calculate_position(lane, x, y):
    # 计算车辆在车道上的相对位置，s表示车辆距离车道起点的相对位置
    s = (x - lane.x0) * math.cos(lane.heading) + (y - lane.y0) * math.sin(lane.heading)
    return s  # 返回相对位置




