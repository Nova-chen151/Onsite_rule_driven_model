#coding=utf-8
#Python2.7
import xml.dom.minidom
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import animation
from socket import *
from time import ctime
import struct
import copy
from collections import defaultdict
from functools import reduce
import csv

class VPath:  # 车辆路径类
    def __init__(self):
        self.oid = []  # 车辆路径的起点
        self.did = []  # 车辆路径的终点
        self.last_time = 0  # 路径的最后时间（例如：路径上车辆通过的最后时间）

class Signal:  # 信号灯类
    def __init__(self):
        self.id = 0             # 信号灯的ID
        self.laneidx_lst = []   # 信号灯控制的车道列表
        self.pos = 0            # 信号灯的位置
        self.timing = []        # 信号灯的定时列表（周期时间等）

    @property
    def clength(self):
        return len(self.timing)  # 返回信号灯定时周期的长度

class Light:  # 红绿灯类
    def __init__(self, light_id, light_pos):
        self.id = light_id      # 红绿灯的ID
        self.color = 0          # 红绿灯的颜色（0: 未知, 1: 红, 2: 黄, 3: 绿）
        self.remain_time = 0    # 剩余时间，指示灯当前的倒计时
        self.pos = light_pos    # 红绿灯的位置

    def is_red(self):
        return self.color == 1  # 判断是否为红灯

    def is_yellow(self):
        return self.color == 2  # 判断是否为黄灯

    def is_green(self):
        return self.color == 3  # 判断是否为绿灯

class Lane(object):  # 车道类
    def __init__(self):
        self.id = -1                # 车道ID
        self.link_id = -1           # 车道连接ID
        self.type = 'driving'       # 车道类型（默认为'驾驶'车道）
        self.width = []             # 车道宽度
        self.speed_limit = -1       # 车道限速
        self.llid = 0               # 左侧车道ID
        self.rlid = 0               # 右侧车道ID
        self.lmark = 'dashed'       # 左侧车道标志（默认为虚线）
        self.rmark = 'dashed'       # 右侧车道标志（默认为虚线）
        self.llane = None           # 左侧车道对象
        self.rlane = None           # 右侧车道对象
        self.in_lane_id_lst = []    # 入车道ID列表
        self.out_lane_id_lst = []   # 出车道ID列表
        self.out_lane_lst = []      # 出车道列表
        self.in_lane_lst = []       # 入车道列表
        self.xy = []                # 车道的坐标列表
        self.direct = []            # 车道的行驶方向
        self.add_length = []        # 车道的长度列表
        self.ownner = None          # 车道的所属对象（如交叉口等）
        self.light = None           # 车道是否有信号灯
        self.cross_lst = []         # 交叉点列表
        self.index = -1             # 车道索引

    @property
    def length(self):
        return self.add_length[-1]  # 返回车道的总长度（最后一个长度值）

    @property
    def index_id(self):
        return self.link_id * 100 + self.id  # 车道的唯一标识符（通过连接ID和车道ID计算）

    def cut_lane(self, position, up_len, down_len):
        sub_lane = Lane()                           # 创建一个子车道
        up_pos = position - up_len                  # 计算子车道的起始位置
        down_pos = position + down_len              # 计算子车道的结束位置
        add_uppos = [i for i in self.add_length]    # 创建车道长度的副本
        add_uppos.append(up_pos)                    # 加入起始位置
        add_downpos = [i for i in self.add_length]  # 创建车道长度的副本
        add_downpos.append(down_pos)                # 加入结束位置
        add_uppos.sort()                            # 对车道起始位置进行排序
        add_downpos.sort()                          # 对车道结束位置进行排序
        uppos_index = add_uppos.index(up_pos)       # 查找起始位置的索引
        downpos_index = add_downpos.index(down_pos) # 查找结束位置的索引
        sub_lane.add_length = self.add_length[max(0, uppos_index - 1):downpos_index]  # 截取子车道的长度
        x = self.xy[0][max(0, uppos_index - 1):downpos_index]  # 截取x坐标
        y = self.xy[1][max(0, uppos_index - 1):downpos_index]  # 截取y坐标
        sub_lane.xy = [x, y]                        # 设置子车道的坐标
        sub_lane.id = self.id                       # 设置子车道的ID
        sub_lane.link_id = self.link_id             # 设置子车道的连接ID
        return sub_lane

    def set_ownner(self, link):
        self.ownner = link  # 设置车道的所属链接对象

    def add_inlane(self, lane):
        if lane not in self.in_lane_lst:    # 如果车道不在入车道列表中
            self.in_lane_lst.append(lane)   # 将车道添加到入车道列表

    def add_outlane(self, lane):
        if lane not in self.out_lane_lst:   # 如果车道不在出车道列表中
            self.out_lane_lst.append(lane)  # 将车道添加到出车道列表

    def has_light(self):
        return self.light is not None  # 判断车道是否有信号灯

    def has_cross_point(self):
        return self.cross_lst  # 判断车道是否有交叉点

    def add_cross(self, this_offset, cross_offset, cross_lane, point):
        for c in self.cross_lst:  # 遍历交叉点列表
            if c.cross_lane is cross_lane:  # 如果已存在相同的交叉车道
                return
        cp = Cross(this_offset, cross_offset, cross_lane, point)  # 创建新的交叉点
        self.cross_lst.append(cp)  # 将交叉点添加到交叉点列表

    def is_driving_lane(self):
        # 判断是否为驾驶车道类型，允许行车的车道类型包括：'driving'、'special1'、'offRamp'、'onRamp'
        return self.type in ['driving', 'special1', 'offRamp', 'onRamp']

    def is_conn(self):
        return self.ownner.junction_id != -1  # 判断车道是否连接到交叉口

    def x(self):
        return self.xy[0]  # 返回车道的x坐标

    def y(self):
        return self.xy[1]  # 返回车道的y坐标

class Cross():
    def __init__(self, self_offset, cross_offset, cross_lane, point):
        self.this_position = self_offset        # 当前车道的位置偏移量
        self.cross_position = cross_offset      # 交叉点的位置偏移量
        self.cross_lane = cross_lane            # 交叉车道对象
        self.point = point                      # 交叉点的坐标或标记

class Link():
    def __init__(self):
        self.id = -1                # Link的唯一标识符
        self.junction_id = -1       # 连接所属的交叉口ID
        self.lane_lst = []          # 连接的车道列表
        self.in_link_lst = []       # 进入此连接的Link列表
        self.out_link_lst = []      # 离开此连接的Link列表

    def add_lane(self, lane):
        self.lane_lst.append(lane)  # 将车道添加到车道列表

    def iter_lane(self):
        for l in self.lane_lst:  # 遍历车道列表
            yield l  # 生成器逐个返回车道对象

    def add_inlink(self, link):
        if link not in self.in_link_lst:  # 如果Link不在输入Link列表中
            self.in_link_lst.append(link)  # 将该Link添加到输入链接列表

    def add_outlink(self, link):
        if link not in self.out_link_lst:  # 如果Link不在输出Link列表中
            self.out_link_lst.append(link)  # 将该Link添加到输出链接列表

# 根据车道中心线坐标计算行驶方向和线长度序列
def get_line_feature(xy):
    xy = np.array(xy)
    # n为中心点个数，2为x,y坐标值
    x_prior = xy[0][:-1]
    y_prior = xy[1][:-1]
    x_post = xy[0][1:]
    y_post = xy[1][1:]
    # 根据前后中心点坐标计算【行驶方向】
    dx = x_post - x_prior
    dy = y_post - y_prior
    direction = list(map(lambda d: d > 0 and d or d + 2 * np.pi, np.arctan2(dy, dx))) #沿x轴方向逆时针转过的角度

    length = np.sqrt(dx ** 2 + dy ** 2)
    for i in range(len(length) - 1):
        length[i + 1] += length[i]
    return direction, length.tolist()

class Graph:
    def __init__(self):
        self.link_map = {}          # 链接映射，存储所有链接信息
        self.lane_map = {}          # 车道映射，存储所有车道信息
        self.vehicles = {}          # 车辆信息，存储所有车辆对象
        self.light_map = {}         # 信号灯映射，存储所有信号灯信息
        self.intersection_map = {}  # 用于存储交叉口信息（仅在shp文件中使用）
        self.path_map = {}          # 路径映射，存储路径信息
        self.replace_linkmap = {}   # 替代链接映射，可能用于替代路径等操作
        self.replace_lanemap = {}   # 替代车道映射，可能用于替代车道等操作

    def add_link(self, link):
        """ 添加链接到图中，如果链接已经存在则抛出异常 """
        if link.id not in self.link_map:
            self.link_map[link.id] = link  # 将链接按id加入链接映射
        else:
            raise Exception("Link is existed ?")  # 如果链接已存在，则抛出异常

    def add_light(self, light):
        """ 添加信号灯到图中，如果信号灯已经存在则抛出异常 """
        if light.id not in self.light_map:
            self.light_map[light.id] = light  # 将信号灯按id加入信号灯映射
        else:
            raise Exception("Link is existed ?")  # 如果信号灯已存在，则抛出异常

    def add_lane(self, lane):
        """ 添加车道到图中，如果车道已经存在则抛出异常 """
        if lane.index_id not in self.lane_map:
            self.lane_map[lane.index_id] = lane  # 将车道按索引id加入车道映射
        else:
            raise Exception("Link is existed ?")  # 如果车道已存在，则抛出异常

    def get_lane_inbetween(self, lane1, lane2):
        """ 获取两个车道之间的车道（交汇处的车道） """
        if lane1 is None or lane2 is None:
            return None  # 如果车道为空，返回空值
        # 遍历lane1的出口车道和lane2的入口车道，寻找交集
        for one1 in lane1.out_lane_id_lst:
            for one2 in lane2.in_lane_id_lst:
                if one1 == one2:  # 找到交集
                    lane = self.get_lane(one1)  # 获取该车道
                    if lane is not None:
                        return lane  # 返回交汇车道
        return None  # 如果没有找到共同车道，返回空值

    def get_lane(self, lane_id):
        """ 根据车道id获取对应的车道 """
        return self.lane_map[lane_id]  # 返回对应车道

    def get_vehicles_in_link(self, link):
        """ 获取在指定链接中的所有车辆 """
        if link is None:
            return []  # 如果链接为空，返回空列表
        vehs = []
        # 遍历所有车辆，判断车辆当前所在的链接
        for veh in self.vehicles.values():
            if veh.current_link.id == link.id:  # 如果车辆当前在指定的链接上
                vehs.append(veh)  # 将该车辆加入返回列表
        return vehs

    def get_vehicles_in_front_link(self, link0, link1, link2):
        """ 获取三个前方链接上的车辆（根据链接顺序分类车辆） """
        links = [link0, link1, link2]  # 将三个链接放入列表
        link_ids = []
        for link in links:
            if link is None:
                link_ids.append(None)  # 如果链接为空，则插入None
            else:
                link_ids.append(link.id)  # 如果链接不为空，加入链接id
        # 为每个链接创建一个车辆列表
        vehs0 = []
        vehs1 = []
        vehs2 = []
        # 遍历所有车辆，根据车辆当前所在的链接分类
        for veh in self.vehicles.values():
            if veh.current_link.id in link_ids:  # 如果车辆在其中一个链接上
                link_index = link_ids.index(veh.current_link.id)  # 找到车辆所在的链接索引
                if link_index == 0:
                    vehs0.append(veh)  # 将车辆加入对应的列表
                elif link_index == 1:
                    vehs1.append(veh)
                elif link_index == 2:
                    vehs2.append(veh)
        return vehs0, vehs1, vehs2  # 返回三个列表，分别包含三个链接上的车辆

    # 获取左车道和右车道上的所有车辆
    def get_vehicles_in_lanes(self, llane, rlane):
        lvehs = []  # 左车道的车辆列表
        rvehs = []  # 右车道的车辆列表
        for veh in self.vehicles.values():  # 遍历所有车辆
            # 如果车辆当前在左车道并且左车道存在
            if llane is not None and veh.current_lane is not None and veh.current_lane.index_id == llane.index_id:
                lvehs.append(veh)  # 将车辆加入左车道列表
            # 如果车辆当前在右车道并且右车道存在
            if rlane is not None and veh.current_lane is not None and veh.current_lane.index_id == rlane.index_id:
                rvehs.append(veh)  # 将车辆加入右车道列表
        return [lvehs, rvehs]  # 返回左右车道上的车辆列表

    # 获取特定车道上的所有车辆
    def get_vehicles_in_lane(self, lane):
        if lane is None:  # 如果车道为空，返回空列表
            return []
        else:
            vehs = []  # 初始化车辆列表
            for veh in self.vehicles.values():  # 遍历所有车辆
                # 如果车辆当前车道的ID与目标车道的ID匹配
                if veh.current_lane is not None and veh.current_lane.index_id == lane.index_id:
                    vehs.append(veh)  # 将车辆加入车辆列表
            return vehs  # 返回车辆列表

    # 构建拓扑结构
    def build_topo(self):
        # 遍历所有的link
        for link in self.link_map.values():
            lane_id_lst = [l.id for l in link.lane_lst]  # 获取link上的所有车道ID
            # 划分左侧车道和右侧车道ID
            llane_id_lst = [l for l in lane_id_lst if l > 0]
            rlane_id_lst = [l for l in lane_id_lst if l < 0]
            # 遍历link中的每一条车道
            for lane in link.iter_lane():
                lane.set_ownner(link)  # 设置车道的owner为当前link
                # 如果车道ID属于左车道ID列表
                if lane.id in llane_id_lst:
                    if lane.id != min(llane_id_lst):  # 如果不是最小ID的车道，设置左车道ID
                        lane.llid = lane.id - 1
                    if lane.id != max(llane_id_lst):  # 如果不是最大ID的车道，设置右车道ID
                        lane.rlid = lane.id + 1
                # 如果车道ID属于右车道ID列表
                elif lane.id in rlane_id_lst:
                    if lane.id != min(rlane_id_lst):  # 如果不是最小ID的车道，设置右车道ID
                        lane.rlid = lane.id - 1
                    if lane.id != max(rlane_id_lst):  # 如果不是最大ID的车道，设置左车道ID
                        lane.llid = lane.id + 1

        # 再次遍历所有link，并更新车道信息
        for link in self.link_map.values():
            if link.id == -765:
                aaa = 1  # 用于调试，检测是否进入了该条件
            for lane in link.iter_lane():
                # 获取车道的左车道和右车道
                lane.llane = self.lane_map.get(lane.link_id * 100 + lane.llid, None)
                lane.rlane = self.lane_map.get(lane.link_id * 100 + lane.rlid, None)
                # 更新车道的出车道和入车道
                for lid in lane.out_lane_id_lst:
                    outlane = self.lane_map.get(lid)  # 获取出车道
                    if outlane is not None:
                        lane.out_lane_lst.append(outlane)  # 将出车道加入到车道的出车道列表
                    else:
                        lane.out_lane_id_lst.remove(lid)  # 如果出车道不存在，从ID列表中移除
                for lid in lane.in_lane_id_lst:
                    inlane = self.lane_map.get(lid)  # 获取入车道
                    if inlane is not None:
                        lane.in_lane_lst.append(inlane)  # 将入车道加入到车道的入车道列表
                    else:
                        lane.in_lane_id_lst.remove(lid)  # 如果入车道不存在，从ID列表中移除

                # 更新车道的入车道列表和出车道列表
                for in_lane_id in lane.in_lane_id_lst:
                    inlane = self.lane_map.get(in_lane_id)
                    if inlane is not None:
                        if inlane.ownner is None:
                            continue
                        lane.add_inlane(inlane)  # 将入车道添加到车道的入车道列表
                        link.add_inlink(inlane.ownner)  # 将入车道的ownner添加到link的入link列表
                for out_lane_id in lane.out_lane_id_lst[::-1]:
                    outlane = self.lane_map.get(out_lane_id)
                    if outlane is not None:
                        if outlane.ownner is None:  # 如果出车道没有ownner，则跳过
                            continue
                        lane.add_outlane(outlane)  # 将出车道添加到车道的出车道列表
                        link.add_outlink(outlane.ownner)  # 将出车道的ownner添加到link的出link列表
                    else:
                        lane_index = lane.out_lane_id_lst.index(out_lane_id)
                        lane.out_lane_id_lst.pop(lane_index)  # 移除无效的出车道ID

                # 更新lane_map和link_map
                self.lane_map[lane.index_id] = lane  # 更新车道映射
                self.link_map[lane.link_id] = link  # 更新link映射

        # 设置交通灯的控制信息
        for light in self.light_map.values():
            for lane_idx in light.laneidx_lst:
                lane = self.lane_map[lane_idx]
                lane.light = Light(light.id, light.pos)  # 将交通灯信息绑定到车道

    def worldxy2_link_lanepos(self, world_x, world_y):
        # 根据车道以及距离中心线起点长度，计算该点的二维坐标和方向角
        min_dists = []  # 存储所有车道与目标点之间的最小距离信息
        for link in self.link_map.values():  # 遍历所有链接
            for lane in link.lane_lst:  # 遍历每个车道
                dist = float('inf')  # 初始化最小距离为无限大
                coord_num = 0  # 初始化坐标计数器
                # 如果车道的附加长度差值小于0.2，则直接使用车道的xy坐标
                if lane.add_length[1] - lane.add_length[0] < 0.2:
                    lane_xy = lane.xy
                else:
                    [lane_xy, _, _] = detail_xy(lane.xy)  # 计算更精细的车道坐标
                # 尝试读取车道的起始位置
                try:
                    pos = lane.add_length[0]
                except:
                    a = 1  # 如果没有找到位置，则跳过
                min_pos = pos  # 初始化最小位置
                # 遍历车道的所有坐标点
                while coord_num < len(lane_xy[0]) - 1:
                    # 计算当前点到下一个点的距离
                    dist_interval = math.sqrt((lane_xy[0][coord_num + 1] - lane_xy[0][coord_num]) ** 2 + (
                            lane_xy[1][coord_num + 1] - lane_xy[1][coord_num]) ** 2)
                    # 计算目标点与当前点之间的距离
                    temp = (lane_xy[0][coord_num] - world_x) ** 2 + (lane_xy[1][coord_num] - world_y) ** 2
                    if dist > temp:  # 如果目标点距离当前点更近，则更新最小距离和最小位置
                        dist = temp
                        min_pos = pos
                    pos += dist_interval  # 更新车道位置
                    coord_num += 1  # 更新坐标计数器
                min_dists.append([dist, min_pos, lane])  # 保存当前车道的最小距离及其对应位置

        # 找到最小距离对应的车道
        min_d = reduce(lambda p1, p2: p1[0] < p2[0] and p1 or p2, min_dists)
        veh_lane = min_d[2].index_id  # 获取最小距离对应的车道ID
        veh_pos = min_d[1]  # 获取最小距离对应的位置
        return [veh_lane, veh_pos]  # 返回车道和位置

    def link_combine(self):
        # 合并具有相同属性的link，便于换道处理，相同属性定义：车道数相同、车道上下游一一对应、link有上下游关系，非连接器
        link_pairs = []  # 存储所有可配对的link对
        for link1 in self.link_map.values():  # 遍历所有链接
            if link1.junction_id != -1:  # 排除与交叉口相关的link
                continue
            for link2 in self.link_map.values():  # 遍历所有链接，与link1进行匹配
                if link1 == link2 or len(link1.lane_lst) != len(
                        link2.lane_lst) or link1.junction_id != link2.junction_id or link2.junction_id != -1 or link2 not in link1.out_link_lst:
                    continue  # 排除不满足条件的link
                lanes1 = link1.lane_lst  # 获取link1的车道列表
                lanes2 = link2.lane_lst  # 获取link2的车道列表
                lane_connect = 0  # 初始化连接计数器
                for i in range(0, len(lanes1)):  # 遍历车道列表
                    if lanes2[i] in lanes1[i].out_lane_lst and len(lanes1[i].out_lane_lst) == 1:  # link2是link1的下游
                        lane_connect += 1  # 计数连接成功的车道
                if lane_connect == len(lanes1):  # 如果所有车道都连接成功，则这两个link可以配对
                    link_pairs.append([link1, link2])  # 将配对的link加入列表

        link_series = self.pair2series(link_pairs, [])  # 将所有可配对的link序列化

        for link_set in link_series:  # 遍历每个link序列进行合并
            out_link_lst = link_set[-1].out_link_lst  # 获取最后一个link的下游链接
            for outlink in out_link_lst:
                outlink.in_link_lst.remove(link_set[-1])  # 移除最后一个link作为下游链接
                outlink.add_inlink(link_set[0])  # 将第一个link作为下游链接
            for lane in link_set[-1].lane_lst:  # 遍历最后一个link的车道
                out_lane_lst = lane.out_lane_lst
                for outlane in out_lane_lst:
                    outlane.in_lane_lst.remove(lane)  # 更新车道的上下游关系
                    lane_index = link_set[-1].lane_lst.index(lane)
                    outlane.in_lane_lst.append(link_set[0].lane_lst[lane_index])  # 添加新的车道
                    outlane.in_lane_id_lst = [x.id for x in outlane.in_lane_lst]  # 更新车道ID列表

            # 将最后一个link的下游链接给第一个link
            link_set[0].out_link_lst = link_set[-1].out_link_lst
            # 将所有link序列信息合并到第一个link上
            for lane in link_set[0].lane_lst:
                lane_index = link_set[0].lane_lst.index(lane)
                lane.out_lane_lst = link_set[-1].lane_lst[lane_index].out_lane_lst
                for link in link_set[1:]:  # 合并每个link的车道信息
                    lane.xy[0] = lane.xy[0] + link.lane_lst[lane_index].xy[0][1:]
                    lane.xy[1] = lane.xy[1] + link.lane_lst[lane_index].xy[1][1:]
                [lane.direct, lane.add_length] = get_lane_feature(lane.xy)  # 重新计算车道的特征

        # 删除合并后的无用路段，存储link替换数据
        for link_set in link_series:
            for link in link_set[1:][::-1]:  # 从后往前删除合并后的link
                if link.id not in self.replace_linkmap.keys():  # 确保该link未被替换
                    self.replace_linkmap[link.id] = link_set[0].id  # 记录替换数据
                    lane_num = len(link.lane_lst)  # 获取车道数
                    for i in range(lane_num):
                        self.replace_lanemap[link.lane_lst[i].id] = link_set[0].lane_lst[i].id  # 记录车道替换数据
                    self.link_map.pop(link.id)  # 删除无用的link
                else:
                    pass  # 如果link已经存在替换映射，则跳过

    # pair2series方法用于将一组lane对（pairs）转换为多个子序列（series），并合并有连接关系的lane对
    def pair2series(self, pairs, series):
        # 如果pairs中只有一个元素，直接将其放入series中并返回
        if len(pairs) == 1:
            series.append(pairs[0])
            return series

        link_pair1 = pairs[0]  # 取出pairs中的第一个元素，作为当前处理的link_pair
        pair_count2 = 0  # 用于记录link_pair1是否有下游链接的计数
        pair_count0 = 0  # 用于记录link_pair1是否有上游链接的计数
        pair_append2 = []  # 用于存储下游链的索引
        pair_append0 = []  # 用于存储上游链的索引

        # 查找下游链接，检查当前link_pair1的末尾元素是否与其他pair的开头元素匹配
        for link_pair2 in pairs[1:]:
            if link_pair1[-1] == link_pair2[0]:
                pair_count2 += 1
                pair_append2.append(pairs.index(link_pair2))

        # 如果只有一个下游link，则将下游pair合并到link_pair1中，并删除pairs中的下游pair
        if pair_count2 == 1:
            [link_pair1.append(x) for x in pairs[pair_append2[0]][1:]]
            pairs.pop(pair_append2[0])

        # 查找上游链接，检查当前link_pair1的开头元素是否与其他pair的末尾元素匹配
        for link_pair0 in pairs[1:]:
            if link_pair1[0] == link_pair0[-1]:
                pair_count0 += 1
                pair_append0.append(pairs.index(link_pair0))

        # 如果只有一个上游link，则将上游pair合并到link_pair1中，并删除pairs中的上游pair
        if pair_count0 == 1:
            [pairs[pair_append0[0]].append(x) for x in link_pair1[1:]]
            pairs.remove(link_pair1)

        # 如果上下游都没有可以合并的链接，则将link_pair1作为单独的子序列放入series中
        if pair_count0 != 1 and pair_count2 != 1:
            series.append(link_pair1)
            pairs.remove(link_pair1)

        # 如果pairs中还有未处理的元素，递归调用pair2series方法
        if pairs:
            return self.pair2series(pairs, series)
        else:
            return series

    # build_cross方法用于计算所有车道交叉点
    def build_cross(self):
        for lane in self.lane_map.values():
            # 只有当lane是一个有效的行车道并且可以连接时，才计算交叉点
            if lane.is_driving_lane() and lane.is_conn():
                self.calc_lane_cross_point(lane)
        # 保存计算得到的交叉点
        self.save_cross_point()

    # calc_lane_cross_point方法用于计算给定车道（lane）与其他车道的交叉点
    def calc_lane_cross_point(self, lane):
        near_set = set()  # 存储与当前lane相连的其他车道
        for l in self.lane_map.values():
            # 找到与当前lane相连且是行车道的其他车道
            if l.is_driving_lane() and l.is_conn() and l is not lane:
                near_set.add(l)

        # 对于每一条相连的车道，检查是否在同一个交叉口，并且不是输入或输出方向上的相同车道
        for cross_lane in near_set:
            if cross_lane.ownner.junction_id == lane.ownner.junction_id and \
                    cross_lane.in_lane_lst[0] is not lane.in_lane_lst[0] and cross_lane.out_lane_lst[0] is not \
                    lane.out_lane_lst[0]:
                calculated = 0  # 标记是否已经计算过交叉点
                for cross in lane.cross_lst:
                    # 如果已经计算过交叉点，就跳过
                    if cross_lane is cross.cross_lane:
                        calculated = 1
                        break
                if not calculated:
                    # 如果未计算过交叉点，调用calc_cross_point方法进行计算
                    self.calc_cross_point(lane, cross_lane)

    # calc_cross_point方法用于计算两个车道（la和lb）的交叉点
    def calc_cross_point(self, la, lb):
        # 跳过开始和结束的位置（边缘位置）
        nmd = []
        for n in range(1, len(la.x()) - 1):  # 对车道la的所有位置进行遍历（跳过边缘）
            for m in range(1, len(lb.x()) - 1):  # 对车道lb的所有位置进行遍历（跳过边缘）
                # 计算两点之间的距离
                d = (la.x()[n] - lb.x()[m]) ** 2 + (la.y()[n] - lb.y()[m]) ** 2
                nmd.append((n, m, d))

        # 找到距离最小的点，即交叉点
        res = reduce(lambda p1, p2: p1[2] < p2[2] and p1 or p2, nmd)

        # 如果最小距离小于0.2，认为两条车道交叉
        if res[2] < 0.2:
            n, m = res[0], res[1]  # 获取交叉点的索引
            la_offset = la.add_length[n]
            lb_offset = lb.add_length[m]

            # 将交叉点信息添加到车道la和lb中
            la.add_cross(la_offset, lb_offset, lb, (la.x()[n], la.y()[n]))
            lb.add_cross(lb_offset, la_offset, la, (la.x()[n], la.y()[n]))

    # load_cross_point方法用于从文件加载交叉点信息
    def load_cross_point(self, filename):
        # 读取CSV文件中的交叉点数据
        for l in read_csv(filename):
            tid = int(l[0])  # 车道id1
            cid = int(l[1])  # 车道id2
            cp = float(l[2])  # 交叉点位置参数1
            tp = float(l[3])  # 交叉点位置参数2
            x = float(l[4])  # 交叉点的x坐标
            y = float(l[5])  # 交叉点的y坐标

            # 获取车道对象
            tlane = tid in self.lane_map and self.lane_map[tid]
            clane = cid in self.lane_map and self.lane_map[cid]

            # 如果车道无效，则跳过该条数据
            if not tlane or not clane:
                continue

            # 将交叉点信息添加到相应的车道中
            tlane.add_cross(tp, cp, clane, (x, y))

    # 保存交叉点信息
    def save_cross_point(self):
        # 打开文件以写入交叉点数据
        with open('lane_cross', 'w') as f:
            # 遍历每一条车道
            for lane in self.lane_map.values():
                # 如果车道不是行驶车道或车道没有连接，则跳过
                if not lane.is_driving_lane or not lane.is_conn():
                    continue
                # 遍历每个交叉点
                for cross in lane.cross_lst:
                    # 写入车道索引，交叉点车道索引，交叉点位置，当前位置，以及点坐标
                    line = '%d,%d,%.5f,%.5f,%.5f,%.5f\n' % (
                    lane.index_id, cross.cross_lane.index_id, cross.cross_position, cross.this_position, cross.point[0],
                    cross.point[1])
                    f.write(line)

    # 获取从给定车道到目标车道的子车道
    def get_sub_lane_to_outlane(self, link, dest_lane_lst):
        ret_lanes = []  # 用于存储符合条件的车道
        all_lanes = list(link.iter_lane())  # 获取所有车道
        # 遍历每一条车道
        for l in all_lanes:
            # 如果车道的出车道列表与目标车道列表有交集
            if len(set(l.out_lane_lst).intersection(dest_lane_lst)) > 0:
                ret_lanes.append(l)
        return ret_lanes

    # 获取从原点到目标的路径
    def get_path(self, origin, destination):
        paths = [[origin]]  # 用于存储路径的队列，初始化时路径包含原点
        visited_links = set([origin])  # 已经访问过的路径集合
        path_length = 0  # 当前路径长度
        max_path_length = 20  # 最大路径长度，防止死循环

        # 当路径队列不为空且路径长度没有超出最大值时
        while paths and path_length < max_path_length:
            next_paths = []  # 用于存储下一轮生成的路径
            # 遍历当前路径列表
            for path in paths:
                last_link = path[-1]  # 当前路径的最后一个车道
                current_link = self.get_link(last_link)  # 获取该车道的连接信息

                # 检查当前连接是否有效或没有出车道
                if not current_link or not current_link.out_link_lst:
                    continue

                # 遍历当前车道的所有出车道
                for out_link in current_link.out_link_lst:
                    # 如果该车道已经访问过，则跳过
                    if out_link.id in visited_links:
                        continue
                    new_path = path + [out_link.id]  # 构建新的路径
                    # 如果到达目标车道，返回路径
                    if out_link.id == destination:
                        return new_path
                    # 否则，继续扩展路径
                    next_paths.append(new_path)
                    visited_links.add(out_link.id)

            # 更新路径列表，继续遍历
            paths = next_paths
            path_length += 1

        # 如果没有找到路径，返回部分路径或直接连接目标
        closest_path = paths[0] if paths else [origin]
        closest_path.append(destination)  # 假设直接连接目标
        return closest_path

    # 根据原点和目标获取路径
    def find_path(self, origin, destination):
        # 遍历所有已存储的路径，查找匹配的路径
        for path in self.path_map.values():
            if path.oid == origin and path.did == destination:
                return path.path_id_lst

        # 如果没有找到路径，抛出异常
        raise Exception("Invalid path ?")

    # 根据路径集创建路径
    def create_path(self, path_set):  # 生成 graph.path_map
        for path in path_set:
            # 如果路径中的路段发生过替换，进行替换
            if path[0] in self.replace_linkmap.keys():
                path[0] = self.replace_linkmap[path[0]]
            if path[1] in self.replace_linkmap.keys():
                path[1] = self.replace_linkmap[path[1]]
            # 添加路径
            self.add_path(path[0], path[1], path[2])

    # 添加路径信息到路径映射中
    def add_path(self, oid, did, flow):
        oid = int(oid)
        did = int(did)
        flow = int(flow)
        key = oid * did  # 计算路径的唯一键

        # 如果路径已经存在，则删除旧路径
        if key in self.path_map:
            del self.path_map[key]

        # 添加新路径
        path = VPath()  # 创建新的路径对象
        self.path_map[key] = path  # 更新路径映射
        path.oid = oid
        path.did = did
        path.path_id_lst = self.get_path(oid, did)  # 获取路径ID列表

    # 更新车辆和信号配时信息
    def update_veh(self, car):  # 更新路网动态信息：车辆、信号配时
        self.vehicles[car.id] = car  # 将车辆信息更新到车辆映射中

    # 更新信号灯状态
    def update_signal(self, sim_time):  # 更新路网动态信息：车辆、信号配时
        # 遍历所有信号灯
        for light in self.light_map.values():
            # 遍历信号灯所管理的车道
            for lane_idx in light.laneidx_lst:
                lane = self.lane_map[lane_idx]
                sim_time = int(sim_time) % light.clength  # 获取当前信号周期的时间
                lane.light.color = light.timing[sim_time]  # 更新信号灯的颜色
                color_index = defaultdict(list)
                # 创建信号灯颜色与时间索引
                for k, va in [(v, i) for i, v in enumerate(light.timing)]:
                    color_index[k].append(va)
                aa = [a - sim_time for a in color_index[lane.light.color]]
                indexs = aa.index(0)
                # 计算剩余时间
                for i in range(indexs + 1, len(aa)):
                    if aa[i] - aa[i - 1] > 1:
                        break
                lane.light.remain_time = i - indexs  # 更新剩余时间

    # 获取一个随机位置偏移的函数
    def get_pos(self, x, y):
        # 根据随机数决定偏移方向，pp 可能是 1 或 -1
        if random.random() > 0.5:
            pp = 1
        else:
            pp = -1
        # 根据当前坐标 (x, y) 以及随机生成的偏移量计算新的坐标 xytext
        xytext = (x + (4 + int(random.random() * 10)) * pp, y + (3 + int(random.random() * 10) * pp))
        return xytext

    # 绘制方法
    def draw(self, ax):
        if ax is None:
            return  # 如果 ax 为 None，直接返回，不进行绘制
        # 定义颜色
        intersection_color = 'k'  # 交叉口边界和车道的颜色设置为黑色
        link_color = 'dimgray'  # 道路链接车道的颜色设置为深灰色

        # 绘制交叉口的边界和车道
        for intersection in self.intersection_map.values():
            if intersection.virtuallink_lst[0].id in self.world:
                ax.plot(intersection.xy[0], intersection.xy[1], linestyle='--', color=intersection_color)  # 绘制交叉口边界
                for link in intersection.virtuallink_lst:  # 遍历交叉口的虚拟链接车道
                    for lane in link.iter_lane():  # 遍历每条车道
                        ax.plot(lane.xy[0], lane.xy[1], color=intersection_color)  # 绘制车道中心线（黑色）

        # 绘制道路链接的车道
        for link in self.link_map.values():
            if link.id not in self.world:  # 忽略不在世界中的道路链接
                continue
            for lane in link.iter_lane():  # 遍历道路链接中的每条车道
                ax.plot(lane.xy[0], lane.xy[1], color=link_color)  # 绘制车道中心线（深灰色）

    # 根据 link_id 获取对应的链接
    def get_link(self, link_id):
        if link_id in self.link_map.keys():
            return self.link_map[link_id]  # 如果在 link_map 中找到该 link，返回对应的 link
        else:
            return self.link_map[self.replace_linkmap[link_id]]  # 否则从 replace_linkmap 获取对应的 link

    # 获取与当前车道连接的下一个链接的车道
    def conn_lanes_of_nextlink(self, lane, link):
        # 获取当前车道的所有出车道，并返回那些属于指定链接的车道
        return [outlane for outlane in lane.out_lane_lst if outlane.ownner is link]

# 获取矩形的四个点，返回经过旋转和位移后的矩形
def get_rect(x, y, width, length, angle):
    # 定义矩形的四个点相对坐标，中心点为 (0, 0)
    rect = np.array(
        [(0, width / 2.0), (0, -width / 2.0), (-length, -width / 2.0), (-length, width / 2.0), (0, width / 2.0)])
    # 旋转角度 theta，单位为弧度
    theta = angle
    # 旋转矩阵
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    # 位移向量
    offset = np.array([x, y])
    # 将矩形的坐标点应用旋转矩阵和位移，得到新的矩形坐标
    transformed_rect = np.dot(rect, R) + offset
    return transformed_rect

# 根据车辆在车道上的位置，计算世界坐标中的位置 (x, y)
def lanepos2_worldxy(lane, position):  # 根据车辆当前状态更新车辆x,y坐标信息
    if lane is None:
        raise ValueError("Lane is None. Cannot calculate position.")  # 如果车道为空，抛出异常
    if position > lane.length:  # 如果车辆的位置大于车道的长度，进行修正
        # 如果位置接近车道的末端，则将位置修正为车道的末端
        if position < lane.length + 0.5:
            position = lane.length
    # 对车道的所有位置进行排序，并找到当前车辆的位置所在的车道段
    temp_len = lane.add_length + [position]
    temp_len.sort()
    temp_rank = temp_len.index(position)
    if temp_rank >= len(lane.direct):  # 如果当前位置超过了车道的方向数，做修正
        temp_rank -= 1
    # 获取该位置对应的方向（航向角）
    heading = lane.direct[temp_rank]
    # 根据车道的 xy 坐标和方向计算车辆的世界坐标 (x, y)
    x = lane.xy[0][temp_rank] + np.cos(heading) * (position - lane.add_length[temp_rank])
    y = lane.xy[1][temp_rank] + np.sin(heading) * (position - lane.add_length[temp_rank])
    return [x, y, heading]  # 返回计算后的世界坐标和方向

# 加载信号灯数据
def load_signals(graph, xosc):
    # 得到文档元素对象
    root = xosc.documentElement
    RoadNetwork = root.getElementsByTagName('RoadNetwork')
    signals = RoadNetwork[0].getElementsByTagName('Signals')
    Controllers = signals[0].getElementsByTagName('Controller')

    for Controller in Controllers:
        # 获取每个信号控制器中的各个信号相位
        Phases = Controller.getElementsByTagName('Phase')
        for phase in Phases:
            time_length = int(phase.getAttribute('duration'))  # 获取该相位的持续时间
            Lights = phase.getElementsByTagName('Signal')
            for light in Lights:
                light_id = int(light.getAttribute('name'))  # 获取信号灯的 ID
                # 如果该信号灯不在 graph.light_map 中，创建新的信号灯对象
                if light_id not in graph.light_map.keys():
                    new_light = Signal()
                    new_light.id = light_id
                else:
                    new_light = graph.light_map[light_id]

                # 获取信号灯的定时设置
                temp_timing = new_light.timing
                light_state = light.getAttribute('state')
                temp_state = light_state.split(';', 2)

                if len(temp_state) == 3:
                    # 根据信号灯的状态设置不同的定时器
                    if temp_state[0] == 'true':
                        temp_timing.extend([1 for _ in range(time_length)])  # 红灯
                    elif temp_state[1] == 'true':
                        temp_timing.extend([2 for _ in range(time_length)])  # 黄灯
                    elif temp_state[2] == 'true':
                        temp_timing.extend([3 for _ in range(time_length)])  # 绿灯
                    else:
                        raise Exception("Signal no timing ?")
                elif len(temp_state) == 2:
                    # 处理没有绿灯的情况
                    if temp_state[0] == 'true':
                        temp_timing.extend([1 for _ in range(time_length)])  # 红灯
                    elif temp_state[1] == 'true':
                        temp_timing.extend([3 for _ in range(time_length)])  # 绿灯
                    else:
                        raise Exception("Signal no timing ?")
                else:
                    raise Exception("Signal type number ?")

                # 将新的信号灯添加到图中的信号灯映射表
                graph.light_map[light_id] = new_light

def get_Refline(geometry):
    Rclinex = []        # 存储x坐标
    Rcliney = []        # 存储y坐标
    Rdirect = []        # 存储方向角（朝向）
    Radd_length = []    # 存储累积长度

    # 遍历所有的路径线（geometry中的每一条线）
    for Rline in geometry:
        step_length = 0.2   # 设置步长为0.2m，可以用于路径计算
        temp_Rclinex = []   # 临时存储当前路径的x坐标
        temp_Rcliney = []   # 临时存储当前路径的y坐标
        temp_Rlength = 0    # 当前路径的长度
        # 从Rline中获取起点位置和其他参数
        Rstartx = float(Rline.getAttribute('x'))        # 起点的x坐标
        Rstarty = float(Rline.getAttribute('y'))        # 起点的y坐标
        Rheading = float(Rline.getAttribute('hdg'))     # 起点的朝向（方向角）
        Rlength = float(Rline.getAttribute('length'))   # 路径的总长度

        if Rlength < 1e-3:
            continue  # 如果路径的长度过小，跳过

        # 初始化第一个点
        temp_Rclinex.append(Rstartx)
        temp_Rcliney.append(Rstarty)
        Rdirect.append(Rheading)  # 添加起点的朝向
        Radd_length.append(float(Rline.getAttribute('s')))  # 累积长度初始化

        # 获取当前路径在geometry中的索引位置
        Rline_index = geometry.index(Rline)

        # 如果当前路径不是最后一条路径，获取下一条路径的起点
        if Rline_index < len(geometry) - 1:
            nextRline = geometry[Rline_index + 1]
            nextx = float(nextRline.getAttribute('x'))
            nexty = float(nextRline.getAttribute('y'))

        # 判断路径类型，并根据路径类型进行不同的计算
        if Rline.getElementsByTagName('line'):
            # 对于直线路径
            while temp_Rlength + step_length < Rlength:
                # 使用步长沿着当前方向更新路径坐标
                temp_Rclinex.append(temp_Rclinex[-1] + step_length * math.cos(Rheading))
                temp_Rcliney.append(temp_Rcliney[-1] + step_length * math.sin(Rheading))
                temp_Rlength += step_length  # 更新当前路径长度
                Rdirect.append(Rheading)  # 记录方向
                Radd_length.append(Radd_length[-1] + step_length)  # 更新累积长度

        elif Rline.getElementsByTagName('arc'):
            # 对于弧线路径
            close2nextp = 0
            arc = Rline.getElementsByTagName('arc')
            curvature = float(arc[0].getAttribute('curvature'))  # 获取曲率
            delta_alpha = step_length * curvature  # 步长对应的方向变化
            temp_heading = Rheading  # 初始方向角

            while temp_Rlength + step_length < Rlength:
                #######
                # 用于平滑弧线/螺旋线尾端的累积误差，用直线连接目标点
                if Rline_index < len(geometry) - 1:
                    dist2nextp = math.sqrt((temp_Rclinex[-1] - nextx) ** 2 + (temp_Rcliney[-1] - nexty) ** 2)
                    # if dist2nextp < 0.2:
                    #     break
                    if dist2nextp < 1.0:
                        temp_heading = np.arctan2(nexty - temp_Rcliney[-1], nextx - temp_Rclinex[-1])
                        # if temp_heading < 0:
                        #     temp_heading += math.pi * 2
                        delta_alpha = 0
                        if close2nextp == 0:
                            Rlength = temp_Rlength + dist2nextp
                            close2nextp = 1
                #######
                temp_Rclinex.append(temp_Rclinex[-1] + step_length * math.cos(temp_heading))
                temp_Rcliney.append(temp_Rcliney[-1] + step_length * math.sin(temp_heading))
                temp_Rlength += step_length
                Rdirect.append(temp_heading)
                Radd_length.append(Radd_length[-1] + step_length)
                temp_heading += delta_alpha

        elif Rline.getElementsByTagName('spiral'):  # TODO:连接处做了平滑处理:是由于车道宽度导致的不平滑
            # 对于螺旋线
            close2nextp = 0
            spiral = Rline.getElementsByTagName('spiral')
            curvStart = float(spiral[0].getAttribute('curvStart'))  # 起始曲率
            curvEnd = float(spiral[0].getAttribute('curvEnd'))  # 结束曲率
            temp_heading = Rheading  # 初始方向角

            # 计算每一小段螺旋线的坐标
            while temp_Rlength + step_length < Rlength:
                curvature = (temp_Rlength + 0.5 * step_length) / Rlength * (curvEnd - curvStart) + curvStart
                delta_alpha = step_length * curvature
                if Rline_index < len(geometry) - 1:
                    dist2nextp = math.sqrt((temp_Rclinex[-1] - nextx) ** 2 + (temp_Rcliney[-1] - nexty) ** 2)
                    if dist2nextp < 1.0:
                        temp_heading = np.arctan2(nexty - temp_Rcliney[-1], nextx - temp_Rclinex[-1])
                        # if temp_heading < 0:
                        #     temp_heading += math.pi * 2
                        delta_alpha = 0
                        if close2nextp == 0:
                            Rlength = temp_Rlength + dist2nextp
                            close2nextp = 1

                # 更新螺旋线的坐标
                temp_Rclinex.append(temp_Rclinex[-1] + step_length * math.cos(temp_heading))  # 以0.1m作为步长
                temp_Rcliney.append(temp_Rcliney[-1] + step_length * math.sin(temp_heading))
                temp_Rlength += step_length
                Rdirect.append(temp_heading)
                Radd_length.append(Radd_length[-1] + step_length)
                temp_heading += delta_alpha

        elif Rline.getElementsByTagName('poly3'):
            pass  # 目前不处理三次多项式路径

        elif Rline.getElementsByTagName('paramPoly3'):
            pass  # 目前不处理参数化三次多项式路径

        else:
            raise Exception("Unknown Geometry !!!")  # 遇到未知路径类型抛出异常

            # 将当前路径计算得到的坐标和信息加入到结果列表中
        Rclinex = Rclinex + temp_Rclinex
        Rcliney = Rcliney + temp_Rcliney

    # 对最终的坐标进行后处理，修正接近点的方向
    for i in range(1, len(Rclinex) - 1):
        if abs(Rcliney[i + 1] - Rcliney[i]) < 1e-6 and abs(Rclinex[i + 1] - Rclinex[i]) < 1e-6 and i > 0:
            # 如果两个点非常接近，则更新方向角
            Rdirect[i] = Rdirect[i - 1]

    return Rclinex, Rcliney, Rdirect, Radd_length

def create_road(graph, xodr, ax):
    #得到文档元素对象
    root = xodr.documentElement
    links = root.getElementsByTagName('road')
    for road in links:
        new_link = Link()
        new_link0 = Link()
        new_link.id = int(road.getAttribute('id'))
        if new_link.id == 809:
            aaa = 1
        new_link0.id = -int(road.getAttribute('id'))
        junction = int(road.getAttribute('junction'))
        new_link.junction_id = junction
        new_link0.junction_id = junction
        temp_link = road.getElementsByTagName('link')
        if temp_link:  # 检查 temp_link 是否非空
            link_successor_id = None
            link_successor = temp_link[0].getElementsByTagName('successor')
            if link_successor and link_successor[0].getAttribute('elementType') == "road":
                link_successor_id = int(link_successor[0].getAttribute('elementId'))
                # new_link.in_link_lst.append(link_predecessor_id) #TODO:还未考虑多个上下游的情况：已考虑junction，找路口进行验证

            link_predecessor_id = None
            link_predecessor = temp_link[0].getElementsByTagName('predecessor')
            if link_predecessor and link_predecessor[0].getAttribute('elementType') == "road":
                link_predecessor_id = int(link_predecessor[0].getAttribute('elementId'))
                # new_link.out_link_lst.append(link_successor_id)  #TODO:还未考虑多个上下游的情况：已考虑junction，找路口进行验证
        else:
            print(f"No 'link' elements found for road id: {road.getAttribute('id')}")

        plan_view = road.getElementsByTagName('planView')
        geometry = plan_view[0].getElementsByTagName('geometry')
        [Rclinex, Rcliney, Rdirect, Radd_length] = get_Refline(geometry)
        elevationProfile = road.getElementsByTagName('elevationProfile') #TODO：暂时没有考虑高程
        temp_lanes = road.getElementsByTagName('lanes')
        laneSection = temp_lanes[0].getElementsByTagName('laneSection') #TODO：可能有多段section
        lanes = laneSection[0].getElementsByTagName('lane')
        lane_border_list = {}
        lane_width_list = {}
        for lane in lanes:
            new_lane = Lane()
            new_lane.id = int(lane.getAttribute('id')) #为了区分不同车道的情况
            if new_lane.id >= 0:
                new_lane.link_id = new_link.id
            else:
                new_lane.link_id = new_link0.id
            if new_lane.index_id in graph.lane_map.keys():
                new_lane = graph.get_lane(new_lane.index_id)
            else:
                graph.add_lane(new_lane)
            new_lane.type = lane.getAttribute('type')
            width = lane.getElementsByTagName('width')
            if not width:
                continue #如果没有width这个标签说明为地面标线，不是车道
            for k in range(0, len(width)):
                a = float(width[k].getAttribute('a'))
                b = float(width[k].getAttribute('b'))
                c = float(width[k].getAttribute('c'))
                d = float(width[k].getAttribute('d'))
                offset_pre = float(width[k].getAttribute('sOffset'))
                temp_alength = Radd_length + [offset_pre]
                temp_alength.sort()
                temp_index = temp_alength.index(offset_pre)
                roadMark = lane.getElementsByTagName('roadMark') #TODO：暂时没有考虑标线
                # m_width = float(roadMark[0].getAttribute('width'))
                temp_width = [a + b * (s - offset_pre) + c * (s - offset_pre) ** 2 + d * (s - offset_pre) ** 3 for s in Radd_length[temp_index:]]
                new_lane.width[temp_index:] = temp_width
            if new_lane.type != 'driving' and new_lane.type != 'special1' and new_lane.type != 'offRamp' and new_lane.type != 'onRamp':
                # if len(lane_border_list) == 0:# 应该先计算中间车道的坐标点，再计算外侧车道
                #     Rclinex = [x + a * math.cos(h + np.sign(new_lane.id) * math.pi / 2.0) for (x, h) in zip(Rclinex, Rdirect)]
                #     Rcliney = [y + a * math.sin(h + np.sign(new_lane.id) * math.pi / 2.0) for (y, h) in zip(Rcliney, Rdirect)]
                lane_border_list[new_lane.id] = new_lane
                lane_width_list[new_lane.id] = new_lane.width
                continue
            lane_successor = lane.getElementsByTagName('successor')
            if lane_successor:
                lane_successor_id = int(lane_successor[0].getAttribute('id'))
                try:
                    if new_link.id == 809:
                        aaa = 1
                    link_successor_id0 = int(np.sign(lane_successor_id)) * link_successor_id
                    suc_id = link_successor_id0 * 100 + lane_successor_id
                    if suc_id in graph.lane_map.keys():
                        suc_lane = graph.get_lane(suc_id)
                    else:
                        suc_lane = Lane()
                        suc_lane.link_id = link_successor_id0
                        suc_lane.id = lane_successor_id
                        graph.add_lane(suc_lane)
                    if suc_id not in new_lane.out_lane_id_lst and new_lane.id < 0:
                        new_lane.out_lane_id_lst.append(suc_id) #目前lane_id_lst存的都是修正过的车道id
                    elif suc_id not in new_lane.in_lane_id_lst and new_lane.id > 0:
                        new_lane.in_lane_id_lst.append(suc_id)  # 目前lane_id_lst存的都是修正过的车道id
                    if new_lane.index_id not in suc_lane.in_lane_id_lst and new_lane.id < 0:
                        suc_lane.in_lane_id_lst.append(new_lane.index_id)
                    elif new_lane.index_id not in suc_lane.out_lane_id_lst and new_lane.id > 0:
                        suc_lane.out_lane_id_lst.append(new_lane.index_id)
                except:
                    pass
            lane_predecessor = lane.getElementsByTagName('predecessor')
            if lane_predecessor:
                lane_predecessor_id = int(lane_predecessor[0].getAttribute('id'))
                link_predecessor_id0 = int(np.sign(lane_predecessor_id)) * (link_predecessor_id if link_predecessor_id is not None else 0)
                pre_id = link_predecessor_id0 * 100 + lane_predecessor_id  # 新增此行

                if pre_id in graph.lane_map.keys():
                    pre_lane = graph.get_lane(pre_id)
                else:
                    pre_lane = Lane()
                    pre_lane.link_id = link_predecessor_id0
                    pre_lane.id = lane_predecessor_id
                    graph.add_lane(pre_lane)
                if pre_id not in new_lane.in_lane_id_lst and new_lane.id < 0:
                    new_lane.in_lane_id_lst.append(pre_id)
                elif pre_id not in new_lane.out_lane_id_lst and new_lane.id > 0:
                    new_lane.out_lane_id_lst.append(pre_id)
                if new_lane.index_id not in pre_lane.out_lane_id_lst and new_lane.id < 0:
                    pre_lane.out_lane_id_lst.append(new_lane.index_id)
                elif new_lane.index_id not in pre_lane.in_lane_id_lst and new_lane.id > 0:
                    pre_lane.in_lane_id_lst.append(new_lane.index_id)
            lane_border_list[new_lane.id] = new_lane
            lane_width_list[new_lane.id] = new_lane.width

        for lane_id, new_lane in sorted(lane_border_list.items()):
            if lane_id <= 0:
                continue
            # if new_lane.type == 'driving' and new_lane.id == 1:
            #     Rclinex0 = Rclinex
            #     Rcliney0 = Rcliney
            # if new_lane.type != 'driving':
            #     Rclinex0 = [x + w * math.cos(h + np.sign(new_lane.id) * math.pi / 2.0) for (x, h, w) in zip(Rclinex, Rdirect, lane_width_list[lane_id])]
            #     Rcliney0 = [y + w * math.sin(h + np.sign(new_lane.id) * math.pi / 2.0) for (y, h, w) in zip(Rcliney, Rdirect, lane_width_list[lane_id])]
            if lane_id - 1 in lane_border_list.keys():
                clinex = [x + (w1 + w2) * 0.5 * math.cos(h + np.sign(new_lane.id) * math.pi / 2.0) for (x, h, w1, w2) in zip(lane_border_list[lane_id-1].xy[0], Rdirect, lane_width_list[lane_id], lane_width_list[lane_id-1])] #应该先计算中间车道的坐标点，再计算外侧车道
                cliney = [y + (w1 + w2) * 0.5 * math.sin(h + np.sign(new_lane.id) * math.pi / 2.0) for (y, h, w1, w2) in zip(lane_border_list[lane_id-1].xy[1], Rdirect, lane_width_list[lane_id], lane_width_list[lane_id-1])]
            else:
                clinex = [x +  w * 0.5 * math.cos(h + np.sign(new_lane.id) * math.pi / 2.0) for (x, h, w) in zip(Rclinex[::-1], Rdirect, lane_width_list[lane_id])]  # 应该先计算中间车道的坐标点，再计算外侧车道
                cliney = [y +  w * 0.5 * math.sin(h + np.sign(new_lane.id) * math.pi / 2.0) for (y, h, w) in zip(Rcliney[::-1], Rdirect, lane_width_list[lane_id])]
            new_lane.xy = [clinex, cliney]
            # new_lane.xy = [clinex[::-np.sign(new_lane.id)], cliney[::-np.sign(new_lane.id)]]  # 车道id为负的话，需要倒序xy坐标
            lane_border_list[new_lane.id] = new_lane
        for lane_id, new_lane in sorted(lane_border_list.items(), reverse=True):
            if lane_id >= 0:
                continue
            # if new_lane.type == 'driving' and new_lane.id == -1:
            #     Rclinex1 = Rclinex
            #     Rcliney1 = Rcliney
            # if new_lane.type != 'driving':
            #     Rclinex1 = [x + w * math.cos(h + np.sign(new_lane.id) * math.pi / 2.0) for (x, h, w) in zip(Rclinex, Rdirect, lane_width_list[lane_id])]
            #     Rcliney1 = [y + w * math.sin(h + np.sign(new_lane.id) * math.pi / 2.0) for (y, h, w) in zip(Rcliney, Rdirect, lane_width_list[lane_id])]
            if lane_id + 1 in lane_border_list.keys():
                clinex = [x + (w1 + w2) * 0.5 * math.cos(h + np.sign(new_lane.id) * math.pi / 2.0) for (x, h, w1, w2) in zip(lane_border_list[lane_id+1].xy[0], Rdirect, lane_width_list[lane_id], lane_width_list[lane_id+1])] #应该先计算中间车道的坐标点，再计算外侧车道
                cliney = [y + (w1 + w2) * 0.5 * math.sin(h + np.sign(new_lane.id) * math.pi / 2.0) for (y, h, w1, w2) in zip(lane_border_list[lane_id+1].xy[1], Rdirect, lane_width_list[lane_id], lane_width_list[lane_id+1])]
            else:
                clinex = [x + w * 0.5 * math.cos(h + np.sign(new_lane.id) * math.pi / 2.0) for (x, h, w) in zip(Rclinex, Rdirect, lane_width_list[lane_id])]  # 应该先计算中间车道的坐标点，再计算外侧车道
                cliney = [y + w * 0.5 * math.sin(h + np.sign(new_lane.id) * math.pi / 2.0) for (y, h, w) in zip(Rcliney, Rdirect, lane_width_list[lane_id])]
            # new_lane.xy = [clinex[::-np.sign(new_lane.id)], cliney[::-np.sign(new_lane.id)]]  # 车道id为负的话，需要倒序xy坐标
            new_lane.xy = [clinex, cliney]
            lane_border_list[new_lane.id] = new_lane

        for lane_id, new_lane in lane_border_list.items():
            [new_lane.direct, new_lane.add_length] = get_line_feature(new_lane.xy)
            avg_width = np.mean(new_lane.width)
            if avg_width < 0.1 or (new_lane.type != 'driving' and new_lane.type != 'special1' and new_lane.type != 'offRamp' and new_lane.type != 'onRamp'): #正常来说只有driving，这个路网之前是定制化的，对不同道路设置了不同的类型
                continue
            # new_lane.type = new_link.type
            if lane_id > 0:
                new_link.lane_lst.append(new_lane)
            else:
                new_link0.lane_lst.append(new_lane)

        if new_link.lane_lst:
            new_link.lane_lst.sort(key=lambda x: x.id, reverse=True)
            graph.add_link(new_link)
        if new_link0.lane_lst:
            new_link0.lane_lst.sort(key=lambda x: x.id, reverse=False)
            graph.add_link(new_link0)

        signals = road.getElementsByTagName('signals')
        if len(signals) == 0:
            pass
        else:
            signal_lst = signals[0].getElementsByTagName('signal')
            for signal in signal_lst:
                sign_id = int(signal.getAttribute('id'))
                if sign_id not in graph.light_map.keys(): #道路标志标线，非信号灯
                    continue
                sign = graph.light_map[sign_id]
                sign.pos = float(signal.getAttribute('s'))
                sign.link = new_link
                try:
                    valid = signal.getElementsByTagName('validity')
                    slane_id = int(valid[0].getAttribute('fromLane'))
                    elane_id = int(valid[0].getAttribute('toLane'))
                    if slane_id and elane_id:
                        for lid in range(slane_id, elane_id + 1):
                            sign.laneidx_lst.append(sign.link.id * 100 + lid)
                    else:
                        sign.laneidx_lst = [l.index_id for l in sign.link.lane_lst]
                except:
                    sign.laneidx_lst = [l.index_id for l in sign.link.lane_lst]


    junctions = root.getElementsByTagName('junction')
    for junction in junctions:
        junction_id = int(junction.getAttribute('id'))
        connections = junction.getElementsByTagName('connection')
        for connection in connections:
            incomingRoad_id = int(connection.getAttribute('incomingRoad'))
            if incomingRoad_id == 765:
                aaa = 1
            connectingRoad_id = int(connection.getAttribute('connectingRoad'))
            laneLinks = connection.getElementsByTagName('laneLink')
            for laneLink in laneLinks:
                pre_id = int(laneLink.getAttribute('from'))
                suc_id = int(laneLink.getAttribute('to'))
                incomingRoad_id0 = np.sign(pre_id) * incomingRoad_id
                connectingRoad_id0 = np.sign(suc_id) * connectingRoad_id
                incomingRoad = graph.link_map[incomingRoad_id0]
                if connectingRoad_id0 in graph.link_map:
                    connectingRoad = graph.link_map[connectingRoad_id0]
                else:
                    print(f"Warning: connectingRoad_id0 {connectingRoad_id0} not found in link_map.")
                    connectingRoad = None  # 或者其他处理方式
                # connectingRoad = graph.link_map[connectingRoad_id0]
                if connectingRoad is not None:
                    for lane in incomingRoad.lane_lst:
                        if lane.id == pre_id:
                            new_id = connectingRoad_id0 * 100 + suc_id
                            if new_id not in lane.out_lane_id_lst:
                                lane.out_lane_id_lst.append(new_id)
                    for lane in connectingRoad.lane_lst:
                        if lane.id == suc_id:
                            new_id = incomingRoad_id0 * 100 + pre_id
                            if new_id not in lane.in_lane_id_lst:
                                lane.in_lane_id_lst.append(new_id)
                else:
                    pass

    graph.build_topo()
    # graph.load_cross_point('lane_cross')
    # graph.build_cross()

    for link in graph.link_map.values():
        link.lane_lst.sort(key=lambda x: x.id, reverse=False)
        for lane in link.lane_lst:
            if isinstance(lane.width, list):
                lane.width = lane.width[0]


    for link in graph.link_map.values():
        if link.id > 0:
            link.lane_lst.sort(key=lambda x: x.id, reverse=False)
        else:
            link.lane_lst.sort(key=lambda x: x.id, reverse=True)
        for lane in link.lane_lst:
            lane.index = link.lane_lst.index(lane) + 1
        # for lane in link.lane_lst:
        #     if link.id > 0:
        #         lane.index = len(link.lane_lst) - link.lane_lst.index(lane)
        #     else:
        #         lane.index = link.lane_lst.index(lane) + 1

    # graph.link_combine()
    return graph

# 根据车道中心线坐标计算行驶方向和线长度序列
def get_lane_feature(xy):
    # 将传入的xy转换为numpy数组，方便后续运算
    xy = np.array(xy)

    # 提取中心线的x和y坐标，x_prior是前一个点的x坐标，y_prior是前一个点的y坐标
    x_prior = xy[0][:-1]
    y_prior = xy[1][:-1]
    x_post = xy[0][1:]
    y_post = xy[1][1:]

    # 计算相邻两个点之间的x和y坐标的差值
    dx = x_post - x_prior
    dy = y_post - y_prior

    # 根据差值计算行驶方向（弧度制），并将方向转为0到2π范围内
    direction = list(map(lambda d: d > 0 and d or d + 2 * np.pi, np.arctan2(dy, dx)))

    # 计算每段中心线的长度
    length = np.sqrt(dx ** 2 + dy ** 2)
    length = length.tolist()

    # 累积每段的长度，得到总长度
    for i in range(len(length) - 1):
        length[i + 1] += length[i]
    length.insert(0, 0)  # 将起始点的长度设为0

    # 返回计算得到的行驶方向和总长度
    return direction, length


def read_csv(file_path):  # 从csv文件中读取数据
    file_path.encode('utf-8')       # 对文件路径进行UTF-8编码
    file = open(file_path)          # 打开csv文件
    file_reader = csv.reader(file)  # 创建csv读取器
    for line in file_reader:        # 逐行读取文件
        yield line                  # 返回每行内容


def detail_xy(xy):
    # 获取车道的行驶方向和总长度
    [direct, add_length] = get_lane_feature(xy)
    dist_interval = 0.1             # 定义加密点的间隔为0.1米
    new_xy = [[], []]               # 存储加密后的点的坐标
    new_direct = []                 # 存储加密后点的行驶方向
    new_add_len = [0]               # 存储加密后的点的累计长度
    temp_length = dist_interval     # 当前点的累计长度

    # 遍历车道中心线上的点
    for k in range(0, len(xy[0]) - 1):
        new_xy[0].append(xy[0][k])          # 添加原点的x坐标
        new_xy[1].append(xy[1][k])          # 添加原点的y坐标
        new_add_len.append(temp_length)     # 添加累计长度
        new_direct.append(direct[k])        # 添加行驶方向

        # 加密点的过程：当当前累计长度小于该段车道的总长度时，继续添加加密点
        while temp_length < add_length[k + 1]:
            temp_length += dist_interval  # 更新累计长度
            new_xy[0].append(new_xy[0][-1] + dist_interval * math.cos(direct[k]))  # 计算新的x坐标
            new_xy[1].append(new_xy[1][-1] + dist_interval * math.sin(direct[k]))  # 计算新的y坐标
            new_add_len.append(temp_length)     # 添加新的累计长度
            new_direct.append(direct[k])        # 添加新的行驶方向

    return [new_xy, new_direct, new_add_len]  # 返回加密后的坐标、方向和长度


def worldxy2_lanepos(world_x, world_y, current_link, flag, last_pos, last_lane):
    # 根据车道以及距离中心线起点的长度，计算该点的二维坐标和方向角
    search_r = 5  # 定义搜索半径
    min_dist = 1  # 定义最小距离
    rest_len = last_lane.length - last_pos  # 计算剩余长度
    bond_lanes = []  # 存储与当前点相关联的车道

    # 根据标志位决定是否包含当前车道
    if flag is True:
        temp_lanes = []  # 如果flag为True，则不处理当前车道
    else:
        temp_lanes = current_link.lane_lst  # 获取当前链接的所有车道

    # 遍历与当前车道相关联的车道
    for temp_lane in temp_lanes:
        temp_lane = temp_lane.cut_lane(last_pos, 0, search_r)  # 从指定位置截取车道
        bond_lanes.append(temp_lane)

    # 如果剩余的长度小于搜索半径减去最小距离，则需要考虑下一个链接的车道
    if rest_len < search_r - min_dist:
        next_links = current_link.out_link_lst  # 获取出链接的车道
        more_len = search_r - rest_len  # 计算需要补充的长度
        for next_link in next_links:
            for next_lane in next_link.lane_lst:
                add_lane = next_lane.cut_lane(0, 0, more_len)  # 截取下一个车道
                if not add_lane.add_length:  # 如果截取的车道没有长度，则跳过
                    continue
                bond_lanes.append(add_lane)

    # 计算每条相关车道与当前位置的距离
    min_dists = []
    for lane in bond_lanes:
        dist = float('inf')  # 初始化最小距离为无限大
        coord_num = 0
        if lane.add_length[1] - lane.add_length[0] < 0.2:  # 如果车道长度过短
            lane_xy = lane.xy
        else:
            # 如果车道长度较长，使用加密后的车道
            [lane_xy, _, _] = detail_xy(lane.xy)

        pos = lane.add_length[0]  # 初始化位置为车道的起点
        min_pos = pos  # 初始化最小位置

        # 遍历车道上的每一个点，计算该点与当前位置的距离
        while coord_num < len(lane_xy[0]) - 1:
            dist_interval = math.sqrt((lane_xy[0][coord_num + 1] - lane_xy[0][coord_num]) ** 2 + (
                        lane_xy[1][coord_num + 1] - lane_xy[1][coord_num]) ** 2)
            temp = (lane_xy[0][coord_num] - world_x) ** 2 + (lane_xy[1][coord_num] - world_y) ** 2  # 计算当前点的距离
            if dist > temp:  # 如果该点距离更近，则更新最小距离和位置
                dist = temp
                min_pos = pos
            pos += dist_interval  # 更新位置
            coord_num += 1  # 遍历下一个点
        min_dists.append([dist, min_pos])  # 存储每条车道的最小距离和位置

    # 选择最小距离的车道
    min_d = reduce(lambda p1, p2: p1[0] < p2[0] and p1 or p2, min_dists)
    lane_ind = min_dists.index(min_d)  # 获取最小距离的车道索引
    veh_lane = bond_lanes[lane_ind].index_id  # 获取该车道的ID

    # 返回车辆所在的车道ID和位置
    veh_pos = min_dists[lane_ind][1]
    return [veh_lane, veh_pos]


def worldxy2_lanepos_long(world_x, world_y, current_link, flag):
    # 根据车道以及距离中心线起点的长度，计算该点的二维坐标和方向角，支持长距离搜索
    if flag is True:
        bond_lanes = []  # 如果flag为True，则不处理当前车道
    else:
        bond_lanes = current_link.lane_lst  # 获取当前链接的所有车道

    # 遍历所有出链接的车道，加入相关车道
    next_links = current_link.out_link_lst
    for next_link in next_links:
        bond_lanes = bond_lanes + next_link.lane_lst  # 合并所有车道

    min_dists = []  # 存储每条车道的最小距离和位置
    for lane in bond_lanes:
        dist = float('inf')  # 初始化最小距离为无限大
        coord_num = 0
        lane_xy = detail_xy(lane.xy)  # 获取加密后的车道坐标
        pos = 0  # 初始化位置为车道的起点

        # 遍历车道上的每一个点，计算该点与当前位置的距离
        while coord_num < len(lane_xy[0]) - 1:
            dist_interval = math.sqrt((lane_xy[0][coord_num + 1] - lane_xy[0][coord_num]) ** 2 + (
                        lane_xy[1][coord_num + 1] - lane_xy[1][coord_num]) ** 2)
            pos += dist_interval  # 更新位置
            temp = (lane_xy[0][coord_num] - world_x) ** 2 + (lane_xy[1][coord_num] - world_y) ** 2  # 计算当前点的距离
            if dist > temp:  # 如果该点距离更近，则更新最小距离和位置
                dist = temp
                min_pos = pos
            coord_num += 1  # 遍历下一个点

        min_dists.append([dist, min_pos])  # 存储每条车道的最小距离和位置

    # 选择最小距离的车道
    min_d = reduce(lambda p1, p2: p1[0] < p2[0] and p1 or p2, min_dists)
    lane_ind = min_dists.index(min_d)  # 获取最小距离的车道索引
    veh_lane = bond_lanes[lane_ind]  # 获取该车道的对象

    # 返回车辆所在的车道对象和位置
    veh_pos = min_dists[lane_ind][1]
    return [veh_lane, veh_pos]


