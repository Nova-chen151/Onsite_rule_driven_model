from roads import D1xodrRoads
import xml.dom.minidom
import matplotlib.pyplot as plt
import random


def find_legal_connections_between_o_d(link_map, o_points, d_points):
    legal_connections = []  # 存储合法的连接路径
    # 遍历所有的起点 o_points
    for origin_link_id in o_points:
        # 遍历所有的终点 d_points
        for destination_link_id in d_points:
            # 如果起点和终点相同，则跳过
            if origin_link_id == destination_link_id:
                continue
            try:
                # 获取起点到终点的可用路径
                path = get_available_path(link_map, origin_link_id, destination_link_id)
                # 如果路径不在合法连接中，则添加路径
                if path not in legal_connections:
                    legal_connections.append(path)
            except Exception:
                pass  # 如果获取路径时发生异常，忽略该连接
    return legal_connections  # 返回所有合法连接路径


def get_available_path(link_map, origin, destination):
    paths = []                      # 存储所有可用路径
    path = [origin]                 # 初始化路径，包含起点
    outlink_id_lst_pre = [origin]   # 初始化前驱路径列表，包含起点
    paths.append(path)              # 将初始路径添加到路径列表中
    path_length = 1                 # 最大路径长度，初始值为 1（即起点）

    # 当前驱路径列表不为空且路径长度小于 20 时继续遍历
    while outlink_id_lst_pre and path_length < 20:
        nextlink_id_lst = outlink_id_lst_pre    # 备份当前前驱路径列表
        paths_new = []                          # 用于存储新的路径列表
        outlink_id_lst_pre = []                 # 清空前驱路径列表，准备存储新的路径
        # 遍历前驱路径列表中的每个链接 ID
        for link_id in nextlink_id_lst:
            outlink_id_lst = []  # 临时存储后续链接 ID 列表
            # 获取当前链接的详细信息
            nextlink = link_map.get(link_id)
            if not nextlink or not nextlink.out_link_lst:  # 如果链接不存在或者没有输出链接，跳过
                continue
            # 遍历当前链接的所有输出链接
            for ll in nextlink.out_link_lst:
                # 如果该链接 ID 不在后续链接列表中且不等于起点，则添加到后续链接列表
                if ll.id not in outlink_id_lst and ll.id != origin:
                    outlink_id_lst.append(ll.id)
                    outlink_id_lst_pre.append(ll.id)  # 将后续链接添加到前驱路径列表
            # 遍历已有路径
            for path in paths:
                # 如果路径的最后一个节点等于当前链接 ID
                if path[-1] == link_id:
                    # 遍历后续链接，避免形成环路
                    for lid in outlink_id_lst:
                        # 如果后续链接 ID 已经在路径中，则跳过
                        if lid in path:
                            # 如果路径未在新路径列表中，则添加
                            if path not in paths_new:
                                paths_new.append(path)
                            continue
                        # 创建新的路径
                        path_new = path + [lid]
                        # 如果新路径不在路径列表中，则添加
                        if path_new not in paths_new:
                            paths_new.append(path_new)
        # 如果找到新的路径，则更新路径列表
        if paths_new:
            paths = paths_new
        path_length += 1  # 增加路径长度
        # 遍历所有路径，如果某条路径的最后一个节点是目标点，返回该路径
        for pathx in paths:
            if pathx[-1] == destination:
                return pathx
    # 如果没有找到可用路径，抛出异常
    raise Exception("No available path")


def organize_legal_connections(legal_connections):
    organized_connections = []  # 存储组织后的连接路径
    # 遍历所有的合法连接路径
    for path in legal_connections:
        # 如果路径长度大于 2，则仅保留起点和终点
        if len(path) > 2:
            organized_connections.append([path[0], path[-1], 0])
        # 如果路径长度等于 2，直接保存该路径
        elif len(path) == 2:
            organized_connections.append([path[0], path[1], 0])
    return organized_connections  # 返回组织后的连接路径


def generate_basic_od_pairs(link_map):
    od_pairs = []                       # 存储 OD 对
    link_ids = list(link_map.keys())    # 获取所有 link ID
    # 遍历所有的 link ID，生成 OD 对
    for i in range(len(link_ids)):
        for j in range(i + 1, len(link_ids)):
            # 生成一个基础的 OD 对，默认值为 0
            od_pairs.append([link_ids[i], link_ids[j], 0])
    return od_pairs  # 返回所有生成的 OD 对


