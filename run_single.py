import os
import xml.dom.minidom
import pickle
import numpy as np
import pandas as pd
import argparse
from simulator import Sim
from tkinter import Tk
from util.modification import adjust_positions, interpolate_short_trajectories, smooth_trajectories

def process_folder(xodr_file, exam_file, output_dir):
    # 读取xodr文件
    xodr = xml.dom.minidom.parse(xodr_file)
    
    # 读取数据文件
    with open(exam_file, 'rb') as f:
        data = pickle.load(f)
    
    # 确定stop_time
    stop_time = 100  # 根据实际情况调整
    
    # 确定total_frames
    total_frames = 32
    
    # 创建frame_labels
    sim_freq = data['sim_freq']
    frame_labels = np.arange(total_frames).astype(float) / sim_freq
    frame_labels = np.round(frame_labels * 10) / 10  # 四舍五入到一位小数
    
    # obj_ids从1开始
    obj_ids = np.arange(1, len(data['ids']) + 1)
    
    # 初始化列表存储数据
    rows = []
    
    # 遍历每个对象和每个帧
    for obj_idx in range(len(data['ids'])):
        for frame_idx in range(total_frames):
            if frame_idx < total_frames and data['valid_mask'][obj_idx, frame_idx]:
                position = data['positions'][obj_idx, frame_idx]
                heading = data['headings'][obj_idx, frame_idx]
                obj_type = data['types'][obj_idx]
                length = data['shapes'][obj_idx, 0]
                width = data['shapes'][obj_idx, 1]
                current_obj_id = obj_ids[obj_idx]
                is_test = 1 if current_obj_id == 1 else 0
                if frame_idx == 0:
                    speed = 0.0
                else:
                    if data['valid_mask'][obj_idx, frame_idx - 1]:
                        delta_pos = position - data['positions'][obj_idx, frame_idx - 1]
                        speed = np.linalg.norm(delta_pos) * sim_freq
                    else:
                        speed = 0.0
                
                row = {
                    'frame_label': frame_labels[frame_idx],
                    'obj_id': current_obj_id,
                    'center_x': position[0],
                    'center_y': position[1],
                    'yaw': heading,
                    'speed': speed,
                    'length': length,
                    'width': width,
                    'obj_type': obj_type,
                    'is_test': is_test
                }
                rows.append(row)
    
    # 创建DataFrame
    df = pd.DataFrame(rows)
    
    # 将frame_label四舍五入到一位小数
    df['frame_label'] = df['frame_label'].round(1)
    
    # 筛选出 frame_label 为 3.0 的所有行，并且 obj_type 为 0
    df_filtered = df[(df['frame_label'] == 3.0) & (df['obj_type'] == 0)]
    
    # 创建一个空列表来存储 vehicles 数据
    vehicles = []
    
    # 将筛选后的数据转换为列表 of lists
    for index, row in df_filtered.iterrows():
        vehicle = [
            row['frame_label'],
            row['obj_id'],
            row['center_x'],
            row['center_y'],
            row['yaw'],
            row['speed'],
            row['length'],
            row['width'],
            row['obj_type'],
            row['is_test']
        ]
        vehicles.append(vehicle)
    
    # 创建Sim类的实例，并传入xodr对象
    sim = Sim(xodr, Tk(), vehicles, stop_time, show_plot=False)
    sim.Run(show_plot=False)
    output = sim.save_data()
    output['positions'] = output['positions'][1:,:,:]
    output['headings'] = output['headings'][1:,:]
    
    # 获取需要合并的对象索引
    predict_mask = data['predict_mask']  # 预测的掩码，表示哪些对象需要更新
    obj_indices = np.where(predict_mask)[0]  # 找到需要更新的对象的索引
    ids_to_concat = [data['ids'][i] for i in obj_indices]  # 获取这些对象的ID

    # 获取原始帧数和输出帧数
    num_frames_original = data['positions'].shape[1]  # 原始数据中的帧数
    num_frames_output = output['positions'].shape[1]  # 输出数据中的帧数
    start_frame = 31  # 从第31帧开始更新

    # 遍历每个需要更新的对象
    for idx, obj_idx in enumerate(obj_indices):
        obj_idx_output = idx  # 假设输出数据与ids_to_concat的顺序一致

        end_frame = start_frame + num_frames_output  # 计算结束帧

        if end_frame > num_frames_original:  # 如果输出帧数超过原始帧数
            frames_to_add = num_frames_original - start_frame  # 计算需要添加的帧数
            if frames_to_add <= 0:  # 如果没有足够空间添加新的帧
                continue  # 跳过此对象

            output_positions_truncated = output['positions'][obj_idx_output, :frames_to_add, :]  # 截取有效的输出位置
            output_headings_truncated = output['headings'][obj_idx_output, :frames_to_add]  # 截取有效的输出朝向
            valid_frames = np.any(output_positions_truncated != 0, axis=1)

            # 只更新有效帧的位置信息和朝向
            data['positions'][obj_idx, start_frame:start_frame + frames_to_add][valid_frames] = \
            output_positions_truncated[valid_frames]
            data['headings'][obj_idx, start_frame:start_frame + frames_to_add][valid_frames] = \
            output_headings_truncated[valid_frames]

            # 根据有效帧更新valid_mask
            data['valid_mask'][obj_idx, start_frame:start_frame + frames_to_add] = valid_frames
        else:
            valid_frames = np.any(output['positions'][obj_idx_output, :, :] != 0, axis=1)

            # 只更新有效帧的位置信息和朝向
            data['positions'][obj_idx, start_frame:end_frame][valid_frames] = output['positions'][
                obj_idx_output, valid_frames]
            data['headings'][obj_idx, start_frame:end_frame][valid_frames] = output['headings'][
                obj_idx_output, valid_frames]

            # 根据有效帧更新valid_mask
            data['valid_mask'][obj_idx, start_frame:end_frame] = valid_frames

    # 应用修正逻辑
    print(f"Processing {os.path.basename(os.path.dirname(xodr_file))}")
    data = adjust_positions(data)
    data = smooth_trajectories(data)
    data = interpolate_short_trajectories(data)

    # 确定统一输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取根目录的文件夹名并添加 "_output" 后缀
    output_file_name = os.path.basename(os.path.dirname(xodr_file)) + '_output.pkl'
    
    # 构造输出文件路径
    output_file = os.path.join(output_dir, output_file_name)
    
    # 保存文件
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"处理完成，输出保存至: {output_file}")

if __name__ == '__main__':
    # 配置命令行参数
    parser = argparse.ArgumentParser(description='处理单个交通仿真场景')
    parser.add_argument('--xodr_file', required=True, help='输入xodr文件')
    parser.add_argument('--exam_file', required=True, help='输入exam文件')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    
    args = parser.parse_args()
    
    process_folder(args.xodr_file, args.exam_file, args.output_dir)

# python run_single.py --xodr_file Onsite\第五赛道_B卷\0_6_straight_straight_19\0_6_straight_straight_19.xodr --exam_file Onsite\第五赛道_B卷\0_6_straight_straight_19\0_6_straight_straight_19_exam.pkl --output_dir Onsite\output_A
