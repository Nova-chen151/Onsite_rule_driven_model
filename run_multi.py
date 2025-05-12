import os
import xml.dom.minidom
import pickle
import numpy as np
import pandas as pd
import glob
import argparse
from simulator import Sim
from tkinter import Tk
from util.modification import adjust_positions, interpolate_short_trajectories, smooth_trajectories, handle_unresolvable_violations

def process_folder(folder_path, output_dir):
    """处理单个文件夹，生成轨迹并应用所有修正"""
    # 读取 xodr 文件
    xodr_file = os.path.join(folder_path, os.path.basename(folder_path) + '.xodr')
    xodr = xml.dom.minidom.parse(xodr_file)

    # 读取数据文件
    data_file = glob.glob(os.path.join(folder_path, '*_exam.pkl'))[0]
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    # 确定仿真参数
    stop_time = 800  # 根据实际情况调整
    total_frames = 32

    # 创建 frame_labels
    sim_freq = data['sim_freq']
    frame_labels = np.arange(total_frames).astype(float) / sim_freq
    frame_labels = np.round(frame_labels * 10) / 10  # 四舍五入到一位小数

    # obj_ids 从 1 开始
    obj_ids = np.arange(1, len(data['ids']) + 1)

    # 初始化列表存储数据
    rows = []
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

    # 创建 DataFrame
    df = pd.DataFrame(rows)
    df['frame_label'] = df['frame_label'].round(1)

    # 筛选 frame_label 为 3.0 且 obj_type 为 0 的行
    df_filtered = df[(df['frame_label'] == 3.0) & (df['obj_type'] == 0)]

    # 创建 vehicles 数据
    vehicles = []
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

    # 运行仿真
    sim = Sim(xodr, Tk(), vehicles, stop_time, show_plot=False)
    sim.Run(show_plot=False)
    output = sim.save_data()
    output['positions'] = output['positions'][1:, :, :]
    output['headings'] = output['headings'][1:, :]

    # 合并仿真输出到原始数据
    predict_mask = data['predict_mask']
    obj_indices = np.where(predict_mask)[0]
    ids_to_concat = [data['ids'][i] for i in obj_indices]

    num_frames_original = data['positions'].shape[1]
    num_frames_output = output['positions'].shape[1]
    start_frame = 31

    for idx, obj_idx in enumerate(obj_indices):
        obj_idx_output = idx
        end_frame = start_frame + num_frames_output

        if end_frame > num_frames_original:
            frames_to_add = num_frames_original - start_frame
            if frames_to_add <= 0:
                continue
            output_positions_truncated = output['positions'][obj_idx_output, :frames_to_add, :]
            output_headings_truncated = output['headings'][obj_idx_output, :frames_to_add]
            valid_frames = np.any(output_positions_truncated != 0, axis=1)
            data['positions'][obj_idx, start_frame:start_frame + frames_to_add][valid_frames] = \
            output_positions_truncated[valid_frames]
            data['headings'][obj_idx, start_frame:start_frame + frames_to_add][valid_frames] = \
            output_headings_truncated[valid_frames]
            data['valid_mask'][obj_idx, start_frame:start_frame + frames_to_add] = valid_frames
        else:
            valid_frames = np.any(output['positions'][obj_idx_output, :, :] != 0, axis=1)
            data['positions'][obj_idx, start_frame:end_frame][valid_frames] = output['positions'][
                obj_idx_output, valid_frames]
            data['headings'][obj_idx, start_frame:end_frame][valid_frames] = output['headings'][
                obj_idx_output, valid_frames]
            data['valid_mask'][obj_idx, start_frame:end_frame] = valid_frames

    # 应用修正逻辑
    print(f"Processing {os.path.basename(folder_path)}")
    data = adjust_positions(data)
    data = smooth_trajectories(data)
    data = interpolate_short_trajectories(data)

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(folder_path) + '_output.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"处理完成，输出保存至: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量处理交通仿真场景')
    parser.add_argument('--input_dir', required=True, help='输入目录路径')
    parser.add_argument('--output_dir', required=True, help='输出目录路径')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for folder_name in os.listdir(args.input_dir):
        folder_path = os.path.join(args.input_dir, folder_name)
        if os.path.isdir(folder_path):
            output_file = os.path.join(args.output_dir, f'{folder_name}_output.pkl')
            if not os.path.exists(output_file):
                print(f'Processing folder: {folder_path}')
                process_folder(folder_path, args.output_dir)
            else:
                print(f'Skipping folder: {folder_path}, {output_file} already exists.')

    # python run_multi.py --input_dir Onsite\第五赛道_B卷 --output_dir Onsite\output-v2
