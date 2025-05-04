import pickle
import numpy as np
import csv
import os

def calculate_velocity_acceleration(valid_mask, positions, dt=0.1):
    na, nt, _ = positions.shape
    
    # 初始化标量速度和加速度
    scalar_velocities = np.zeros((na, nt))  # [na, nt]
    scalar_accelerations = np.zeros((na, nt))  # [na, nt]
    
    for i in range(na):
        # 获取有效状态的索引
        valid_indices = np.where(valid_mask[i])[0]
        if len(valid_indices) < 3:
            continue  # 至少需要3个有效状态来计算速度/加速度
        
        # 计算速度向量
        valid_positions = positions[i, valid_indices]  # [num_valid, 2]
        velocity_vectors = (valid_positions[1:] - valid_positions[:-1]) / dt  # [num_valid-1, 2]
        
        # 计算标量速度（大小）
        valid_scalar_velocities = np.linalg.norm(velocity_vectors, axis=1)  # [num_valid-1]
        
        # 填充标量速度：对第一个位置使用第一个速度
        valid_scalar_velocities = np.concatenate(([valid_scalar_velocities[0]], valid_scalar_velocities))
        if len(valid_scalar_velocities) > len(valid_indices):
            valid_scalar_velocities = valid_scalar_velocities[:len(valid_indices)]
        scalar_velocities[i, valid_indices] = valid_scalar_velocities
        
        # 沿运动方向计算加速度
        if len(valid_scalar_velocities) >= 3:
            valid_scalar_accelerations = (valid_scalar_velocities[2:] - valid_scalar_velocities[1:-1]) / dt
            # 填充加速度：将前两个时间步设置为0
            valid_scalar_accelerations = np.concatenate(([0, 0], valid_scalar_accelerations))
            if len(valid_scalar_accelerations) > len(valid_indices):
                valid_scalar_accelerations = valid_scalar_accelerations[:len(valid_indices)]
            scalar_accelerations[i, valid_indices] = valid_scalar_accelerations
    
    return scalar_velocities, scalar_accelerations

def validate_dynamics(pkl_path, warmup=31):
    # 加载pickle文件
    with open(pkl_path, 'rb') as handle:
        data = pickle.load(handle)
    
    # 提取数据
    positions = data['positions']  # [Na, Nt, 2]
    headings = data['headings']  # [Na, Nt]
    valid_mask = data['valid_mask']  # [Na, Nt]
    predict_mask = data['predict_mask']  # [Na]
    scene_name = data['scene_name']
    sim_freq = data['sim_freq']
    dt = 1.0 / sim_freq  # 时间步长

    # 计算速度和加速度
    _, accelerations = calculate_velocity_acceleration(valid_mask, positions, dt)
    
    # 定义运动约束阈值
    acc_threshold_min = -9.8  # 最小加速度 (m/s^2)
    acc_threshold_max = 9.8   # 最大加速度 (m/s^2)
    heading_threshold_min = -0.7  # 最小朝向变化 (rad)
    heading_threshold_max = 0.7   # 最大朝向变化 (rad)
    
    # 初始化违规标志
    is_acc_violation = False
    is_heading_violation = False
    is_invalid_mask = False
    is_continuity_violation = False
    
    # 检查predict_mask=True的代理的约束
    pred_agent_indices = np.where(predict_mask)[0]
    for agent_idx in pred_agent_indices:
        # 检查第32帧（索引31）是否有效
        if not valid_mask[agent_idx][warmup]:
            is_invalid_mask = True
            # print(f"代理 {agent_idx} 在场景 {scene_name} 的第 {warmup} 帧有无效掩码")
        
        # 检查valid_mask的连续性
        agent_valid_mask = valid_mask[agent_idx]
        first_valid_idx = None
        last_valid_idx = None
        for t in range(len(agent_valid_mask)):
            if agent_valid_mask[t]:
                first_valid_idx = t
                break
        for t in range(len(agent_valid_mask)-1, -1, -1):
            if agent_valid_mask[t]:
                last_valid_idx = t
                break
        if first_valid_idx is not None and last_valid_idx is not None:
            for t in range(first_valid_idx, last_valid_idx + 1):
                if not agent_valid_mask[t]:
                    is_continuity_violation = True
                    break
        
        # 检查加速度约束
        agent_acc = accelerations[agent_idx]
        valid_frames = np.where(valid_mask[agent_idx])[0][warmup:]
        valid_acc = agent_acc[valid_mask[agent_idx]][warmup:]  # 检查预热后的帧
        if np.any(valid_acc < acc_threshold_min) or np.any(valid_acc > acc_threshold_max):
            # acc_violation_frames = valid_frames[(valid_acc < acc_threshold_min) | (valid_acc > acc_threshold_max)]
            # violating_acc_values = valid_acc[(valid_acc < acc_threshold_min) | (valid_acc > acc_threshold_max)]
            # print(f"代理 {agent_idx} 在场景 {scene_name} 的加速度违规，帧号: {acc_violation_frames+1}")
            # print(f"违规的加速度值: {[f'{x:.2f}' for x in violating_acc_values]}")
            # print(f"代理 {agent_idx} 的加速度: {[f'{x:.2f}' for x in agent_acc]}")
            # print(f"代理 {agent_idx} 的有效加速度: {[f'{x:.2f}' for x in valid_acc]}")
            is_acc_violation = True
        
        # 检查朝向变化约束
        agent_headings = headings[agent_idx]
        valid_headings = agent_headings[valid_mask[agent_idx]]
        heading_changes = np.diff(valid_headings)
        heading_changes = np.arctan2(np.sin(heading_changes), np.cos(heading_changes))
        heading_changes = heading_changes[warmup-1:]  # 检查预热后的帧
        valid_frames_headings = np.where(valid_mask[agent_idx])[0][1:][warmup-1:]
        if np.any(heading_changes < heading_threshold_min) or np.any(heading_changes > heading_threshold_max):
            # heading_violation_frames = valid_frames_headings[
            # (heading_changes[warmup-1:] < heading_threshold_min) | 
            # (heading_changes[warmup-1:] > heading_threshold_max)]
            # print(f"代理 {agent_idx} 在场景 {scene_name} 的朝向违规，帧号: {heading_violation_frames}")
            # print(f"违规的朝向变化: {heading_changes[warmup-1:][(heading_changes[warmup-1:] < heading_threshold_min) | (heading_changes[warmup-1:] > heading_threshold_max)]}")
            is_heading_violation = True
    
    # 计算总违规
    total_violation = int(is_acc_violation or is_heading_violation or is_invalid_mask or is_continuity_violation)
    
    # 打印此场景的结果
    # print(f"场景: {scene_name}")
    # print(f"加速度违规: {is_acc_violation}")
    # print(f"朝向违规: {is_heading_violation}")
    # print(f"无效掩码: {is_invalid_mask}")
    # print(f"连续性违规: {is_continuity_violation}")
    print(f"总违规: {total_violation}\n")
    
    # 返回此场景的结果
    return {
        'scene_name': scene_name,
        'is_out_dynamic': total_violation,
        'acceleration_violation': is_acc_violation,
        'heading_violation': is_heading_violation,
        'invalid_mask': is_invalid_mask,
        'continuity_violation': is_continuity_violation
    }

def process_pkl_folder(folder_path, output_csv_path='dynamic_constraints.csv'):
    # 初始化结果列表
    all_results = []
    
    # 获取文件夹中所有.pkl文件
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    
    if not pkl_files:
        print(f"在 {folder_path} 中未找到.pkl文件")
        return
    
    # 处理每个.pkl文件
    for pkl_file in pkl_files:
        pkl_path = os.path.join(folder_path, pkl_file)
        # print(f"正在处理 {pkl_path}...")
        try:
            result = validate_dynamics(pkl_path)
            all_results.append(result)
        except Exception as e:
            print(f"处理 {pkl_path} 时出错: {str(e)}")
    
    # 准备CSV数据
    headers = ['场景名称', '加速度违规', '朝向违规',
               '无效掩码', '连续性违规', '总违规']
    csv_data = []
    for result in all_results:
        csv_data.append([
            result['scene_name'],
            int(result['acceleration_violation']),
            int(result['heading_violation']),
            int(result['invalid_mask']),
            int(result['continuity_violation']),
            int(result['is_out_dynamic'])
        ])
    
    # 写入CSV
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(csv_data)
    
    # print(f"结果已保存到 {output_csv_path}")
    return all_results

if __name__ == "__main__":
    # 示例用法
    folder_path = r"Onsite\output-v2"  # 替换为你的文件夹路径
    results = process_pkl_folder(folder_path)