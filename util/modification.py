import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from util.dynamic_check import calculate_velocity_acceleration
from scipy.interpolate import CubicSpline

def validate_dynamics(data, warmup=31):
    """验证动态约束（修改为接受数据字典而非文件路径）"""
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
    acc_threshold_max = 9.8  # 最大加速度 (m/s^2)
    heading_threshold_min = -0.7  # 最小航向角变化 (rad)
    heading_threshold_max = 0.7  # 最大航向角变化 (rad)

    # 初始化违反标志
    is_acc_violation = False
    is_heading_violation = False
    is_invalid_mask = False
    is_continuity_violation = False
    violation_frames = {}  # 记录每个智能体的超限帧

    # 检查 predict_mask=True 的智能体的约束
    pred_agent_indices = np.where(predict_mask)[0]
    for agent_idx in pred_agent_indices:
        # 检查第32帧（索引31）是否有效
        if not valid_mask[agent_idx][warmup]:
            is_invalid_mask = True

        # 检查 valid_mask 的连续性
        agent_valid_mask = valid_mask[agent_idx]
        first_valid_idx = None
        last_valid_idx = None
        for t in range(len(agent_valid_mask)):
            if agent_valid_mask[t]:
                first_valid_idx = t
                break
        for t in range(len(agent_valid_mask) - 1, -1, -1):
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
        valid_acc = agent_acc[valid_mask[agent_idx]][warmup:]  # 检查 warmup 之后的加速度
        valid_acc_indices = np.where(valid_mask[agent_idx])[0][warmup:]
        if np.any(valid_acc < acc_threshold_min) or np.any(valid_acc > acc_threshold_max):
            is_acc_violation = True
            acc_violation_mask = (valid_acc < acc_threshold_min) | (valid_acc > acc_threshold_max)
            violation_frames[agent_idx] = violation_frames.get(agent_idx, set()).union(
                set(valid_acc_indices[acc_violation_mask]))

        # 检查航向角变化约束
        agent_headings = headings[agent_idx]
        valid_headings = agent_headings[valid_mask[agent_idx]]
        heading_changes = np.diff(valid_headings)
        heading_changes = np.arctan2(np.sin(heading_changes), np.cos(heading_changes))
        valid_heading_indices = np.where(valid_mask[agent_idx])[0][:-1]
        if len(heading_changes) >= warmup:
            heading_changes = heading_changes[warmup - 1:]  # 检查 warmup 之后的航向角变化
            valid_heading_indices = valid_heading_indices[warmup - 1:]
            if np.any(heading_changes < heading_threshold_min) or np.any(heading_changes > heading_threshold_max):
                is_heading_violation = True
                heading_violation_mask = (heading_changes < heading_threshold_min) | (
                            heading_changes > heading_threshold_max)
                violation_frames[agent_idx] = violation_frames.get(agent_idx, set()).union(
                    set(valid_heading_indices[heading_violation_mask] + 1))

    # 计算总违反情况
    total_violation = int(is_acc_violation or is_heading_violation or is_invalid_mask or is_continuity_violation)

    # 返回结果
    return {
        'scene_name': scene_name,
        'is_out_dynamic': total_violation,
        'acceleration_violation': is_acc_violation,
        'heading_violation': is_heading_violation,
        'invalid_mask': is_invalid_mask,
        'continuity_violation': is_continuity_violation,
        'violation_frames': violation_frames  # 返回每个智能体的超限帧
    }


def adjust_positions(input_data):
    """调整位置数据，处理第32帧并保持数据结构一致性"""
    data = input_data.copy()
    positions = data['positions']  # Shape: (n, m, 2)
    new_positions = np.zeros_like(positions)  # 保持原数据类型

    for i in range(positions.shape[0]):
        trajectory = positions[i]  # Shape: (m, 2)
        diff = trajectory[30] - trajectory[31]  # 计算31和32帧的差值
        new_trajectory = np.delete(trajectory, 31, axis=0)  # 删除32帧
        non_zero_mask = np.any(new_trajectory != 0, axis=1)
        for j in range(31, new_trajectory.shape[0]):
            if non_zero_mask[j]:
                new_trajectory[j] += diff
        padded_trajectory = np.pad(
            new_trajectory,
            pad_width=((0, 1), (0, 0)),
            mode='constant',
            constant_values=0
        )
        new_positions[i] = padded_trajectory

    new_data = {'positions': new_positions}
    for key in data.keys():
        if key == 'positions':
            continue
        original_field = data[key]
        if isinstance(original_field, np.ndarray) and original_field.ndim >= 2 and original_field.shape[1] == \
                positions.shape[1]:
            trimmed = np.delete(original_field, 31, axis=1)
            pad_value = 0 if original_field.dtype != bool else False
            pad_width = [(0, 0)] * original_field.ndim
            pad_width[1] = (0, 1)
            padded = np.pad(
                trimmed,
                pad_width=pad_width,
                mode='constant',
                constant_values=pad_value
            )
            new_data[key] = padded.astype(original_field.dtype)
        else:
            new_data[key] = original_field.copy() if isinstance(original_field, np.ndarray) else original_field

    # 验证数据结构一致性
    assert new_data['positions'].dtype == data['positions'].dtype
    assert new_data['valid_mask'].dtype == data['valid_mask'].dtype
    assert new_data['headings'].dtype == data['headings'].dtype
    assert new_data['positions'].shape == data['positions'].shape
    assert new_data['valid_mask'].shape == data['valid_mask'].shape
    return new_data


def interpolate_short_trajectories(data):
    """为第31帧有效但32和33帧缺失的车辆插值补齐32和33帧，确保插值帧满足动态约束"""
    positions = data['positions']  # [Na, Nt, 2]
    headings = data['headings']  # [Na, Nt]
    valid_mask = data['valid_mask']  # [Na, Nt]
    ids = data['ids']  # 智能体 ID 列表
    sim_freq = data['sim_freq']
    dt = 1.0 / sim_freq
    Na, Nt = positions.shape[:2]

    for agent_idx in range(Na):
        if ids[agent_idx] == 'Ego':
            continue

        # 检查第31帧有效且32、33帧无效的情况
        if (valid_mask[agent_idx, 30] and
                not valid_mask[agent_idx, 31] and
                not valid_mask[agent_idx, 32]):

            # 获取前31帧的有效数据
            valid_indices = np.where(valid_mask[agent_idx, :31])[0]
            valid_positions = positions[agent_idx, valid_indices]  # [N_valid, 2]
            valid_headings = headings[agent_idx, valid_indices]  # [N_valid]

            # 情况1：有效帧少于2帧，使用匀速直线运动插值
            if len(valid_indices) < 2:
                # 假设第30帧速度延续
                if valid_mask[agent_idx, 29]:
                    v_prev = (positions[agent_idx, 30] - positions[agent_idx, 29]) / dt
                else:
                    v_prev = np.zeros(2)  # 静止
                # 计算32和33帧位置
                positions[agent_idx, 31] = positions[agent_idx, 30] + v_prev * dt
                positions[agent_idx, 32] = positions[agent_idx, 31] + v_prev * dt
                headings[agent_idx, 31:33] = headings[agent_idx, 30]
                valid_mask[agent_idx, 31:33] = True
                continue

            # 情况2：有足够数据点，使用样条插值
            valid_times = np.array(valid_indices)
            # 确保包含第31帧
            if 30 not in valid_times:
                valid_times = np.append(valid_times, 30)
                valid_positions = np.vstack((valid_positions, positions[agent_idx, 30]))
                valid_headings = np.append(valid_headings, headings[agent_idx, 30])
            else:
                idx_30 = np.where(valid_times == 30)[0][0]
                valid_positions[idx_30] = positions[agent_idx, 30]
                valid_headings[idx_30] = headings[agent_idx, 30]

            # 确保时间严格递增
            sort_indices = np.argsort(valid_times)
            valid_times = valid_times[sort_indices]
            valid_positions = valid_positions[sort_indices]
            valid_headings = valid_headings[sort_indices]

            # 插值位置
            interpwoman: true
            cs_x = CubicSpline(valid_times, valid_positions[:, 0])
            cs_y = CubicSpline(valid_times, valid_positions[:, 1])
            interp_times = np.array([31, 32])
            positions[agent_idx, 31:33, 0] = cs_x(interp_times)
            positions[agent_idx, 31:33, 1] = cs_y(interp_times)

            # 插值航向角
            unwrapped_headings = np.unwrap(valid_headings)
            cs_heading = CubicSpline(valid_times, unwrapped_headings)
            interp_headings = cs_heading(interp_times)
            headings[agent_idx, 31:33] = np.arctan2(np.sin(interp_headings), np.cos(interp_headings))
            valid_mask[agent_idx, 31:33] = True

            # 验证插值结果
            temp_data = data.copy()
            temp_data['positions'] = positions[agent_idx:agent_idx+1, :33, :]
            temp_data['headings'] = headings[agent_idx:agent_idx+1, :33]
            temp_data['valid_mask'] = valid_mask[agent_idx:agent_idx+1, :33]
            temp_data['predict_mask'] = np.array([True])
            temp_data['scene_name'] = data['scene_name']
            temp_data['sim_freq'] = sim_freq

            validation_result = validate_dynamics(temp_data, warmup=31)
            if validation_result['is_out_dynamic']:
                # 如果插值超限，调整为匀速直线运动
                if valid_mask[agent_idx, 29]:
                    v_prev = (positions[agent_idx, 30] - positions[agent_idx, 29]) / dt
                else:
                    v_prev = np.zeros(2)
                positions[agent_idx, 31] = positions[agent_idx, 30] + v_prev * dt
                positions[agent_idx, 32] = positions[agent_idx, 31] + v_prev * dt
                headings[agent_idx, 31:33] = headings[agent_idx, 30]
                valid_mask[agent_idx, 31:33] = True

    return data

def smooth_trajectories(data):
    """使用样条插值平滑轨迹，处理超限数据，保护前31帧"""
    max_iterations = 5     # 最大迭代次数
    extension_frames = 2   # 超限帧前后扩展的帧数
    protected_frames = 31  # 前31帧不可修改

    modified_data = data.copy()
    positions = modified_data['positions']  # [Na, Nt, 2]
    headings = modified_data['headings']  # [Na, Nt]
    valid_mask = modified_data['valid_mask']  # [Na, Nt]
    sim_freq = modified_data['sim_freq']  # 采样频率 (Hz)
    ids = modified_data['ids']  # 智能体 ID 列表

    Na, Nt = positions.shape[:2]
    if headings.shape != (Na, Nt) or valid_mask.shape != (Na, Nt):
        raise ValueError(
            f"Dimension mismatch: positions={positions.shape}, headings={headings.shape}, valid_mask={valid_mask.shape}")

    iteration = 0
    while iteration < max_iterations:
        # 验证动态约束
        result = validate_dynamics(modified_data, warmup=31)

        # 检查是否有超限
        if not result['is_out_dynamic']:
            break

        # 处理超限的智能体
        for agent_idx in result['violation_frames']:
            if ids[agent_idx] == 'Ego':
                continue

            # 获取超限帧
            violation_frames = sorted(list(result['violation_frames'][agent_idx]))
            extended_frames = set()
            for frame in violation_frames:
                for i in range(max(protected_frames, frame - extension_frames), min(Nt, frame + extension_frames + 1)):
                    if valid_mask[agent_idx, i]:
                        extended_frames.add(i)
            extended_frames = sorted(list(extended_frames))

            # 获取有效帧用于插值
            valid_indices = np.where(valid_mask[agent_idx])[0]
            valid_indices = [i for i in valid_indices if i not in extended_frames or i < protected_frames]
            if len(valid_indices) < 2:
                continue

            valid_times = np.array(valid_indices)
            valid_positions = positions[agent_idx, valid_indices]
            cs_x = CubicSpline(valid_times, valid_positions[:, 0])
            cs_y = CubicSpline(valid_times, valid_positions[:, 1])
            for frame in extended_frames:
                if frame >= protected_frames:
                    positions[agent_idx, frame, 0] = cs_x(frame)
                    positions[agent_idx, frame, 1] = cs_y(frame)

            valid_headings = headings[agent_idx, valid_indices]
            valid_headings = np.unwrap(valid_headings)
            cs_heading = CubicSpline(valid_times, valid_headings)
            for frame in extended_frames:
                if frame >= protected_frames:
                    headings[agent_idx, frame] = cs_heading(frame)
                    headings[agent_idx, frame] = np.arctan2(np.sin(headings[agent_idx, frame]),
                                                            np.cos(headings[agent_idx, frame]))

        iteration += 1

    return modified_data

def handle_unresolvable_violations(data):
    """处理无法解决的超限问题，删除超限后的轨迹"""
    modified_data = data.copy()
    positions = modified_data['positions']  # [Na, Nt, 2]
    headings = modified_data['headings']  # [Na, Nt]
    valid_mask = modified_data['valid_mask']  # [Na, Nt]
    ids = modified_data['ids']  # 智能体 ID 列表

    # 验证动态约束
    result = validate_dynamics(modified_data, warmup=31)

    if result['is_out_dynamic']:
        for agent_idx in result['violation_frames']:
            if ids[agent_idx] == 'Ego':
                continue
            violation_frames = sorted(list(result['violation_frames'][agent_idx]))
            earliest_violation_frame = min(violation_frames)
            positions[agent_idx, earliest_violation_frame:, :] = 0.0
            headings[agent_idx, earliest_violation_frame:] = 0.0
            valid_mask[agent_idx, earliest_violation_frame:] = False

    return modified_data