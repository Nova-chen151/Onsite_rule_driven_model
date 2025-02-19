class Field:
    # 构造函数，初始化字段的各个属性
    def __init__(self, one_record):
        # 设置 connector_id 属性，将 one_record 中第一个元素转换为整数
        self.connector_id = int(one_record[0])
        # 设置 type 属性，将 one_record 中第二个元素转换为整数
        self.type = int(one_record[1])
        # 根据 type 的值判断是否为 'car' 或 'bike'
        self.type = 'car' if self.type == 1 else 'bike'
        # 设置 start_lane_number 属性，将 one_record 中第三个元素转换为整数
        self.start_lane_number = int(one_record[2])
        # 设置 end_lane_number 属性，将 one_record 中第四个元素转换为整数
        self.end_lane_number = int(one_record[3])
        # 设置 start_end1_x 属性，将 one_record 中第五个元素转换为浮点数
        self.start_end1_x = float(one_record[4])
        # 设置 start_end1_y 属性，将 one_record 中第六个元素转换为浮点数
        self.start_end1_y = float(one_record[5])
        # 设置 start_end1_yaw 属性，将 one_record 中第七个元素转换为浮点数
        self.start_end1_yaw = float(one_record[6])
        # 设置 start_end2_x 属性，将 one_record 中第八个元素转换为浮点数
        self.start_end2_x = float(one_record[7])
        # 设置 start_end2_y 属性，将 one_record 中第九个元素转换为浮点数
        self.start_end2_y = float(one_record[8])
        # 设置 start_end2_yaw 属性，将 one_record 中第十个元素转换为浮点数
        self.start_end2_yaw = float(one_record[9])
        # 设置 target_end1_x 属性，将 one_record 中第十一个元素转换为浮点数
        self.target_end1_x = float(one_record[10])
        # 设置 target_end1_y 属性，将 one_record 中第十二个元素转换为浮点数
        self.target_end1_y = float(one_record[11])
        # 设置 target_end1_yaw 属性，将 one_record 中第十三个元素转换为浮点数
        self.target_end1_yaw = float(one_record[12])
        # 设置 target_end2_x 属性，将 one_record 中第十四个元素转换为浮点数
        self.target_end2_x = float(one_record[13])
        # 设置 target_end2_y 属性，将 one_record 中第十五个元素转换为浮点数
        self.target_end2_y = float(one_record[14])
        # 设置 target_end2_yaw 属性，将 one_record 中第十六个元素转换为浮点数
        self.target_end2_yaw = float(one_record[15])

class ConnectorSetting:
    # 构造函数，初始化连接器的设置
    def __init__(self, one_record):
        # 设置 connector_id 属性，将 one_record 中第一个元素转换为整数
        self.connector_id = int(one_record[0])
        # 设置 connector_type 属性，将 one_record 中第二个元素转换为整数
        connector_type = int(one_record[1])
        # 根据 connector_type 的值判断是否为 'car' 或 'bike'
        self.car_type = 'car' if connector_type == 1 else 'bike'
        # 设置 start_section_id 属性，将 one_record 中第三个元素转换为整数
        self.start_section_id = int(one_record[2])
        # 设置 start_lane_id 属性，将 one_record 中第四个元素按空格分割并转换为整数列表
        self.start_lane_id = [int(lane_id) for lane_id in one_record[3].split()]
        # 设置 end_section_id 属性，将 one_record 中第五个元素转换为整数
        self.end_section_id = int(one_record[4])
        # 设置 end_lane_id 属性，将 one_record 中第六个元素按空格分割并转换为整数列表
        self.end_lane_id = [int(lane_id) for lane_id in one_record[5].split()]
        # 设置 direction 属性，表示连接器的方向
        self.direction = one_record[6]

class SectionSetting:
    # 构造函数，初始化路段的设置
    def __init__(self, one_record):
        # 设置 section_id 属性，将 one_record 中第一个元素转换为整数
        self.section_id = int(one_record[0])
        # 设置 section_type 属性，将 one_record 中第二个元素转换为整数
        section_type = int(one_record[1])
        # 根据 section_type 的值判断是否为 'car' 或 'bike'
        self.car_type = 'car' if section_type == 1 else 'bike'
        # 设置 lane_number 属性，将 one_record 中第三个元素转换为整数
        self.lane_number = int(one_record[2])
        # 设置 end1_x 属性，将 one_record 中第四个元素转换为浮点数
        self.end1_x = float(one_record[3])
        # 设置 end1_y 属性，将 one_record 中第五个元素转换为浮点数
        self.end1_y = float(one_record[4])
        # 设置 end1_yaw 属性，将 one_record 中第六个元素转换为浮点数
        self.end1_yaw = float(one_record[5])
        # 设置 end2_x 属性，将 one_record 中第七个元素转换为浮点数
        self.end2_x = float(one_record[6])
        # 设置 end2_y 属性，将 one_record 中第八个元素转换为浮点数
        self.end2_y = float(one_record[7])
        # 设置 end2_yaw 属性，将 one_record 中第九个元素转换为浮点数
        self.end2_yaw = float(one_record[8])

class SignalSetting:
    # 构造函数，初始化信号灯的设置
    def __init__(self, one_record):
        # 设置 id 属性，将 one_record 中第一个元素转换为整数
        self.id = int(one_record[0])
        # 设置 red_time 属性，将 one_record 中第二个元素转换为整数
        red_time = int(one_record[1])
        # 设置 green_time 属性，将 one_record 中第三个元素转换为整数
        green_time = int(one_record[2])
        # 设置 yellow_time 属性，将 one_record 中第四个元素转换为整数
        yellow_time = int(one_record[3])
        # 创建一个列表表示信号灯的时间安排，包含红灯、绿灯、黄灯的时长
        self.schedule = list([red_time, green_time, yellow_time])
        # 设置 offset 属性，将 one_record 中第五个元素转换为浮点数
        self.offset = float(one_record[4])

class SignalToConnectorSetting:
    # 构造函数，初始化信号灯与连接器的设置
    def __init__(self, one_record):
        # 设置 connector_id 属性，将 one_record 中第一个元素转换为整数
        self.connector_id = int(one_record[0])
        # 设置 signal_id 属性，将 one_record 中第二个元素转换为整数
        self.signal_id = int(one_record[1])
