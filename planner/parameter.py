class VehicleTypeParameter:
    def __init__(self):
        # 反应时间：驾驶员的反应时间，单位为秒（s）
        self.reaction_time = 1.0  # s

        # 期望速度：车辆在正常情况下希望达到的速度，单位为米每秒（m/s）
        self.desired_speed = 20.0  # m/s

        # 舒适减速度：车辆在刹车时，舒适的减速度，单位为米每秒平方（m/s^2）
        self.comfort_dec = -2.0  # m/s^2


class Parameters:
    def __init__(self):
        # 创建一个VehicleTypeParameter对象，表示车辆类型的相关参数
        self.vehicle_type_param = VehicleTypeParameter()

        # 仿真步长：模拟过程中每一步的时间间隔，单位为秒（s）
        self.sim_step = 0.1  # s


class IDMParameter:
    def __init__(self):

        # 加速度参数：车辆能够达到的最大加速度，单位为米每秒平方（m/s^2）
        self.a = 1.5

        # 减速度参数：车辆能够达到的最大减速度，单位为米每秒平方（m/s^2）
        self.b = 2.0

        # 最小间距：车辆之间的最小安全距离，单位为米（m）
        self.min_gap = 2.0

        # 时间头距：车辆之间保持的最小时间间隔，单位为秒（s）
        self.time_headway = 1.0

        # 期望速度：车辆希望达到的目标速度，单位为米每秒（m/s）
        self.desired_speed = 16.7

        # ：影响车距计算的参数，决定了跟车时速度对距离的影响程度
        self = 4.0  # 无量纲（常数）

