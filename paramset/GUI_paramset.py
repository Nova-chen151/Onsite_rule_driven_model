# -*- coding: utf-8 -*-
from tkinter import *
# from Tkinter import *
# 导入ttk
from tkinter import ttk
import tkinter as tk

class VEH_params:
    def __init__(self):
        self.length = 0
        self.width = 0
        self.net_dist = 0
        self.reac_time = 0
        self.desired_speed = 0
        self.max_acc = 0
        self.comfort_dec = 0

    def get_value_lst(self):
        return [self.max_acc, self.comfort_dec, self.net_dist, self.desired_speed, self.reac_time, self.length, self.width]

class VEHs_params:
    def __init__(self):
        self.params_truck_agsv = VEH_params()
        float_params([3.0, 2.0, 0.5, 18, 0.8, 8.5, 2.8], self.params_truck_agsv)
        self.params_truck_norm = VEH_params()
        float_params([2.8, 1.8, 0.8, 15, 1.0, 8.5, 2.8], self.params_truck_norm)
        self.params_truck_cnsv = VEH_params()
        float_params([2.5, 1.5, 1.0, 13, 1.2, 8.5, 2.8], self.params_truck_cnsv)
        self.params_car_agsv = VEH_params()
        float_params([4.0, 3.0, 0.5, 20, 0.8, 4.5, 2.4], self.params_car_agsv)
        self.params_car_norm = VEH_params()
        float_params([3.8, 2.5, 0.8, 18, 1.0, 4.5, 2.4], self.params_car_norm)
        self.params_car_cnsv = VEH_params()
        float_params([3.5, 2.0, 1.0, 16, 1.2, 4.5, 2.4], self.params_car_cnsv)
        # return [self.params_truck_agsv, self.params_truck_norm, self.params_truck_cnsv, self.params_car_agsv, self.params_car_norm, self.params_car_cnsv]

class App:
    def __init__(self, path_set, root):
        self.master = root
        root.title("Grid布局")
        root.title('参数设置')
        # root.iconbitmap(r'tops_2.ico')
        root.attributes("-alpha", 0.95)
        self.initWidgets(path_set)
        root.mainloop() #注释：用户界面不显示


    def initWidgets(self, path_set):
        init_root = self.master

        # 创建一个输入组件
        p = Frame(self.master)
        # 定义字符串的元组：需求模型
        p1 =  Label(init_root, text='· 需求模型'  , font=('华文细黑', 14, 'bold')).grid(row=0, column=0, padx=3, pady=8,)  # 对Label内容进行 表格式 布局
        p2 =  Label(init_root, text='起点路段：', font=('华文细黑', 11)).grid(row=1, column=0, padx=0, pady=3)
        p3 =  Label(init_root, text='终点路段：', font=('华文细黑', 11)).grid(row=2, column=0, padx=0, pady=3)
        p4 =  Label(init_root, text=' 流  量 ：', font=('华文细黑',  11)).grid(row=3, column=0, padx=0, pady=3)
        p6 =  Label(init_root, text='(veh/h)', font=('华文细黑', 10)).grid(row=3, column=2, padx=0, pady=3, sticky='W')
        p7 = Label(init_root, text='·路径列表：', font=('华文细黑',  12)).grid(row=6, column=0, padx=0, pady=0)

        #仿真参数
        Label(init_root, text='· 仿真参数', font=('华文细黑', 14, 'bold')).grid(row=0, column=3, padx=15, pady=8)
        Label(init_root, text='仿真时间：', font=('华文细黑', 11)).grid(row=1, column=3, padx=15, pady=0)
        Label(init_root, text='(s)', font=('华文细黑', 11)).grid(row=1, column=5, padx=0, pady=1, sticky='W')
        Label(init_root, text='大车比例：', font=('华文细黑', 11)).grid(row=1, column=6, padx=15, pady=8)
        Label(init_root, text='激进驾驶员比例：', font=('华文细黑', 11)).grid(row=2, column=3, padx=15, pady=8)
        Label(init_root, text='保守驾驶员比例：', font=('华文细黑', 11)).grid(row=2, column=5, columnspan=2, padx=15, pady=8)

        #车辆模型
        p8 = Label(init_root, text='· 车辆模型', font=('华文细黑', 14, 'bold')).grid(row=3, column=3, padx=15, pady=8)  # 对Label内容进行 表格式 布局
        p9 =  Label(init_root, text='·驾驶员类型', font=('华文细黑',  12)).grid(row=4, column=3, padx=15, pady=0, sticky='W')
        p99 = Label(init_root, text='·车辆类型', font=('华文细黑', 12)).grid(row=4, column=6, padx=15, pady=0, sticky='W')

        p10 =  Label(init_root, text='·跟驰模型参数', font=('华文细黑',  12)).grid(row=6, column=3, padx=15, pady=0, sticky='W')
        p11 =  Label(init_root, text='最大加速度：', font=('华文细黑',  11)).grid(row=7, column=3, padx=15, pady=0)
        p11 =  Label(init_root, text='(m/s2)', font=('华文细黑',  10)).grid(row=7, column=5, padx=0, pady=1, sticky='W')
        p13 = Label(init_root, text='舒适减速度：', font=('华文细黑',  11)).grid(row=8, column=3, padx=15, pady=8)
        p14 = Label(init_root, text='(m/s2)', font=('华文细黑',  10)).grid(row=8, column=5, padx=0, pady=1, sticky='W')
        p15 = Label(init_root, text='停 车 间 距 ：', font=('华文细黑',  11)).grid(row=9, column=3, padx=15, pady=8)
        p16 = Label(init_root, text='(m)', font=('华文细黑',  10)).grid(row=9, column=5, padx=0, pady=1, sticky='W')
        p17 = Label(init_root, text='期 望 速 度 ：', font=('华文细黑',  11)).grid(row=10, column=3, padx=15, pady=8)
        p18 = Label(init_root, text='(km/h)', font=('华文细黑',  10)).grid(row=10, column=5, padx=0, pady=1, sticky='W')
        p19 = Label(init_root, text='反 应 时 间 ：', font=('华文细黑',  11)).grid(row=11, column=3, padx=15, pady=8)
        p20 = Label(init_root, text='(s)', font=('华文细黑',  10)).grid(row=11, column=5, padx=0, pady=1, sticky='W')

        p21 = Label(init_root, text='·车辆参数', font=('华文细黑',  12)).grid(row=6, column=6, padx=15, pady=0, sticky='w')
        p22 = Label(init_root, text='车身长度：', font=('华文细黑',  11)).grid(row=7, column=6, padx=15, pady=8, sticky='w')
        p23 = Label(init_root, text='(m)', font=('华文细黑',  10)).grid(row=7, column=8, padx=0, pady=1, sticky='W')
        p24 = Label(init_root, text='车身宽度：', font=('华文细黑',  11)).grid(row=8, column=6, padx=15, pady=8, sticky='w')
        p25 = Label(init_root, text='(m)', font=('华文细黑',  10)).grid(row=8, column=8, padx=0, pady=1, sticky='W')

        # 设置变量
        e1 =  StringVar(init_root)
        e2 =  StringVar(init_root)
        e3 =  StringVar(init_root)
        e4 =  StringVar(init_root)
        e5 =  StringVar(init_root)
        e6 =  StringVar(init_root)
        e7 =  StringVar(init_root)

        self.sim_duration = StringVar(init_root, value='1000')
        self.truck_rate = StringVar(init_root, value='0.1')
        self.driver_agsv = StringVar(init_root, value='0.1')
        self.driver_cnsv = StringVar(init_root, value='0.1')

        Olink =  Entry(init_root, width=8, bd=2, textvariable=StringVar())
        Olink.grid(row=1, column=1, padx=0, pady=8)
        Dlink =  Entry(init_root, width=8, bd=2, textvariable=StringVar())
        Dlink.grid(row=2, column=1, padx=0, pady=0)
        flow =  Entry(init_root, width=8, bd=2, textvariable=StringVar())
        flow.grid(row=3, column=1, padx=0, pady=0)
        self.od_params = [Olink, Dlink, flow]

        max_acc =  Entry(init_root, width=8, bd=2, textvariable=e1)
        max_acc.grid(row=7, column=4, padx=0, pady=0)
        comf_dec =  Entry(init_root, width=8, bd=2, textvariable=e2)
        comf_dec.grid(row=8, column=4, padx=0, pady=0)
        stop_gap =  Entry(init_root, width=8, bd=2, textvariable=e3)
        stop_gap.grid(row=9, column=4, padx=0, pady=0)
        des_speed =  Entry(init_root, width=8, bd=2, textvariable=e4)
        des_speed.grid(row=10, column=4, padx=0, pady=0)
        reac_time =  Entry(init_root, width=8, bd=2, textvariable=e5)
        reac_time.grid(row=11, column=4, padx=0, pady=1)
        veh_length =  Entry(init_root, width=8, bd=2, textvariable=e6)
        veh_length.grid(row=7, column=7, padx=0, pady=1)
        veh_width =  Entry(init_root, width=8, bd=2, textvariable=e7)
        veh_width.grid(row=8, column=7, padx=0, pady=0)
        self.veh_params = [e1, e2, e3, e4, e5, e6, e7] #[max_acc, comf_dec, stop_gap, des_speed, reac_time, veh_length, veh_width]
        self.params = VEHs_params()

        e6 = Entry(init_root, width=8, bd=2, textvariable=self.sim_duration).grid(row=1, column=4, padx=0, pady=0)
        e7 = Entry(init_root, width=8, bd=2, textvariable=self.truck_rate).grid(row=1, column=7, padx=0, pady=0)
        e8 = Entry(init_root, width=8, bd=2, textvariable=self.driver_agsv).grid(row=2, column=4, padx=0, pady=0)
        e9 = Entry(init_root, width=8, bd=2, textvariable=self.driver_cnsv).grid(row=2, column=7, padx=0, pady=0)

        # 创建Combobox组件
        self.cb1 = ttk.Combobox(self.master, textvariable=StringVar(),postcommand=self.param_set)  # 当用户单击下拉箭头时触发self.set方法
        self.cb1.grid(row=5, column=3, columnspan=3, padx=0, pady=5)
        self.cb1['values'] = ['激进型', '稳健型', '保守型']# 为Combobox配置多个选项
        self.cb1.bind("<<ComboboxSelected>>", self.param_select)
        self.cb2 = ttk.Combobox(self.master, textvariable=StringVar(),postcommand=self.param_set)  # 当用户单击下拉箭头时触发self.set方法
        self.cb2.grid(row=5, column=6, columnspan=3, padx=0, pady=5)
        self.cb2['values'] = ['大型车', '小轿车']  # 为Combobox配置多个选项
        self.cb2.bind("<<ComboboxSelected>>", self.param_select)

        #OD矩阵
        self.od_matrix=[]
        topF = Frame(self.master)
        topF.grid(row=7, rowspan=6, column=0, columnspan=3, padx=0, pady=5)
        Label(topF, text='      O               D      FLOW     '    , font=('华文细黑', 11)).pack(side=TOP, fill=Y, expand=YES, anchor=W)
        # 定义StringVar变量
        self.v = StringVar()
        # 创建Listbox组件，与v变量绑定
        self.lb = Listbox(topF, listvariable = self.v)
        self.lb.pack(side=LEFT, fill=BOTH, expand=YES, anchor=W)#.grid(row=7, rowspan=4, column=0, columnspan=3, sticky='E', padx=5, pady=5)
        # 创建Scrollbar组件，设置该组件与self.lb的纵向滚动关联
        scroll = Scrollbar(topF, command=self.lb.yview)
        scroll.pack(side=RIGHT, fill=Y)
        # scrollx = Scrollbar(topF, command=self.lb.xview, orient='horizontal')
        # scrollx.grid(side=BOTTOM, fill=X)
        # 设置self.lb的纵向滚动影响scroll滚动条
        self.lb.configure(yscrollcommand=scroll.set)
        # self.lb.configure(xscrollcommand=scrollx.set)
        if path_set:
            self.add_default_path(path_set)

        #选择是否在交叉口内部使用二维仿真
        self.d2_on = IntVar()
        c18 = Checkbutton(init_root, text='交叉口内部二维', variable=self.d2_on, onvalue=1, offvalue=0).grid(row=10, column=7, columnspan=1)

        # 按钮
        Button(init_root, text='开始仿真', width=11, command=self.quit).grid(row=11, column=6, columnspan=2, sticky=E, padx=15, pady=5)
        botton1 = Button(init_root, text='添加', font=('华文细黑', 10), width=10, command=self.supp).grid(row=4, column=0, padx=5, pady=5)
        botton2 = Button(init_root, text='删除', font=('华文细黑', 10), width=10, command=self.dele).grid(row=4, column=1, padx=5, pady=5)

    def show(self):
        pass
        # print("仿真时间 :%s" % e1.get())
        # print("小汽车平均速度 :%s" % e4.get())
        # print("小汽车最大速度 :%s" % e5.get())
        # print("公交车平均速度 :%s" % e7.get())
        # print("公交车最大速度 :%s" % e8.get())
        # print("小汽车比例 :%s" % e10.get())
        # print("公交车比例 :%s" % e11.get())
        # print("冒险型比例 :%s" % e13.get())
        # print("常规型比例 :%s" % e14.get())
        # print("保守型比例 :%s" % e15.get())
        # print("压迫换道比例 :%s" % e17.get())
        # print("18 %s" % var18.get())
        # print("19 %s" % var19.get())

    def add_default_path(self, path_set):
        for path in path_set:
            od_lst = []
            od_string = ' '
            for od_param in path:
                od_lst.append(od_param)
                od_string = od_string + str(od_param) + '   '
            self.od_matrix.append(od_lst)
            self.lb.insert(END, od_string)

    def supp(self):
        od_lst=[]
        od_string=' '
        for od_param in self.od_params:
            od_lst.append(od_param.get())
            od_string = od_string + str(od_param.get()) + '   '
        self.od_matrix.append(od_lst)
        self.lb.insert(END, od_string)

    def dele(self):
        for each in self.lb.curselection()[::-1]:
            del self.od_matrix[each]
            self.lb.delete(each)

    def param_select(self, *args):
        # current = self.cb1.current()
        if self.cb1.get() == '激进型' and self.cb2.get() == '大型车': #[max_acc, comf_dec, stop_gap, des_speed, reac_time, veh_length, veh_width]
            veh_params = self.params.params_truck_agsv.get_value_lst()
        elif self.cb1.get() == '激进型' and self.cb2.get() == '小轿车':
            veh_params = self.params.params_car_agsv.get_value_lst()
        elif self.cb1.get() == '稳健型' and self.cb2.get() == '大型车':
            veh_params = self.params.params_truck_norm.get_value_lst()
        elif self.cb1.get() == '稳健型' and self.cb2.get() == '小轿车':
            veh_params = self.params.params_car_norm.get_value_lst()
        elif self.cb1.get() == '保守型' and self.cb2.get() == '大型车':
            veh_params = self.params.params_truck_cnsv.get_value_lst()
        elif self.cb1.get() == '保守型' and self.cb2.get() == '小轿车':
            veh_params = self.params.params_car_cnsv.get_value_lst()
        else:
            return

        for i in range(len(veh_params)):
            self.veh_params[i].set(str(veh_params[i]))

    def param_set(self):
        if self.cb1.get() == '激进型' and self.cb2.get() == '大型车':
            float_params(self.veh_params, self.params.params_truck_agsv)
        elif self.cb1.get() == '激进型' and self.cb2.get() == '小轿车':
            float_params(self.veh_params, self.params.params_car_agsv)
        elif self.cb1.get() == '稳健型' and self.cb2.get() == '大型车':
            float_params(self.veh_params, self.params.params_truck_norm)
        elif self.cb1.get() == '稳健型' and self.cb2.get() == '小轿车':
            float_params(self.veh_params, self.params.params_car_norm)
        elif self.cb1.get() == '保守型' and self.cb2.get() == '大型车':
            float_params(self.veh_params, self.params.params_truck_cnsv)
        elif self.cb1.get() == '保守型' and self.cb2.get() == '小轿车':
            float_params(self.veh_params, self.params.params_car_cnsv)
        else:
            return

    def quit(self):
        self.master.quit()
        # self.params = VEH_params()
        # self.float_params(self.veh_params)
        # return [self.od_matrix, self.params, float(self.sim_duration.get()), float(self.truck_rate.get())]

def float_params(veh_params, veh_type):
    if isinstance(veh_params[0],StringVar): #保存文本框数据
        veh_type.max_acc = float(veh_params[0].get())
        veh_type.comfort_dec = float(veh_params[1].get())
        veh_type.net_dist = float(veh_params[2].get())
        veh_type.desired_speed = float(veh_params[3].get())
        veh_type.reac_time = float(veh_params[4].get())
        veh_type.length = float(veh_params[5].get())
        veh_type.width = float(veh_params[6].get())
    else: #初始化参数
        veh_type.max_acc = veh_params[0]
        veh_type.comfort_dec = veh_params[1]
        veh_type.net_dist = veh_params[2]
        veh_type.desired_speed = veh_params[3]
        veh_type.reac_time = veh_params[4]
        veh_type.length = veh_params[5]
        veh_type.width = veh_params[6]

if __name__ == "__main__":
    root = tk.Tk()
    # root.title("界面演示")
    path_set = [[-20, -20, 500]]
    App(path_set, root)
    root.mainloop()