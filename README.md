<div align="center">
<a href="https://onsite.com.cn/">
    <!-- Please provide path to your logo here -->
    <img src="doc/fig/ONSITE-blue-logo-cn_name.svg" alt="OnSite" width="800">
</a>

# OnSite场景生成赛道——基于规则驱动的自动驾驶测试场景生成模型
</div>

<div align="center">
<a href="https://onsite.com.cn/"><img src="https://img.shields.io/badge/OnSite-3.0-blue"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://tops.tongji.edu.cn/"><img src="https://img.shields.io/badge/TCU-TOPS-purple"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="./LICENSE"><img src="https://img.shields.io/badge/LICENSE-BSD%203-yellow"></a>
</div>

## 目录
* [1 环境配置](#jump1)
* [2 数据准备](#jump2)
* [3 文件说明](#jump3)
* [4 运行测试](#jump4)
* [5 核心算法](#jump5)
* [6 致谢](#jump6)
* [7 更改日志](#jump7)

## <span id="jump1">1 环境配置
+ 克隆存储库
```
git clone https://github.com/(后补充)
cd Traffic_Sim
```
+ 建立虚拟环境 *（可选conda/venv）*，需指定版本为 **Python 3.6.8**
```
conda create -n onsite python=3.6.8
conda activate onsite
```
+ 安装核心依赖
```
pip install -r requirements.txt
```
> **备注：** 为了兼容性，您可以尝试不同的python版本，例如，Python 3.9.12已被确认是可以正常运行工作。

## <span id="jump2">2 数据准备
* 下载[Onsite场景生成赛道赛题](https://pan.baidu.com/s/16twMhQg13O2et2mdYLsfAg?pwd=egwp)（以场景生成赛道的样例场景为例）。下载并解压缩zip文件后，请按以下方式组织数据集目录：
```
/path/to/Onsite/
└── A/
    ├── 0_140_straight_straight_141/
    │   ├── 0_140_straight_straight_141_exam.pkl      # 场景参数文件
    │   └── 0_140_straight_straight_141.xodr          # OpenDRIVE路网文件
    ├── 0_1049_merge_1066/
    │   ├── 0_1049_merge_1066_exam.pkl
    │   └── 0_1049_merge_1066.xodr
    ├── ...（其他场景目录）
```

## <span id="jump3">3 文件说明
### 3.1 文件结构说明
```
Traffic_Sim
├── run_multi_task.py
├── simulator.py
├── requirements.txt
├── README.md
├── vehicles
│   ├── init.py
│   ├── D1Vehciles.py
│   └── vehcile.py
├── roads
│   ├── init.py
│   ├── D1xodrRoads.py
│   └── road.py
├── planner
│   ├── __init__.py
│   ├── parameter.py
│   ├── net_param.mat
│   └── planner.py
├── Onsite
├── paramset
├── utils
└── doc
```
### 3.2 文件功能说明

>   | 文件名                 | 功能          |
>   |:----------:|:--------------:|
>  | run_mutli_task.py    | 仿真场景生成测试主程序|
>   | simulator.py         | 仿真场景生成模块   |
>   | README.md            | 测试工具说明文档   |
>   | requirement          | Python环境依赖 |

### 3.3 文件夹功能说明

>   | 文件夹名称    | 功能           |
>   |:----------:|:--------------:|
>   | vehicles | 仿真场景生成车辆设置     |
>   | roads    | 仿真场景生成路网设置       |
>   | planner  | 仿真场景生成规控模块         |
>   | Onsite   | Onsite场景生成赛道赛题 |
>   | paramset | 仿真场景生成相关参数设置 |
>   | utils    | 仿真场景生成模块使用的组件和工具 |
>   | doc      | 资源文件夹，用于存放图片等 |

## <span id="jump4">4 运行测试
+ 单个交通仿真场景生成测试：
```
python ./run_single.py
```
+ 批量交通仿真场景生成测试：
```
python ./run_multi.py
```

## <span id="jump5">5 核心算法
车辆在路网中的行为大致可以划分为跟驰、换道、合流、分流、冲突等交互模块，而其中合流、分流、冲突都可以将其视为跟驰行为的特例进行考虑。因此下面将主要从跟驰、换道两大模型对车辆的微观交互行为进行描述。

### 5.1 跟驰模型
#### 5.1.1 一般路段跟驰模型
跟驰模型也就车辆不发生换道，仅根据前车的运动状态信息（位置、速度等）来控制本车的行动。 本文的跟驰模型采用**智能驾驶员模型（IDM模型）** ，它以一个统一的公式，可以描述交通状态从拥堵到自由的车辆跟驰情况。

$$\dot{v}=a\left[1-\left(\frac{v}{v_0}\right)^\delta-\left(\frac{s^*(v,\Delta v)}{s}\right)^2\right]$$

$$s^{*}(v, \Delta v)=s_{0}+\max \left(0, s_{1} \sqrt{\frac{v}{v_{0}}}+Tv+\frac{v * \Delta v}{2 \sqrt{a b}}\right)$$

其中， 
$v$：本车加速度；
$v_0$：期望速度；
$∆v$：本车与前车速度之差；
$s$：与前车距离(车头到车尾);
$s^*$：与前车的期望距离；
$T$：反应时间；
$a$：起步加速度；
$b$：舒适减速度；
$δ$：加速度指数；
$s_0$：静止安全距离；
$s_1$：与速度有关的安全距离选择参数。

#### 5.1.2 合流下的跟驰模型
主要分两种情况进行讨论：

* **与合流前车距离大于 $10m$：** 车辆以 $5m/s$的车速接近合流点，如果需要减速，减速度不超过舒适减速度。

* **与合流前车距离小于 $10m$：** 如果计算的合流距离小于合流前车车长，车辆以最大加速度减速；否则车辆按照跟驰模型的输入参数计算减速度。

#### 5.1.3 分流下的跟驰模型
按照确定分流前车时确定的参数，输入跟驰模型计算减速度。

#### 5.1.4 交叉口内冲突避让模型
对于每一个冲突点，如果相交连接器上有冲突前车，就假设该冲突点有一辆静止的车辆，计算跟驰模型的加速度。

#### 5.1.5 强制性换道条件下的跟驰
对于具有强制性换道动机但由于间隙不足没有完成换道的车辆（简称为“具有强制换道动机车辆”），不能让车辆一直向下游行驶，否则很有可能直到路径末端也没有合适的间隙，从而无法按照规定的路径行驶，因此车辆在路径末端需要停车等待以寻找合适的间隙。

另一方面，如果不同车道所有车辆等待间隙的停车点都在同一个地方，那距离目标车道相隔远的车辆因为目标车道上也有等待汇入的车辆很可能就没有办法汇入造成拥堵。因此，距离期望车道越远的车道上的车辆的停车点应该更加位于上游的地方。在仿真中，我们设置的停车点到路径路段末端距离为 $相隔车道数×30m$，

以快速路为例，车辆必须要完成换道的位置距离入匝道始端分别 $60m/30m/0m$ （**如下图所示**）。

![Alt text](doc\fig\强制性换道场景1.png "强制性换道场景1")

还有一种更极端的情况，即两辆车分别要换道到对方车道上（**如图下所示**，最左侧车道上的直行车辆和中间车道的右转车辆分别要向对方车道变道），但由于间隙始终不足，以上述的跟驰逻辑，两车会一直沿着原车道行驶并在相同的位置停车，阻塞道路通行。

![Alt text](doc\fig\强制性换道场景2.png "强制性换道场景2")

因此在考虑强制换道停车点时，将车辆所处车道属性也考虑进来： $停车点到路径路段末端距离=相隔车道数×30m+当前车道在路段中左数序号×20$

#### 5.1.6 信号灯反应模型
当车辆行驶靠近交叉口进口道停车线时，根据控制该方向进口道的信号灯灯色不同（红/黄/绿）将会有不同的决策行为，具体说明如下：

* **当车辆进入信号灯**（信号灯的位置放置在进口道路段末端）上游 $50m$ 范围内，才对信号灯做出反应；
* **当信号灯灯色为绿灯时**，车辆采用一般化的跟驰模型即可；
* **当信号灯灯色为绿灯时**，只有路段最下游的第一辆车（也就是距离信号灯最近的到达车辆）会对信号灯反应，而后面的车辆是跟随前车进行跟驰的。此时可以将停车线假想为一辆静止的车辆（车尾与停车线平齐）
* **当信号灯灯色变为黄色时**，路段下游第一辆车需要判断以当前速度是否可以通过信号灯，如果不可以通过，则默认在黄灯后放置一静止的虚拟车辆，使车辆减速（一般这种情况车辆离黄灯会比较远，这样不会出现车辆突然减速的情况），后续车辆跟驰排队即可；如果计算结果可以通过交叉口则继续以一般路段的跟驰模型跟随前车行驶。

#### 5.1.7 换道状态下的跟驰
换道状态下的跟驰是指车辆在换道过程中，需要同时考虑对原车道前车以及目标车道前车的跟驰，另一方面也会对原车道和目标车道后车产生影响。目前的处理逻辑是将换道过程分为三个阶段，其中阶段Ⅰ认为换道车辆还处于原车道，仅对原车道前车做跟驰，同时只影响原车道后车；阶段Ⅱ认为换道车辆同时对原车道前车和目标车道前车做跟驰，并且同时影响原车道后车和目标车道后车；阶段Ⅲ车辆位于目标车道，仅对目标车道前车做跟驰，同时只影响目标车道后车。具体使用时，根据换道车辆行驶过的距离占换道轨迹长度的比例进行划分，将轨迹前 $\frac{1}{5}$ 作为阶段Ⅰ，中间 $\frac{2}{5}$ 作为阶段Ⅱ，最后 $\frac{2}{5}$ 作为阶段Ⅲ。

#### 5.1.8 对智能车的避让
智能车在交通流环境中相当于一个移动的障碍物，由于其驾驶算法受外部控制，我们无法预先得知智能车的跟驰/换道决策行为，因此也无法简单地用传统的一位交通流模型进行避让（比如智能车的换道轨迹是未知的，就无法确定智能汽车对周车影响的三阶段划分方法）。因此对智能车的避让是将其看车一个有速度的障碍物进行避让的。
具体方法如下：
* 由于仿真中将智能车车体近似看作一个矩形，首先获取智能车车身顶点四个点，以及相邻顶点的中点，从而粗略地描述出智能车车身的 $8$个关键点；
* 计算本车与智能车的距离，如果距离小于 $100m$，则需要考虑上述车身 $8$个点到本车航向角射线的距离，如果距离小于 $车宽/2$，需要对其做避障处理；
* 计算 $AV$的运动在本车行车方向的分量，对其计算跟驰加速度。


### 5.2 换道模型
换道模型描述的是驾驶员根据自身需求，针对周围车辆的运动状态信息，变换车道的过程。换道相比跟驰更加复杂，以至于难以用单一的模型来描述，通常来说，现有的换道行为建模大多将换道模型根据车辆换道的过程解耦为四个阶段：**动机生成-车道选择 -间隙选择-换道执行**。由于换道模型涉及的变量很多（包括本车道以及相邻两车道的前后车信息），为简化模型形式同时保证仿真精度，下列模型适用于中低流量下的仿真，对于高密度交通流、复杂道路无论是涉及的变量、模型形式、规则数量都需要做相应的增加和调整。根据换道目的不同，还可以将换道模型划分为两类：**强制性换道（mandatory lane changing, MLC）** 和 **任意性换道（discretionary lane changing, DLC）**。
#### 5.2.1 强制性换道
**（一）换道动机模型** 

根据强制性换道的定义，其换道动机来源于当前车道无法行驶到目的地。将车辆必须要执行强制换道的位置称之为强制换道点，比如入匝道处的强制换道点就是加速车道末端、左转车的强制换道点就是进口道画实线处。通常，驾驶员在接近强制换道点时才会产生换道动机。根据实证数据统计，将强制性换道的换道动机定义为，距离强制换道点小于 $150m$。

**（二）车道选择模型** 

强制性换道的目标车道是固定的。强制性换道的车辆会向着接近期望车道的方向换道。

**（三）间隙选择模型** 

![Alt text](doc\fig\换道间隙场景.png "换道间隙场景")

间隙选择模型就是判断当前目标车道上对应的间隙是否满足安全插入的条件，该问题目前使用最多的是临界间隙模型：也就是根据换道车辆与目标车道前后车的相对运动状态计算出可接受间隙阈值，如果实际间隙大于该阈值就执行换道，否则继续在原车道行驶。具体选择的是对数线性临界间隙模型，形式如下：

$$G_g=\beta_gX+\varepsilon_g$$

式中， $G_g$为驾驶员接受的最小间隙；由于间隙选择时需要前方间隙和后方间隙都满足阈值条件，因此需要计算两个临界阈值， $g=[lead,lag]$； $\beta_g$为接受间隙的列向量变量，具体变量含义见**表1**所示； $X_i$为标定的 $\beta_g$的对应行向量系数； $\varepsilon_g$为随机误差项，假设其负责正态分布，均值为 $0$。标定后的系数见**表2**所示。

> **表1 $\beta_g$列向量具体内容**
> | 参数 | $\beta_g^0$ |  $\beta_g^1$  |  $\beta_g^2$  | $\beta_g^3$ |    $\beta_g^4$    |
> |:-:|:-:|:-:|:-:|:-:|:-:|
> |说明|常量| $max(0,∆V_g)$ | $min(0,∆V_g)$ |$V_g$ | $1-exp(\alpha d)$ |
**注：** 如果 $g=lead$， $∆V_g=V_{sub}-V_{lead}$；如果 $g=lag$， $∆V_g=V_{lag}-V_{sub}$， $V_{sub}$为本车。 $d$为到强制换道点的距离， $\alpha$为距离影响系数。

> **表2 线性间隙选择模型系数**
> |             $g$             | $lead$ | $lag$ |
> |:-:|:-:|:-:|
> |    最小间距$\beta^0$（单位：$m$）    |  1.00  | 1.50  |
> |   $\beta^1$系数值（单位：$/mph$）   |  0.15  |  0.1  |
> |   $\beta^2$系数值（单位：$/mph$）   |  0.30  | 0.35  |
> |   $\beta^3$系数值（单位：$/mph$）   |  0.20  | 0.25  |
> |   $\beta^4$系数值（单位：$/mph$）   |  0.10  | 0.10  |
> | 随机项$\varepsilon$标准差（单位：$m$） |  1.00  | 1.50  |
> |       $\alpha$距离影响系数        | 0.008  |
**注：** 单位转换关系， $1mph=0.44704m/s$。

**（四）换道执行模型**

换道执行模型主要是为换道车辆规划换道轨迹，即在原车道换道起点和目标车道换道终点之间规划一条连续平滑的曲线。常用的拟合曲线包括正弦曲线、圆弧曲线、横向加速度正反梯形法、贝塞尔曲线、多项式曲线等。考虑到实际路网中曲线方程在旋转变化后的易解析性和计算成本，选择采用三阶贝塞尔曲线：

$$\begin{cases}
B_x(q)=P_x^s(1-q)^3+3P_x^{C1}q(1-q)^2+3P_x^{C2}q^2(1-q)+P_x^eq^3 \\
B_y(q)=P_y^s(1-q)^3+3P_y^{C1}q(1-q)^2+3P_y^{C2}q^2(1-q)+P_y^eq^3 & 
\end{cases}$$

其中， $B$是生成的贝塞尔曲线上的点， $P^s/P^e$分别是贝塞尔曲线的起点和终点， $P^{C1}/P^{C2}$分别是三阶贝塞尔曲线的两个控制点，下标 $x/y$分别表示上述点的横纵坐标，当 $q$在 $0$~ $1$之间逐渐变化时，就在起终点之间生成了一条贝塞尔曲线。

贝塞尔曲线的起点 $P^s$就是车辆判断间隙可接受时所处的位置。由于常态交通流模型都假设车辆是沿着车道中心线行驶的，因此终点 $P^e$位于目标车道的中心线上，具体位置根据车辆的速度计算换道长度 $LC_{Dist}$获得：

$$LC_Dist=
\begin{cases}
2*L_{veh^{\prime}} & \quad ifV_{sub}\leq20km/h \\
3*L_{veh^{\prime}} & \quad if20km/h<V_{sub}\leq30km/h \\
4*L_{veh^{\prime}} & \quad if30km/h<V_{sub}\leq40km/h \\
5*L_{veh^{\prime}} & \quad ifV_{sub}<50km/h & 
\end{cases}$$

在操作时，可首先从起点位置 $P^{s}$沿着车道线向下游延长 $LC_Dist$的长度得到点 $P^{e^\prime}$，再从 $P^{e^\prime}$向目标车道中心线做垂线，垂点位置就是贝塞尔曲线的终点 $P^{e}$，**如下图所示**。

![Alt text](doc\fig\换道终点示意图.png "换道终点示意图")

控制点 $P^{C1}/P^{C2}$的计算方法如下：首先分别以贝塞尔曲线起点 $P^s$向切线方向、终点 $P^e$向切线反方向绘制两条射线（**下图a**中蓝色箭头射线）；随后连接换道起终点（**下图a**中绿色实线），并在连线上取三分点（**下图a**中绿色实心点）。以连线三分点分别向两个射线方向做垂线，垂点（**下图a**中橙色实心点）即为控制点 $P^{C1}/P^{C2}$的位置。根据这两个控制点生成的贝塞尔曲线**下如图b**所示。

![Alt text](doc\fig\基于控制点的换道轨迹生成示意图.png "基于控制点的换道轨迹生成示意图")

#### 5.2.2 任意性换道
**（一）换道动机模型**

实际交通流环境中为获取更好的驾驶条件，主要的任意性换道动机可以分为三种情况：

* 车速未达期望车速。本车车速低于本车期望速度（或车道限速） $20km/h$以下，且前车速度低于本车速度 $20%$以上，目标车道前车速度高于本车道前车速度 $10km/h$以上，且目标车道前车间距大于当前车道前车间距 $2$倍车长;
* 不希望跟随大型车辆。本车道前方车辆为大型车辆（大货车，大客车），且与前方大车的距离小于 $2$倍时距，目标车道前车速度高于本车道前车速度，且目标车道前车间距大于当前车道前车间距 $2$倍车长。
* 寻找最佳的排队位置。本车速度大于前车速度、且前车速度大于等于前前车速度，同时目标车道前车距离大于本车道前车距离 $2$倍车长以上。

**（二）车道选择模型**

任意性换道的目标车道是根据周围的交通状态动态变化的，假设只有位于期望车道上的车辆才会产生任意性换道动机，且任意性换道只会在期望车道之间进行换道。因此任意性换道可能有两个目标车道（即两侧车道均为期望车道），也可能只有一个目标车道（即一侧车道为期望车道），甚至没有目标车道（即仅有当前车道为期望车道，此时无法做出任意性换道）。

**（三）间隙选择模型**

任意性换道的间隙选择模型和强制性换道形式上基本不变，有两点小的改变：一是调整了模型系数，二是将距离强制换道点的影响项去除。最终的效果是，相比于强制性换道，任意性换道对于可接受间隙的要求更高。

> **表3  任意性间隙选择模型系数**
> |$g$|$lead$|$lag$|
> | :-: | :-: | :-: |
> |    最小间距$\beta^0$（单位：$m$）    |  1.00  | 1.50  |
> |   $\beta^1$系数值（单位：$/mph$）   |  0.2  |  0.15  |
> |   $\beta^2$系数值（单位：$/mph$）   |  0.35  | 0.45  |
> |   $\beta^3$系数值（单位：$/mph$）   |  0.25  | 0.25  |
> |   $\beta^4$系数值（单位：$/mph$）   |  —  | —  |
> | 随机项$\varepsilon$标准差（单位：$m$） |  1.00  | 1.50  |
**注：** 单位转换关系， $1mph=0.44704m/s$。

**（四）换道执行模型**

模型与强制性换道完全一致。

## <span id="jump6">6 致谢
衷心感谢[TOPS课题组](https://tops.tongji.edu.cn/index.htm)的集体努力与卓越贡献。

## <span id="jump7">7 更改日志
### [Ver 3.1.1] - 2025.02.17
* **首次上传：** 本项目首次上传至GitHub，包含基础功能和核心代码。
* **环境配置：** 提供了详细的Python环境配置步骤和依赖安装指导。
* **数据准备：** 提供了Onsite场景生成赛道样例赛题的下载链接和数据组织方式。
* **运行测试：** 提供了运行测试脚本的详细步骤。
* **文件说明：** 对项目的文件结构和功能进行了详细的说明。
