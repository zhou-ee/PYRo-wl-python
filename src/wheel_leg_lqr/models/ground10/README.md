# 十状态固定腿长地面模型

独立建立五坐标能量方程，输出连续时间 `A(10×10)`、`B(10×4)`，并由 `scheduling/ground10_lqr.py` 设计 `K(4×10)`。原十四状态模型继续使用原有入口。

## 状态、输入与配置

坐标为 `q = [x, psi, theta, beta_1, beta_2]`。状态与输入顺序为：

```text
state = [x, dx, psi, dpsi, theta, dtheta, beta_1, dbeta_1, beta_2, dbeta_2]
input = [T_w1, T_w2, T_p1, T_p2]
```

`psi` 为 yaw，`theta` 为 pitch。角度和力矩符号延续现有模型：腿角的物理旋转方向与机体 pitch 相反，正髋力矩对 `theta` 和对应 `beta_i` 的广义力均为正。

整机质量、惯量、几何、质心曲线和前后质心开关读取包根 `config.py`。本目录 [config.py](config.py) 独立维护十状态 Q/R 与采样周期，不读取十四状态控制配置。`L1/L2` 是每个冻结点的外部参数；更换腿长后更新调度，方程内没有 `L_bar`、roll 或其导数，也没有轴向输入。没有新增 RDOB、台阶接触仿真或腾空切换接口。

## 几何假设

采用独立双腿、平均髋高的局部平衡近似。固定腿长且去除 roll 后，如果继续要求两个髋处于同一刚体上、两轮保持严格双接触，还必须约束 `L1*cos(beta_1)-L2*cos(beta_2)` 为常量，两个腿角便不能在任意姿态独立变化。本模型用平均高度近似替代这项闭环约束，因此不能当作原十四状态完整接触力学的精确降阶模型。

两腿长度在对动能、势能和虚功求导前已固定。腿长、横滚支撑及其快速调节由本模型之外的机构或控制环维持；快速变腿长或接触切换不在冻结点验证范围内。

航向坐标系原点位于水平轮中心中点，令：

```text
s_bar = (L1*sin(beta_1) + L2*sin(beta_2))/2
h_bar = (L1*cos(beta_1) + L2*cos(beta_2))/2
wheel_x_i = -s_bar + Li*sin(beta_i)
```

左右符号 `sigma_i = +1/-1`，`a_w/a_h` 分别为轮/髋半轮距。粒子位置为：

```text
wheel_i = [wheel_x_i, sigma_i*a_w, 0]
leg_com_i = [wheel_x_i - L_wp_i*sin(beta_i) + x_com_i*cos(beta_i),
             sigma_i*a_h,
             L_wp_i*cos(beta_i) + x_com_i*sin(beta_i)]
body_com = [-s_bar, 0, h_bar] + R_y(theta)*body_com_from_hip
```

轮高固定在各自局部接触参考面；机体竖直位置取双腿髋高平均。`height` 指轮心以上的平均髋高，不含轮半径。`L_wp_i` 和 `x_com_i` 在 `Li` 处求值后作为常量，前后质心开关关闭时 `x_com_i=0`，沿腿质心调度仍保留。

## 动力学与耦合边界

粒子速度在航向系内为 `v_i = e_x*dx + d(r_i)/dt + dpsi*(e_z × r_i)`。由粒子动能、轮滚动惯量、腿摆杆惯量与无 roll 的机体角速度 `omega_body=[-sin(theta)*dpsi, dtheta, cos(theta)*dpsi]` 建立五维质量矩阵 `M`。

腿摆杆惯量使用 `mp*(Li²/12 + (L_wp_i-Li/2)²)`；沿用该等效杆近似，直接读取完整公共质心曲线精度。势能为各粒子的 `m*g*z` 之和，重力向量为 `g_q = dV/dq`。

轮力矩广义力由 `J_w_i.T*e_x/r_w` 加上对应腿角的反作用力矩得到；髋力矩仅作用于 `theta` 和对应 `beta_i`。由这四列虚功映射建立 `S(5×4)`，没有 FL 列。

静态配平满足 `S(q0)*u0 = g_q(q0)`。令 `H = d(g_q-S*u0)/dq`，先在五维方程求解：

```text
delta_ddq = -solve(M, H)*delta_q + solve(M, S)*delta_u
```

然后按状态顺序交错排列得到 A/B。这里使用势能和输入映射的解析导数，仅围绕零速度静态配平线性化；接口不提供完整的非线性速度项仿真。

以下内容在推导中明确排除：

- 不导入十四状态模型的几何闭环、质量矩阵、A/B 或逆矩阵。
- 不固定 `L_bar` 后继续让单腿长度随腿角变化；本模型的 `dLi/dq` 恒为零。
- 不引入质心曲线的腿长导数、气弹簧刚度、轴向力、roll 支撑反力的反馈项。
- 不把整个系统强行分成互不耦合的通道；俯仰、腿摆、差动偏航之间由上述动能和虚功重新产生的耦合仍保留。

默认机体质心前后坐标为零时，平衡腿角为 `atan(mp*x_com_i/(mb*Li/2+mp*L_wp_i))`。非零机体前后质心采用零 pitch、两髋等分静态力矩的配平约定；求解后检查完整五维残差。导出的平衡腿角为当前物理参数的三次拟合，误差记录在 JSON；NPZ 的 q0/u0 为节点精确配平值。

## 计算入口

```powershell
cd E:\Software\Develop\RM\Wheel_leg\PYRo-wl-python
# 单点 A/B、离散矩阵和 K
python scripts/run_ground10_lqr.py --mode node --l1 0.24 --l2 0.32
# 腿长调度，默认 0.02 m 网格、三次张量积
python scripts/run_ground10_lqr.py --step 0.02
# 重新 editable 安装后也可运行 wheelleg-ground10-lqr
```

```python
from wheel_leg_lqr.models.ground10.model import Ground10Model
from wheel_leg_lqr.scheduling.ground10_lqr import compute_gain_node

plant = Ground10Model().linearize(0.24, 0.32)
node = compute_gain_node(0.24, 0.32)
print(plant.A.shape, plant.B.shape, node.gain.shape)
```

输出至 `outputs/ground10_lqr`：单点结果 `ground10_node.json`；调度结果 `ground10_gain_fit.json`、`ground10_gain_nodes.npz`、`ground10_lqr_coef.h` 和 README。固件命名空间为 `wheel_leg_ground10`，含 `evaluateGain`、`evaluateTrim`；使用 `u=u0-K*(state-trim_state)`。不沿用六输入头文件的数据布局。

## 验证与依赖

`tests/test_ground10.py` 以独立数值速度映射核验动能、有限差分核验势能与线性化；检查曲线导数隔离、roll/轴向参数隔离、前后质心开关、非零机体质心配平和四输入可控性。调度检查网格及单元中心的 float32 闭环稳定性，并对最多 25 个代表节点执行饱和恢复验证；C++ 求值器与 Python 对照。

只依赖公共物理参数、公共质心曲线及 NumPy/SymPy。控制设计、调度与导出在各自组件中，物理模型不读取 Q/R。
