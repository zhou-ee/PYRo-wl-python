# 六阶输入等效降阶 DOB：推导与上位机接口

本实现使用六维辅助状态 `z`，观测 Tw1、Tw2、Tp1、Tp2、FL1、FL2 的输入等效扰动，不再建立十四维预测状态。LQR 仍使用已有状态测量。观测六路与补偿哪些通道是两个独立选择。

本阶段提供单点设计、独立 L1/L2 二维系数调度、连续参考方程、离散更新、上位机调度步进及 C++ 求值头文件。默认覆盖 `[0.18,0.38]² m²`，不固定平均腿长。尚未接入下位机，冻结点检查不构成实际时变补偿闭环稳定性证明。

## 1. 固定配平点的连续推导

令 `v = delta(qdot)`，所有量均在同一个静态配平点取增量：

$$M_0\dot v+D_0v+K_0\delta q=S_0(\delta u+d_u).$$

输入顺序为 `(Tw1, Tw2, Tp1, Tp2, FL1, FL2)`，前四项单位 N·m，后两项单位 N。定义

$$S_0^\dagger=(S_0^TS_0)^{-1}S_0^T,\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_6),\quad \lambda_i>0.$$

列满秩意味着 $S_0^\dagger S_0=I_6$。实现使用 SVD 计算等价左伪逆，避免直接求逆正规方程；若在给定 SVD 容差下失去列满秩，则拒绝设计。

从希望实现的滤波关系出发：

$$\dot{\hat d}_u=\Lambda\left[
S_0^\dagger(M_0\dot v+D_0v+K_0\delta q)-\delta u-\hat d_u
\right].$$

定义

$$\Gamma=\Lambda S_0^\dagger M_0,\qquad z=\hat d_u-\Gamma v.$$

因为配平点矩阵固定，$\dot z=\dot{\hat d}_u-\Gamma\dot v$，加速度项严格相消：

$$\boxed{\dot z=-\Lambda(z+\Gamma v)
+\Lambda S_0^\dagger(D_0v+K_0\delta q)-\Lambda\delta u}$$

$$\boxed{\hat d_u=z+\Gamma v}.$$

供代码预计算的形式为

$$B_q=\Lambda S_0^\dagger K_0,\qquad
B_v=\Lambda S_0^\dagger D_0-\Lambda\Gamma,$$

$$\dot z=-\Lambda z+B_q\delta q+B_vv-\Lambda\delta u.$$

注意矩阵乘法顺序：一般不能交换 $\Lambda$ 与其他矩阵。接口 `rates_rad_s` 接受 $\lambda_i$；命令行带宽单位为 Hz，内部换算 $\lambda_i=2\pi f_i$。

代回真实动力学得到 $\dot{\hat d}_u=\Lambda(d_u-\hat d_u)$。令 $e=d_u-\hat d_u$：

$$\dot e=\dot d_u-\Lambda e.$$

恒定扰动时 $e_i(t)=e_i(0)e^{-\lambda_i t}$；缓变扰动有跟踪误差。该结论要求模型准确、状态无噪声且输入时序正确，不是补偿闭环的鲁棒性证明。

## 2. 非匹配扰动能屏蔽到什么程度

若动力学还包含任意广义外力 $f\in\mathbb R^7$：

$$M_0\dot v+D_0v+K_0\delta q=S_0(\delta u+d_u)+f,$$

则观测器实际满足

$$\dot{\hat d}_u=\Lambda(d_u+S_0^\dagger f-\hat d_u).$$

定义 $P_\perp=I_7-S_0S_0^\dagger$，则

$$f=S_0S_0^\dagger f+P_\perp f,\qquad S_0^\dagger P_\perp f=0.$$

因此左伪逆严格屏蔽的是 $\ker(S_0^T)$ 中的正交广义力分量。该空间在此模型中为一维，但通常不是某一个固定坐标轴。任意“不完全匹配”的外力仍可能有非零输入空间投影，并会折算为 $S_0^\dagger f$。屏蔽只指它不进入理想扰动估计，并不表示它对机器人运动没有影响。

这里的正交性取决于给定的广义力坐标及其尺度；若改变混合力/力矩坐标的缩放，一般会改变非匹配力的投影。当前实现遵循题设的欧氏左伪逆，没有引入加权度量。

## 3. 离散实现：使用两端状态，不测加速度

采用对连续 `z` 方程的梯形积分。设 $h$ 为采样周期，输入 $\delta u_k$ 在区间 $[t_k,t_{k+1})$ 内保持，使用两端的 $\delta q_k,v_k,\delta q_{k+1},v_{k+1}$：

$$R=(I_6+\tfrac h2\Lambda)^{-1},\qquad
A_z=R(I_6-\tfrac h2\Lambda),$$

$$C_q=\tfrac h2RB_q,\quad C_v=\tfrac h2RB_v,\quad C_u=-hR\Lambda,$$

$$\boxed{z_{k+1}=A_zz_k+C_q(\delta q_k+\delta q_{k+1})
+C_v(v_k+v_{k+1})+C_u\delta u_k}$$

$$\hat d_{u,k+1}=z_{k+1}+\Gamma v_{k+1}.$$

`step_poles` 和 `step_input_gain` 是对角矩阵的六个对角元素，运行时逐元素相乘。若区间内输入有已知变化，应分段更新或使用与实际积分一致的输入处理，不能填入下一周期尚未施加的指令。

辅助状态的离散极点为 $(1-h\lambda_i/2)/(1+h\lambda_i/2)$，对有限正数在单位圆内。$h\lambda_i>2$ 会出现负极点，说明数值响应逐采样交替；即使数值稳定，也不代表带宽选择合理。实际设计应使采样能充分解析观测动态。

这是连续观测器的二阶数值实现，不是机器人与观测器联合系统的精确 ZOH 离散化，不能把上述辅助极点直接当作所有实际采样估计误差的精确极点。测试用独立的精确植物离散化验证了步长收敛性。

核心观测器只有六维 `z`；上一个测量点的 q/v 缓存和外部补偿滤波器还需要额外存储。没有显式加速度求导也不意味着免疫测量噪声：$\Gamma v$ 会引入速度噪声，`z + Gamma*v` 的抵消还涉及 float32 精度。

## 4. 配平、气弹簧与时序约定

- 广义坐标顺序：`x, psi, theta, phi, l_bar, beta_1, beta_2`。从现有十四状态提取 `q = state[0::2]`、`v = state[1::2]`；位置要减 `trim_position`。静态配平速度为零。角度增量应在同一局部坐标中保持连续。
- 当前 `core.model.linearize_static_trim` 没有速度阻尼模型，因此默认 `D0=0`。API 可传完整 7×7 阻尼矩阵，CLI 可传七项对角阻尼。默认值不代表实际阻尼为零，未建模部分可能进入等效扰动。
- 观测器输入是实际施加的总广义输入增量，包含补偿、限幅与执行映射的影响；不能使用未经限幅的 LQR 原始输出。
- 模型中的 `trim_total_input` 包含总支撑力。若获取的是主动电机折算腿力，应加当前气弹簧模型力后，再减总配平力。导出的 `trim_active_input` 仅作对照，不能将主动/总力混用。气弹簧模型残差会进入扰动估计。
- 有当前测量时先用当前 `z` 重构扰动并计算控制。在下一次测量到来后，使用上一区间的实际输入调用 `step_rdob`，得到与新测量同一时刻的 `z` 和扰动。
- 初次启动用 `z0 = initial_d - Gamma*v0`；若希望零扰动初始化，即 `z0=-Gamma*v0`。不能在非零速度下简单将 `z0` 清零。
- 底层 `step_rdob` 没有默认泄漏、估计限幅或补偿滤波；部署参考接口 `step_scheduled_rdob` 默认使用与当前固件 LESO 相同的 `0.1×` 执行器上限，也可显式传 `disturbance_limits=None` 关闭。限幅在端点重基准化之前完成，始终维护 `z = d_hat - Gamma*(v-v0)` 的一致性。后续补偿可设置 Tw/Tp/FL 独立增益，不要把补偿增益乘入观测器动力学。

若在线改变 $\Gamma$，连续推导会多出 $-\dot\Gamma v$；变化配平点也有坐标导数项。单点公式只适用于固定模型。二维调度步进采用第 6 节的分段固定模型和端点重基准化，不把新旧配平坐标直接混入一次固定模型更新。

## 5. 使用方式

```powershell
python run_rdob.py --step 0.02 --fit-tolerance 0.005 --channel-tolerance 0.01 --sample-time 0.001 --output-dir data/generated/rdob
```

安装项目后也可使用 `wheelleg-rdob`，参数相同。默认值按 RDOB 的梯形离散公式反算，使六路离散扰动极点均为 `0.992`，与当前下位机 LESO 的扰动极点相同；这只对齐扰动估计的名义收敛极点，不表示两种观测器的完整瞬态会一致。

默认生成二维调度的 `rdob_schedule.npz`、`rdob_schedule.json`、`rdob_schedule.h`、`rdob_schedule_report.md`。JSON/NPZ 包含 Gamma 的三次全张量直接系数、Bq 的总阶三次直接系数以及全部训练与留出参数点。

需要单点诊断时显式选择：

```powershell
python run_rdob.py --mode node --l1 0.24 --l2 0.32 --output-dir data/generated/rdob_node
```

单点模式生成：

- `rdob.npz`：双精度模型、连续与离散系数、配平值。
- `rdob.json`：同一组系数、单位、顺序、输入约定与数值诊断。
- `rdob_coefficients.h`：C++11 `float` 系数与更新公式注释，供后续下位机实现。
- `rdob_report.md`：当前配平点和使用限制。

Python 接口：

```python
from wheel_leg_lqr.rdob import compute_rdob_node
from wheel_leg_lqr.control.rdob import initialize_rdob, reconstruct_disturbance, step_rdob

node = compute_rdob_node(0.28, 0.28)
observer = node.observer
# q_prev/v_prev 是增量；u_prev 是上一区间实际施加的总输入增量。
z = initialize_rdob(observer, v_prev)
d_hat = reconstruct_disturbance(observer, z, v_prev)
z, d_hat = step_rdob(observer, z, q_prev, v_prev, u_prev, q_now, v_now)
```

数值测试覆盖：加速度项消去、六通道独立性、正交非匹配力与混合外力的区别、恒定扰动收敛、输入时序与补偿符号、实际模型对称/不对称配平点、采样收敛阶、初始化、非法参数、导出精度、二维域覆盖、调度切换重基准化及 C++ float32 求值一致性。

## 6. 独立 L1/L2 二维调度

参数为真实左右腿长，而不是固定平均腿长后的长度差：

$$L_1,L_2\in[0.18,0.38],\qquad \bar L=(L_1+L_2)/2.$$

每个节点调用原有 `make_trim_geometry(L1,L2)`，重新计算两腿配平摆角、几何偏移、质量矩阵、刚度矩阵和输入映射。原版单点默认值 0.28 m 只是一个例子，从未构成 deltaL 调度；新版默认入口建立完整的二维调度。

RDOB 只拟合两个完整密集对象：`Gamma` 使用三次全张量直接多项式，`Bq` 使用总阶三次直接多项式。`q0` 与 `u0_total` 由既有 LQR 调度提供，RDOB 在步进调用中使用上一区间的 LQR 配平值。由这些量重构

$$B_v=\Lambda S_0^\dagger D_0-\Lambda\Gamma,$$

再按相同固定带宽与采样周期计算离散系数，避免分别拟合相关矩阵而破坏这一代数关系。`rates/step_poles/step_endpoint_weight/step_input_gain/step_velocity_scale` 全部按六通道数组导出，即使当前六路数值相同也不折叠成标量；`D0=0` 时整个阻尼增益矩阵严格为零，速度端点项使用 `step_velocity_scale[i]*Gamma[i][j]`。非零阻尼保留完整计算路径；生成器若发现零阻尼假设与拟合矩阵不一致，会拒绝导出。

六个离散极点、六个输入更新系数、六个端点积分权重和六个 Gamma 速度缩放系数在导出时按 float32 顺序预计算，因此每次二维求值不再重复进行这部分除法。

L1、L2 各自映射到 `[-1,1]`；状态和输入保留 SI 坐标。Gamma 的每个元素使用 $0\le i,j\le3$ 的 16 项张量积幂基，Bq 的每个非零元素使用 $i+j\le3$ 的 10 项幂基。运行时以嵌套 Horner 求值，不在线求 SVD、伪逆或三角函数。

默认网格步长 0.02 m，有 121 个训练点；留出集含 100 个单元中点、40 个边界中点和固定种子的 24 个随机点。搜索过程复用同一批动力学快照，不为每个候选重新建模。

默认每个拟合矩阵最大相对 Frobenius 误差预算为 `0.005=0.5%`，且必须在模拟 C++ 顺序的 float32 求值中通过。这里的百分比定义为 `norm(M_fit-M)/norm(M)`，不是逐元素百分比；零值附近的元素不能用逐元素相对误差衡量。对重构 `Gamma*M0^-1*S0` 相对于 `Lambda` 的输入通道映射误差使用独立的 `0.01=1%` 上限。

默认物理参数下，Gamma 的 float32 最大矩阵误差约为 0.01990%，Bq 约为 0.20939%，最大输入通道映射误差约为 0.51837%。结果与当前参数有关，以新生成的报告为准。

继续检查平均腿长配平值，以及精确局部 LQR 配合 RDOB 的冻结点离散闭环极点。闭环分别检查 Tw+FL 全增益补偿与全部六路全增益补偿，不包含饱和、噪声、延迟和接触切换；它也不代替旧 LQR 拟合系数的联合验证，矩阵误差上限也不等于实际闭环状态误差上限。

当前没有强制左右交换对称性，因接口允许不相等的左右带宽和一般阻尼矩阵。完整域内两种腿长顺序均参与验证。生成头文件是纯数据契约；固件自行实现系数求值，并负责拒绝非有限或越界腿长，不能静默外推。

上位机运行时接口：

```python
from wheel_leg_lqr.scheduling.rdob_schedule import compute_rdob_schedule, step_scheduled_rdob
from wheel_leg_lqr.control.rdob import initialize_rdob

schedule = compute_rdob_schedule()  # 离线设计一次
previous_lengths = (0.24, 0.26)
coefficients = schedule.evaluate(*previous_lengths)
z = initialize_rdob(coefficients, measured_velocity)
# LQR 同时给出 previous_trim_position=q0 和 previous_trim_total_input=u0。
# position/velocity 为物理测量；applied_total_input 为刚结束区间的实际总输入。
z, d_hat = step_scheduled_rdob(
    schedule, z, previous_lengths, next_lengths,
    previous_position, previous_velocity, applied_total_input,
    next_position, next_velocity,
    previous_trim_position, previous_trim_total_input,
)
```

这个步进先用上一区间的 `L1/L2` 系数及配平点完成梯形更新，两个位置端点都减同一个旧 `q0`，输入也减旧 `u0`；然后在端点计算新系数，并设置

$$z_{new}=\hat d_{old}-\Gamma_{new}v_{endpoint}.$$

这使扰动重构值在系数切换瞬间保持连续，避免把坐标变化直接解释成扰动。它仍是分段固定的局部模型近似，不表示任意快速变腿长都稳定。C++ `rdob_schedule.h` 只提供腿长范围、六通道离散常量及 `Gamma/Bq` 拟合表，不生成 `Coefficients` 或 `evaluate()`；下位机自行求值时必须遵循相同的元素顺序、端点时序和重基准化。

报告列出直接多项式项集、训练/留出误差、float32 误差、闭环裕度、存储和运算上界。运行时物化矩阵用于端点更新与 Gamma 重基准化，计数包含这一部分；目标硬件耗时尚需实测。

本页方程是按题设矩阵模型推导。降阶 DOB 的一般背景可参见 [Kim & Rew, Automatica (2013)](https://doi.org/10.1016/j.automatica.2013.01.014)。
