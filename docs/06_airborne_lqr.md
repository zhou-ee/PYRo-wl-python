# 06 腾空双髋小模型 LQR：腿角与机体 Pitch

## 目标与接口

腾空或跳跃离地期间，两个轮子都没有可靠地面约束。本控制器只用 `T_p1, T_p2` 两路髋关节力矩，控制与十四状态模型同号的 `beta_1, beta_2, theta`（`theta` 即机体 pitch）。每个角度包含角速度，因此实际离散状态是六维：

| 小模型索引 | 状态 | 十四状态索引 |
|---:|---|---:|
| 0, 1 | `beta_1, dbeta_1` | 10, 11 |
| 2, 3 | `beta_2, dbeta_2` | 12, 13 |
| 4, 5 | `theta, dtheta` | 4, 5 |

输入按 `T_p1, T_p2` 排列，单位 N·m。令 `e=x_air-x_ref`，控制律是 `T_p = -K(L1,L2)e`，输出分别限幅到现有 `INPUT_LOWER_LIMITS[2:4]` 和 `INPUT_UPPER_LIMITS[2:4]`。离地切换应由接触检测决定，重新触地后再切回地面十四状态 LQR；切换瞬间应保持参考角和力矩连续。

## 无接触动力学

本模型的物理参数在 `airborne/config.py` 独立保存。初值来自当前 CAD/整机参数集，包含机体质量和俯仰惯量、轮/腿质量、机体和腿质心拟合；代码不导入地面十四状态模型的配置、质量矩阵、重力项、配平结果或 `A/B/K`。腿的局部俯仰惯量采用杆件惯量与 CAD 质心偏移组成的估计。两腿长度在一个冻结调度点视为常量。把髋中心的自由水平、竖直平移加入动能，再消去自由平移，得到三角度惯量：

\[
M_a=M_{rr}-M_{rt}M_{tt}^{-1}M_{tr},\qquad
y=[\beta_1,\beta_2,\theta]^T.
\]

均匀重力只加速整机质心，在自由落体坐标中不产生对这些角度的回复刚度。轮子不接触地面，故不用滚动约束和轮电机转动惯量；`T_w` 与 `F_L` 也不进入小模型。围绕零角度、零角速度的线性化为

\[
\ddot y=M_a^{-1}G T_p,\qquad
G=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix}.
\]

`G` 由两处髋关节内力矩独立推得，符号约定与现有状态接口一致：`beta` 的正向与机体正 pitch 的物理旋转方向相反，因此 `T_p` 在两个广义坐标上同号。这里保留的是实际髋部惯性耦合，不含地面滚动或接触耦合。

### 机体质心前偏时的 `F_L → pitch` 通道

原双髋 LQR 保持 `L1/L2` 冻结，输入仍只有 `T_p1/T_p2`。为分析伸腿力的扰动，独立模型另外开放 `q_f=[beta1,beta2,theta,L1,L2]` 五个相对坐标，把腿长和 CAD 腿质心关于腿长的导数计入动能，再消去自由平移。正 `F_Li` 定义为沿腿轴伸腿：下端受远离髋中心的力，机体在髋中心受反向力。分析量是飞行期间的**净虚拟轴向力变化**，不能把地面支撑配平力直接当作腾空恒定输入。

若机体质心相对髋中心向前 `c_x`、向上 `c_z`，髋中心处的正 `F_Li` 对机体质心产生的 pitch 力矩臂为

\[
\ell_i=c_x\cos(\theta+\beta_i)+c_z\sin(\theta+\beta_i),
\qquad \tau_{\theta,\mathrm{body}}=\ell_1 F_{L1}+\ell_2 F_{L2}.
\]

这里只看机体本身的力矩时，`ddot(theta)=tau_body/I_yy`。整机自由悬浮时，两腿和轮也会反向运动；在零初始广义速度下，瞬时角加速度应从独立的五坐标质量矩阵求出：

\[
\ddot q_f=M_f^{-1}
\begin{bmatrix}0&0\\0&0\\0&0\\1&0\\0&1\end{bmatrix}
\begin{bmatrix}F_{L1}\\F_{L2}\end{bmatrix}.
\]

在 `L1=L2=0.28 m`、`beta1=beta2=theta=0`、机体质心向前 `c_x=0.03 m`、向上 `c_z=0.032 m` 时：

| 量 | `F_L1` 单路每 1 N | 两路各 100 N |
|---|---:|---:|
| 机体自身 pitch 力矩 | 0.03 N·m | 6 N·m |
| 只按机体 `I_yy=0.582892610 kg·m²` 计算 | 0.0514675 rad/s² | 10.2935 rad/s² |
| 整机无接触五坐标模型计算 | **0.0689856 rad/s²** | **13.7971 rad/s²**（约 790.5°/s²） |

若机体质心居中，五坐标模型仍有每路约 `0.0035948 rad/s²/N` 的小耦合，来自腿组件的非共线质心。将机体质心前移 3 cm 后，每路系数增加约 `0.0653908 rad/s²/N`。在上述对称姿态，`F_L1=+F, F_L2=-F` 的 pitch 影响相互抵消；不等腿长或腿角时需分别用当前系数计算。`F_L` 仍是内力，整体角动量守恒；它通过腿长自由度与机体姿态的惯性耦合改变 pitch。

复现计算：

```powershell
python run_airborne_fl_pitch.py --body-com-x 0.03 --l1 0.28 --l2 0.28 --fl1 100 --fl2 100
```

此结果是当前姿态和腿长处的**瞬时**线性输入映射，不代表 100 N 可持续施加，也不包含电机电流、腿长行程、气弹簧或接触约束。要求某个持续时间内的 pitch 变化时，还需给出 `F_L(t)`、腿长控制及行程限制并积分变参数动力学。

## 可控性与控制目标

两路内力矩不能任意独立保持三个绝对角度。整体转动方向 `v=[-1,-1,1]^T` 满足 `v^T G=0`，总俯仰角动量 `v^T M_a \dot y` 守恒。六状态系统可控秩为 4；完整离散闭环始终有两个单位圆上的整体角度/角速度极点。这个限制不能通过调大 Q 或直接对六状态调用 LQR 消除。

可控髋部形状为 `r=[theta+beta_1, theta+beta_2]^T`。程序用质量矩阵求形状到物理角度的映射 `R_s`，满足 `G^T R_s=I` 与 `v^T M_a R_s=0`，随后在四状态 `r_1, dr_1, r_2, dr_2` 上做离散 LQR。腾空权重与地面十四状态权重独立，当前 `Q_DIAG=[1200,6,1200,6,400,25]`（六状态顺序见上表），`R_DIAG=[0.1,0.1]`。它们定义在 `airborne/config.py`，也可从命令行覆盖。

回投到六维物理状态时，采用 Q 加权的可达投影：`H=(R_s^T Q_angle R_s)^(-1)R_s^T Q_angle`，`K_6=K_4 diag(H,H)`（按交错状态排列）。这样在角动量约束下优先减小加权角度误差，包含 pitch 反馈。若腾空时已有非零总俯仰角动量，或目标三角度与当前守恒量不相容，三个绝对角度无法同时收敛到目标；控制器只稳定四个可控形状模态。长时间腾空时还要考虑腿角行程及力矩饱和。

### 用 Q 选择姿态取舍

`Q_angle=diag(q_beta1,q_beta2,q_theta)` 同时进入 LQR 代价和可达投影。因此增大 `q_theta/q_beta` 会允许腿角偏离目标，换取更小的 pitch 误差；增大 `q_beta/q_theta` 则更偏向腿角回零。`dbeta/dtheta` 权重主要改变运动速度和阻尼，`R` 权重主要改变力矩使用与响应快慢。若总角动量为零且初始姿态本来满足目标，三个角度可维持目标；从不相容的初始姿态出发时，Q 决定可达的折中终态。

例如固定 `L1=L2=0.28 m`、初始 `theta=10°`、`beta1=beta2=0°`、初始角速度为零，用线性模型计算无饱和的可达终态：

| Q 位置权重 `(beta1,beta2,theta)` | 终态腿角（左右相同） | 终态 pitch |
|---|---:|---:|
| `(120,120,400)` 示例基线 | −3.59° | 9.16° |
| `(120,120,4000)` 更偏向 pitch | −20.40° | 5.20° |
| `(1200,1200,400)` 当前默认，更偏向腿角 | −0.39° | 9.91° |

这是冻结线性模型的取舍示例，不包含腿角行程限制。较高的 pitch 权重会要求更大摆腿幅度，应结合机构行程与 `T_p` 限幅选择。

## 拟合与导出

从项目根目录运行：

```powershell
python run_airborne_lqr.py --step 0.01 --output-dir data/generated/airborne_lqr
```

例如更偏向 pitch，可在不修改地面 LQR 权重的情况下运行：

```powershell
python run_airborne_lqr.py --step 0.01 --q-diag 120 6 120 6 4000 25 --output-dir data/generated/airborne_lqr_pitch
```

`--q-diag` 按六状态顺序接收六个正数，也可用 `--r-diag RTP1 RTP2` 覆盖两个力矩惩罚。每次改权重都要重新计算并导出系数；JSON/NPZ/头文件均记录所用的 Q/R 对角线。腾空物理参数若变更，只需更新 `airborne/config.py` 并重生成；更改地面 LQR 配置不会改变腾空增益。

安装包后可用 `wheelleg-airborne-lqr`。默认在 `L1,L2∈[0.18,0.38] m` 的 21×21 网格求增益，按两腿长度各三次的张量积多项式拟合。归一化坐标是 `lambda_i=clamp((Li-0.28)/0.10,-1,1)`；数据按 `C[input][state][p][q]` 升幂储存。生成目录内有：

- `airborne_lqr_coef.h`：固件 C++ 纯系数头，独立命名空间 `airborne_lqr`。
- `airborne_lqr_schedule.npz`：节点增益、拟合系数、状态与输入顺序。
- `airborne_lqr_fit.json`、`README.md`：误差与可控模态极点检查。

以下示例与头文件的系数次序一致。可在现有十四状态估计量上取小模型误差，再沿用下位机执行器限幅与接触状态机：

```cpp
#include <algorithm>
#include "airborne_lqr_coef.h"

// x14 and ref14 use the existing interleaved fourteen-state order.
float e[airborne_lqr::STATE_DIM] = {
    x14[10] - ref14[10], x14[11] - ref14[11],
    x14[12] - ref14[12], x14[13] - ref14[13],
    x14[4]  - ref14[4],  x14[5]  - ref14[5],
};
float lam1 = std::clamp((L1 - airborne_lqr::LENGTH_CENTER_M) /
                        airborne_lqr::LENGTH_HALF_RANGE_M, -1.0f, 1.0f);
float lam2 = std::clamp((L2 - airborne_lqr::LENGTH_CENTER_M) /
                        airborne_lqr::LENGTH_HALF_RANGE_M, -1.0f, 1.0f);
float tp[airborne_lqr::INPUT_DIM] = {};
for (unsigned i = 0; i < airborne_lqr::INPUT_DIM; ++i) {
    for (unsigned j = 0; j < airborne_lqr::STATE_DIM; ++j) {
        float value = 0.0f;
        for (int p = airborne_lqr::K_POLY_DEGREE; p >= 0; --p) {
            float row = 0.0f;
            for (int q = airborne_lqr::K_POLY_DEGREE; q >= 0; --q)
                row = row * lam2 + airborne_lqr::K_COEF[i][j][p][q];
            value = value * lam1 + row;
        }
        tp[i] -= value * e[j];
    }
    tp[i] = std::clamp(tp[i], -60.0f, 60.0f);
}
```

验证程序在全部网格点和相邻网格中心用 `float32` 系数重算闭环四个可控极点，要求模长严格小于 1；报告另列节点拟合误差。完整系统的两个单位极点是物理不变量，不计入“四个可控极点稳定”的结论。模型假设飞行期间腿长变化相对控制周期缓慢、横滚和偏航耦合可忽略；目前没有包含接触重建、可变腿长的惯性耦合和实机电机动态。
