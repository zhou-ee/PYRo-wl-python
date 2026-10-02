# 前向模型加速度：上位机生成与调度契约

依据《前向模型加速度设计与实现规范》的静态线性化方案，复用现有 `core.model`，生成轮心中点 C 的前向模型加速度系数。拟合方式按用户要求限定为**二维总阶不超过三次**，每个通道 10 项；不采用三次全张量的 16 项。

## 生成与产物

```powershell
python run_model_acceleration.py --output-dir data/generated/model_acceleration
```

安装包后可用同参数的 `wheelleg-model-acceleration`。默认腿长域为 `[0.18,0.38]² m²`，训练间隔 0.01 m，共 441 点。独立验证包括 41×41 加密网格中不属于训练网格的 1240 点，以及 64 个固定种子随机点及其左右镜像，共 1368 点。

| 文件 | 内容 |
|---|---|
| `model_acceleration_schedule.h` | 10×10 的 float32 系数表、腿长范围与归一化常量；不生成函数或结构体 |
| `model_acceleration_schedule.npz` | float32/float64 表、参考系数、训练与验证点、q0、总 u0、主动 u0、气弹簧力、κg |
| `model_acceleration_schedule.json` | 系数项序、模型参数、输入约定、各通道绝对和归一化误差、镜像检查 |
| `model_acceleration_schedule_report.md` | 可读的误差、镜像与工作点报告 |

NPZ 的参考系数和工作点数组按 `training_points` 后接 `holdout_points` 排列。产物目录遵循项目现有规则，不纳入 Git；重新运行命令即可生成。

## 从动力学提取 10 个系数

广义坐标及输入沿用项目顺序：

\[
q=[x,\psi,\theta,\phi,L_{\rm bar},\beta_1,\beta_2]^T,
\qquad u=[T_{w1},T_{w2},T_{p1},T_{p2},f_{l1},f_{l2}]^T.
\]

**φ 在项目中是机体 roll，β1/β2 是虚拟腿摆角**。参考方案把 φ 写为“腿部摆角”，这里按现有动力学代码对齐。

静态配平满足 `S0 u0 = G0`。既有线性化采用 `D0=0`，刚度为

\[
K_0=\left.\frac{\partial(G(q)-S(q)u_0)}{\partial q}\right|_{q_0},
\qquad
M_0\delta\ddot q=-K_0\delta q+S_0\delta u.
\]

求解一次 `M0.T r = e_x`，得到 `r.T = e_x.T M0^{-1}`，无需显式求逆。于是

\[
k=-r^T K_{0,:,\mathcal I},\qquad b=r^T S_0,
\qquad \mathcal I=(2,3,5,6),
\]

这里列索引为 Python 的 0 基索引，选取 θ、φ、β1、β2。调度点每拍取实测 L1/L2，状态 L_bar 也取其均值，因此腿长偏差恒为零，按用户要求不提取或调度这一通道。完整地面模型仍保留七个广义坐标。

生成器自动检查前向行的 x/ψ 系数和全部速度系数为零，再提取

\[
c=[k_\theta,k_\phi,k_{\beta1},k_{\beta2},
b_{w1},b_{w2},b_{p1},b_{p2},b_{f1},b_{f2}].
\]

令 q_a=[θ,φ,β1,β2]，运行时只需

\[
\boxed{\ddot x_{\rm model}
=\sum_{j=0}^{3}c_j\delta q_{a,j}
+\sum_{j=0}^{5}c_{j+4}\bigl(u_j-u_{0j}+\hat d_{u,j}(k-1)\bigr).}
\]

RDOB 未启用时取扰动为零。每个系数的量纲为 `(m/s²)/(对应状态或输入单位)`；角度用 rad，长度用 m，力矩用 N·m，腿力用 N。该模型描述静态工作点附近的局部加速度，不包括完整非线性速度耦合及腾空动力学。

## 总阶三次多项式

沿用现有 LQR/RDOB 的中心和半区间归一化：

\[
\xi=\frac{L_1-L_c}{L_s},\quad \eta=\frac{L_2-L_c}{L_s},
\quad L_c=0.28\ {\rm m},\quad L_s=0.10\ {\rm m}.
\]

每个通道

\[
c(\xi,\eta)=\sum_{i+j\le3}a_{ij}\xi^i\eta^j,
\]

系数表列顺序固定为

```text
1, eta, eta², eta³, xi, xi*eta, xi*eta², xi², xi²*eta, xi³
```

不含 `xi² eta²`、`xi³ eta` 等总阶超过三次的项。10 个通道共 100 个 float，系数表 400 字节；归一化另有 4 个 float 常量。

清晰的 MCU 标量求值顺序如下，生成头文件提供所引用的常量。固件已将该表同步到 `wl_config.h`，在 `_calc_ddot_x_model()` 中求值并复用 `Hxi/Bx`。

```cpp
using namespace wheel_leg_model_acceleration;
float x = (L1 - MODEL_ACCEL_LENGTH_CENTER) * MODEL_ACCEL_LENGTH_INV_SCALE;
float y = (L2 - MODEL_ACCEL_LENGTH_CENTER) * MODEL_ACCEL_LENGTH_INV_SCALE;
// L1/L2 须在生成域内；边界的 float 舍入按 [-1, 1] 截断。
if (x < -1.0f) x = -1.0f;
if (x > 1.0f) x = 1.0f;
if (y < -1.0f) y = -1.0f;
if (y > 1.0f) y = 1.0f;
for (unsigned i = 0; i < 10; ++i)
{
    const float* a = MODEL_ACCEL_COEF[i];
    const float row0 = ((a[3] * y + a[2]) * y + a[1]) * y + a[0];
    const float row1 = (a[6] * y + a[5]) * y + a[4];
    const float row2 = a[8] * y + a[7];
    c[i] = ((a[9] * x + row2) * x + row1) * x + row0;
}
```

Python float32 求值使用同样的常量舍入和 Horner 运算顺序；C++ 编译测试以关闭 FMA 收缩的方式核对导出表。开启 FMA 时允许正常的末位舍入差异。

## 工作点与总输入

生成器直接调用 `linearize_static_trim(L1,L2)`，不另定义或拟合配平点：

- `theta0=phi0=0`；`L_bar0=(L1+L2)/2`。
- `beta_i0=compute_equilibrium_beta_from_length(Li)`，通常非零。
- `u0` 是满足静力学平衡的**总输入**，不是主动电机腿力。

例如 `L1=L2=0.28 m` 时，腿角配平约为 `−0.02694833 rad`，每腿总配平支撑力约为 `100.593394 N`。运行时的 q0 和总 u0 应由同一节点的 LQR/RDOB 配平路径提供；生成文件中的逐点配平数据用来核对，不作为一套独立的 MCU 配平调度表。

实际主动腿力必须先加当前气弹簧力：`u_total[4+i] = f_motor[i] + f_gas(Li)`，然后只减一次总 `u0`。如果调用者仅有 LQR 的主动前馈 `FL_U0`，应先恢复总配平输入再相减。

调度变量每拍取实际 L1/L2、状态 `L_bar` 也取实际均值时，`delta_L_bar=0`。上下位机均删除腿长偏差项，状态系数只保留 θ、φ、β1、β2。当前模块采用这一连续调度约定，不用于冻结腿长工作点后产生非零平均腿长偏差的计算。全部保留增量为零时模型输出严格为零。

上位机参考计算：

```python
from wheel_leg_lqr.core.model import linearize_static_trim
from wheel_leg_lqr.model_acceleration import (
    compute_model_acceleration_schedule, predict_model_acceleration,
)

schedule = compute_model_acceleration_schedule()
trim = linearize_static_trim(L1, L2)
ddot_x_model = predict_model_acceleration(
    schedule.evaluate(L1, L2), q, actual_total_input,
    trim.geometry.q, trim.equilibrium_input,
    previous_disturbance=d_previous,  # 六维；未启用时省略
)
```

`previous_disturbance` 的锁存与时序由调用者负责，函数不更新 RDOB，不创建同拍反馈循环。

当前下位机按用户要求不使用扰动估计：仅计算 `k*delta_q + b*(output-U0)`，在平衡状态 `_gain_calculate()` 之后调用。主动腿力与主动配平腿力中的同一工作点气弹簧力抵消；`delta_L_bar=0`，腿长控制目标不作为模型工作点。上面的 Python 函数仍保留可选扰动接口，当前固件没有使用该接口对应的扰动项。下位机完整计算与调用说明见固件仓库的 `docs/model_acceleration_implementation.md`。

## 验证与误差定义

先在原始模型检查镜像关系，然后将系数表投影到已确认的左右对称子空间：交换 xi/eta 的幂次与左右通道，k_phi 反号，其余自映射通道保持符号。投影保留总阶三次结构，在对称训练网格上保证解析镜像关系。float32 全域求值后再次检查全部镜像关系及对称线关系，失败则终止生成。

误差定义为 `error = fitted - reference`，各通道的参考尺度是全部训练和验证样本的 `max(abs(reference))`，下限 `1e-12`。分别报告绝对 RMS/最大误差与按该固定峰值归一后的百分比，避免 k_phi 在对称线上为零时逐点相对误差无意义。不同量纲通道不合并计算一个矩阵范数。

默认 `--max-error 0.02` 是导出硬门限；`--rms-target 0.005` 是文档建议的报告目标，未达目标会在控制台、JSON 和报告中明确列出。训练误差、独立验证误差和 float32 额外舍入误差分开报告。三次拟合不会因超出 RMS 建议值而自动升阶或切换全张量。系数误差比例不能直接等同于最终加速度误差比例。

```powershell
python -m pytest -p no:cacheprovider -q tests/test_model_acceleration.py
```

测试覆盖既有 A/B 的行索引、10 项提取、独立非线性静态小信号响应、工作点增量、平均腿长通道消除、气弹簧总输入约定、RDOB 上一拍扰动、原始模型与 float32 镜像、独立验证点和 C++ 导出表求值。JSON 使用 schema_version=2，记录活动状态列 `[2,3,5,6]`；旧 11 通道表需要重新生成。物理参数或配平模型变化后须重新生成。这些是离线模型检查，未进行实机动态融合验证。
