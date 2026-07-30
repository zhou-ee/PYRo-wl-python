# 固定轮腿动力学与 LQR / Fixed Wheel-Legged Dynamics and LQR

本项目只按照用户新文档实现固定几何下的动力学、 A/B 矩阵、离散化和 LQR。

This project implements only the fixed-geometry dynamics, A/B matrices, discretization, and LQR defined in the user document.

## 1. 参数只在这里填写 / Edit parameters here

所有你需要填写的数值都在下面这个文件：

```text
wheel_legged_lqr/config.py
```

All robot-specific values must be filled in this file:

```text
wheel_legged_lqr/config.py
```

### 1.1 文档中的物理变量 / Physical variables from the document

| Python name | Document symbol | Unit /  | What to enter /  |
|---|---|---|---|
| `m_w` | `m_w` | kg | 轮部质量 / wheel mass |
| `I_w` | `I_w` | kg*m^2 | 轮部绕转动惯量 / wheel inertia |
| `R_w` | `R_w` | m | 轮半径 / wheel radius |
| `m_p` | `m_p` | kg | 腿部质量 / leg mass |
| `I_p` | `I_p` | kg*m^2 | 腿部绕转动惯量 / leg inertia |
| `m_b` | `m_b` | kg | 机体质量 / body mass |
| `I_b` | `I_b` | kg*m^2 | 机体绕转动惯量 / body inertia |
| `L_c` | `L_c` | m | 髋关节到机体质心距离 / hip-to-body-COM distance |
| `g` | `g` | m/s^2 | 重力加速度 / gravity |

特别注意：`I_w`、`I_p`、`I_b`必须填入你自己的实际转动惯量，程序不会根据外形尺寸自动推导。

`I_w`, `I_p`, and `I_b` must be your measured or independently calculated inertias. The program never derives them from unrelated body dimensions.

### 1.2 固定几何 / Fixed geometry

```python
L = 0.24
L_wp = 0.12
L_ph = 0.12
```

这三个值需满足 `L = L_wp + L_ph`。

These values must satisfy `L = L_wp + L_ph`.

### 1.3 工作点、采样和 LQR 权重 / Working point, sampling, and LQR weights

```python
x_0 = None
dx_0 = None
beta_0 = None
dbeta_0 = None
gamma_0 = None
dgamma_0 = None
T_w_0 = None
T_p_0 = None
Ts = None
Q_diag = [None, None, None, None, None, None]
R_diag = [None, None]
```

状态顺序为 `[x, dx, beta, dbeta, gamma, dgamma]`。

The state order is `[x, dx, beta, dbeta, gamma, dgamma]`.

The nominal state variables are `x_0`, `dx_0`, `beta_0`, `dbeta_0`, `gamma_0`, and `dgamma_0`. The nominal inputs are `T_w_0` and `T_p_0`. /  `x_0``dx_0``beta_0``dbeta_0``gamma_0``dgamma_0` `T_w_0`  `T_p_0`

## 2. 代码分工 / Code responsibilities

| File | Function /  |
|---|---|
| `config.py` | 填写所有实际参数 / enter all actual values |
| `parameters.py` | 参数数据类与合法性检查 / data class and validation |
| `dynamics.py` | 文档坐标系下的运动学、内力和动力学 / document dynamics |
| `linearization.py` | 计算 A=df/dX 和 B=df/dU / compute A and B |
| `control_tools.py` | 调用现成离散化和 LQR 库 / call control libraries |
| `lqr.py` | 组合完整计算流程 / run complete pipeline |
| `cli.py` | 运行程序并保存结果 / run and save results |

## 3. 使用的现成库 / Libraries used

- `scipy.signal.cont2discrete(..., method="zoh")`: 计算 Ad 和 Bd;
- `control.dlqr(Ad, Bd, Q, R)`: 计算 K、P 和闭环极点;
- `control.ctrb(A, B)`: 计算可控性矩阵;

不再手写 Riccati 方程求解。

The Riccati equation is solved by `python-control`, not by handwritten numerical code.

## 4. 安装环境 / Environment

Use the PyTorch environment:

```text
E:\AI_Develop\conda\envs\pytorch
```

Install dependencies:

```powershell
& E:\AI_Develop\conda\envs\pytorch\python.exe -m pip install -r requirements.txt
```

## 5. 运行 / Run

填写 `config.py` 中的所有 `None` 后，运行：

After filling every `None` in `config.py`, run:

```powershell
cd E:\Develop\a_PYRo\PYRo-wl\PYRo-wl_2027\python_lqr
& E:\AI_Develop\conda\envs\pytorch\python.exe wheel_legged_lqr\cli.py
```

Equivalent module form:

```powershell
& E:\AI_Develop\conda\envs\pytorch\python.exe -m wheel_legged_lqr.cli
```

如果参数没有填写，程序会明确提示哪个变量缺失。

If a value is missing, the program reports the exact variable that must be filled.

## 6. 输出 / Output

After successful configuration, the program writes:

```text
artifacts/lqr_fixed.npz
artifacts/lqr_fixed.json
```

`lqr_fixed.npz` contains `A`, `B`, `Ad`, `Bd`, `Q`, `R`, `P`, `K`, closed-loop poles, and controllability rank.

## 7. 测试 / Tests

Automated tests use a separate synthetic test fixture and never fill or modify your production configuration.

```powershell
& E:\AI_Develop\conda\envs\pytorch\python.exe -m pytest -q
```
