# python-wl

双轮腿机器人主机侧建模、LQR 增益调度、RDOB 系数生成与离线分析代码。实现按职责拆分，便于测试、替换和扩展。

## 目录结构

```text
python-wl/
├─ src/wheel_leg_lqr/
│  ├─ core/                 # 物理参数、状态维度、符号模型与静态线性化
│  ├─ control/              # 离散化、LQR、RDOB 与饱和仿真
│  ├─ scheduling/           # LQR/RDOB 的 L1/L2 调度和多项式拟合
│  ├─ analysis/             # 符号能控性、带宽、QR 权重优化
│  ├─ artifacts/            # 产物写盘的公共工具
│  └─ *.py                  # 兼容门面，旧代码无需立即改导入路径
├─ data_process/            # CAD 质心数据拟合
├─ data/generated/          # L1/L2 调度与符号分析生成物
├─ tests/                   # 单元与数值回归测试
└─ run_*.py                 # Windows 双击/命令行启动脚本
```

依赖安装：

```powershell
cd E:\Software\Agent\pack\PYRo-wl-python
python -m pip install -e .
```

常用命令：

```powershell
wheelleg-schedule --step 0.02 --output-dir data/generated
wheelleg-controllability --output-dir data/generated
wheelleg-bandwidth --height 0.24
wheelleg-optimize-qr --max-iter 400
wheelleg-step-drop --step-height 0.18 --down-leg 0 --horizon 1.0
python -m data_process.fit_leg_com
```

未安装时可使用：

```powershell
$env:PYTHONPATH = "src"
python -m wheel_leg_lqr.gain_schedule --help
```

默认在 L1/L2 网格上均匀选取 5×5、共 25 个代表节点执行耗时较高的 5 秒饱和恢复仿真；所有节点仍会执行 LQR、拟合误差和闭环极点稳定性检查。发布前需要逐节点饱和验证时使用：

```powershell
wheelleg-schedule --exhaustive-validation --output-dir data/generated
```

## 设计约定

- `core.config` 是物理常量、状态/输入顺序和权重矩阵的唯一来源。
- `core.model` 只负责几何、质量矩阵、重力、输入映射和静态线性化。
- 气弹簧由下位机按当前腿长实时抵消，因此不进入 `A/B/K`；调度器只把导出的主动腿力前馈改为 `FL_U0=F_L,total-F_gas(L)`。
- `control` 提供不依赖调度策略的通用控制运算。
- `scheduling` 只组织 L1/L2 配平点扫描、张量积拟合和验证；文件格式由调度模块调用 `artifacts.io` 完成。
- 生成物必须写到显式 `--output-dir`，不要覆盖源码或参考数据。

## 六阶输入等效 RDOB

六维扰动观测器默认按实际左右腿长 `L1,L2` 独立二维调度，范围为 `[0.18,0.38]² m²`。`L_bar=(L1+L2)/2` 随腿长变化，不固定为 0.28 m。观测全部六路输入等效扰动，补偿通道在观测器外独立选择。数学推导、采样时序与气弹簧输入约定见 [RDOB 设计说明](docs/input_equivalent_rdob.md)。

五个调度对象的作用、完整推导、离散更新与 C++ 示例见 [RDOB 设计与实现专题文档](docs/rdob_design_and_implementation.md)。

```powershell
python run_rdob.py --step 0.02 --fit-tolerance 0.005 --channel-tolerance 0.01 --sample-time 0.001 --output-dir data/generated/rdob
```

导出 `rdob_schedule.npz/json`、C++ 纯数据头文件 `rdob_schedule.h` 和验证报告。头文件不生成 `Coefficients` 或 `evaluate()`，只提供腿长范围、`Gamma/Bq` 拟合表及六通道 `rate/z pole/endpoint weight/input gain/velocity scale` 数组；即使当前六路相同也保持独立布局。默认 `D0=0`，六路名义离散极点均约为 `0.992`，按 RDOB 梯形离散反算约为 `1.278353 Hz`。部署参考步进默认将扰动估计限制为对应执行器上限的 `0.1×`，并在限幅后重基准化内部状态。运行时按上一区间系数更新并在端点重基准化辅助状态。冻结点稳定性检查不保证任意变腿长过程稳定。安装后可用同参数的 `wheelleg-rdob` 命令。

`Gamma` 使用三次全张量直接拟合，`Bq` 使用二维总阶三次直接拟合；`q0` 和总 `u0` 由既有 LQR 调度提供，不在 RDOB 中重复存储。默认矩阵拟合误差不超过 **0.5%**、匹配通道映射误差不超过 **1%**（相对 Frobenius 范数）。全零阻尼矩阵从 C++ 存储与求值中直接消去；常量离散极点和积分权重离线预计算。

保留单点诊断：`python run_rdob.py --mode node --l1 0.24 --l2 0.32 --output-dir data/generated/rdob_node`。单点文件名与调度文件名不同，C++ 调度应包含 `rdob_schedule.h`。

## 测试

```powershell
python -m pytest -p no:cacheprovider -q
```

当前参数来自最新 CAD/动力学提交；若修改物理参数，应重新生成 `data/generated` 下的产物并运行完整测试。

## 单腿下台阶接触筛查

wheelleg-step-drop 扩展了静态模型中的地面高度自由度，模拟下降腿释放、低台阶和机械限位的弹簧阻尼接触、双支撑恢复以及 LQR+Tw+FL RDOB 补偿。它用于识别当前控制器无法安全覆盖的过渡；接触刚度和阻尼仍需由实物辨识。0.18 m 台阶的当前结果和建模边界见 docs/step_drop_simulation.md。
