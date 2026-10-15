# PYRo-wl-python

双轮腿机器人离线建模、地面/腾空 LQR 调度、六通道输入等效 RDOB 与分析工具。

工作区：`E:\Software\Develop\RM\Wheel_leg\PYRo-wl-python`。原工作区保留在 `E:\Software\Agent\pack\PYRo-wl-python`。

## 安装与运行

```powershell
cd E:\Software\Develop\RM\Wheel_leg\PYRo-wl-python
python -m pip install -e ".[test]"
# 需要 CAD Excel 拟合时
python -m pip install -e ".[cad]"

wheelleg-schedule --step 0.02
wheelleg-airborne-lqr --step 0.02
wheelleg-rdob --step 0.02
wheelleg-rdob --mode node --l1 0.24 --l2 0.32
wheelleg-controllability
wheelleg-bandwidth --height 0.24
wheelleg-optimize-qr --max-iter 400
wheelleg-step-drop --step-height 0.18 --horizon 1.0
wheelleg-airborne-fl-pitch --body-com-x 0.03 --fl1 100 --fl2 100
wheelleg-fit-leg-com
```

未安装时可以运行 `python scripts/run_gain_schedule.py --step 0.02` 等启动器；也可设置 `$env:PYTHONPATH="src"` 后运行 `python -m wheel_leg_lqr.cli.ground_lqr`。

## 结构与组件说明

每个组件目录都有 README，介绍职责、配置、实际接口、示例和依赖边界。

| 组件 | 职责 |
|---|---|
| [统一配置与包入口](src/wheel_leg_lqr/README.md) | 整机质量、惯量、几何、质心曲线、被动机构与执行器参数 |
| [物理模型](src/wheel_leg_lqr/models/README.md) | 共享物理数据，独立建立各接触模式的方程 |
| [地面模型](src/wheel_leg_lqr/models/ground/README.md) | 七坐标、十四状态、六输入，配平与地面 Q/R |
| [腾空模型](src/wheel_leg_lqr/models/airborne/README.md) | 六状态、双髋输入、四维可控形状空间 |
| [接触模型](src/wheel_leg_lqr/models/contact/README.md) | 九坐标释放、地面碰撞和机械限位 |
| [通用控制](src/wheel_leg_lqr/control/README.md) | 显式 Q/R、采样时间与任意维数的 LQR/RDOB |
| [腿长调度](src/wheel_leg_lqr/scheduling/README.md) | 网格、拟合、float32 与闭环验证 |
| [仿真](src/wheel_leg_lqr/simulation/README.md) | 饱和恢复和单腿下台阶 |
| [分析](src/wheel_leg_lqr/analysis/README.md) | 能控性、带宽、QR 优化、腾空腿力影响 |
| [CAD 标定](src/wheel_leg_lqr/calibration/README.md) | 原始 Excel 质心曲线拟合 |
| [导出](src/wheel_leg_lqr/artifacts/README.md) | C++、NPZ、JSON、参数指纹与输出路径 |
| [命令行](src/wheel_leg_lqr/cli/README.md) | 参数解析、流程组装与结果展示 |
| [启动脚本](scripts/README.md) | Windows 未安装运行 |
| [参考数据](data/reference/README.md) | CAD 输入原样保留 |
| [测试](tests/README.md) | 物理、控制、调度、导出与依赖回归 |
| [专题文档](docs/README.md) | 推导、固件契约和迁移结果 |

## 配置约定

整机参数只在 `src/wheel_leg_lqr/config.py` 维护。各模型读取同一个不可变 `RobotParameters`；独立实验用 `dataclasses.replace()` 创建参数快照，通过 `parameters=` 传入。地面编译缓存以物理参数对象为键。

地面 Q/R 在 `models/ground/config.py`，腾空 Q/R 在 `models/airborne/config.py`，接触参数在 `models/contact/config.py`。物理线性化不读取 Q/R；腾空加权形状投影单独位于 `reduction.py`。

现有参数的维护入口如下。修改后重新启动计算命令并重新生成产物，无需同步修改测试期望值。

| 参数 | 配置文件 |
|---|---|
| 质量、惯量、质心、几何、腿长范围、执行器限幅、气弹簧 | `src/wheel_leg_lqr/config.py` |
| 地面 Q/R、采样周期、配平/分析高度 | `src/wheel_leg_lqr/models/ground/config.py` |
| 腾空 Q/R、采样周期 | `src/wheel_leg_lqr/models/airborne/config.py` |
| 接触刚度、阻尼、积分子步数 | `src/wheel_leg_lqr/models/contact/config.py` |
| RDOB 默认极点/通道带宽、扰动限幅比例 | `src/wheel_leg_lqr/scheduling/config.py` |

数值回归测试显式使用 `tests/reference_parameters.py` 中的固定机器人与控制器样本，该文件不是实际模型配置。当前配置的传递、缓存和默认行为另有检查；计算命令仍使用当前配置进行稳定性、拟合与饱和恢复验证。

腾空腿力分析默认读取统一质心；`--body-com-x` 用于显式覆盖实验位置。Python 进程启动时读取配置，文件修改后需重新运行命令。

腿部前后质心调度开关位于统一 `config.py` 的 `RobotParameters.leg_com_fore_aft_enabled`，默认 `True`，保留 CAD 前后偏移曲线。设为 `False` 时，模型令 `x_com(L)=0`，腿质心在腿部矢状平面内始终位于髋—轮连线上；沿腿方向的 `L_wp(L)` 仍随腿长变化。地面配平、腾空、接触和导出的平衡腿角均采用此开关；当前配平公式下关闭后的平衡腿角为零。原始 CAD 曲线保留，重新开启即可恢复。切换后重新运行所需计算命令，生成对应的 K、前馈、RDOB 和固件头文件。

气弹簧沿用下位机抵消约定：A/B/K 使用总虚拟腿力，主动腿力前馈减去当前气弹簧力。RDOB 使用实际施加的总输入减配平输入。

## 输出与验证

默认写入 `outputs/ground_lqr`、`outputs/airborne_lqr`、`outputs/rdob`。显式 `--output-dir` 优先；相对路径以当前工作目录解析。参考 CAD 目录禁止作为生成物输出路径。

地面调度默认用 25 个代表节点做耗时饱和验证，逐节点验证可用 `--exhaustive-validation`。腾空只验证可控形状子空间，完整模型的两个守恒模态保持中性。RDOB 冻结点检查不等价于任意变腿长过程稳定性保证。

```powershell
python -m pytest -p no:cacheprovider -q
```

前向模型加速度估计已从新工作区移除。保留动力学加速度、RDOB 和腾空 FL→pitch 分析。

原代码库与新工作区的同参数数值验证见 [迁移对照报告](docs/migration_comparison.md)。
