# 模块索引

| 目录 | 说明 |
|---|---|
| `src/wheel_leg_lqr` | [建模包与统一物理配置](src/wheel_leg_lqr/README.md) |
| `src/wheel_leg_lqr/models` | [物理模型](src/wheel_leg_lqr/models/README.md) |
| `src/wheel_leg_lqr/models/ground` | [地面双支撑模型](src/wheel_leg_lqr/models/ground/README.md) |
| `src/wheel_leg_lqr/models/airborne` | [腾空双髋模型](src/wheel_leg_lqr/models/airborne/README.md) |
| `src/wheel_leg_lqr/models/contact` | [释放与碰撞接触模型](src/wheel_leg_lqr/models/contact/README.md) |
| `src/wheel_leg_lqr/control` | [通用控制算法](src/wheel_leg_lqr/control/README.md) |
| `src/wheel_leg_lqr/scheduling` | [腿长调度与拟合](src/wheel_leg_lqr/scheduling/README.md) |
| `src/wheel_leg_lqr/simulation` | [闭环与接触仿真](src/wheel_leg_lqr/simulation/README.md) |
| `src/wheel_leg_lqr/analysis` | [离线分析](src/wheel_leg_lqr/analysis/README.md) |
| `src/wheel_leg_lqr/calibration` | [CAD 腿部质心拟合](src/wheel_leg_lqr/calibration/README.md) |
| `src/wheel_leg_lqr/artifacts` | [生成物导出](src/wheel_leg_lqr/artifacts/README.md) |
| `src/wheel_leg_lqr/cli` | [命令行流程入口](src/wheel_leg_lqr/cli/README.md) |
| `scripts` | [Windows 启动脚本](scripts/README.md) |
| `data` | [输入数据](data/README.md) |
| `data/reference` | [CAD 原始参考数据](data/reference/README.md) |
| `outputs` | [离线生成物](outputs/README.md) |
| `tests` | [功能与数值回归验证](tests/README.md) |
| `docs` | [设计与使用文档](docs/README.md) |

## 旧模块到新模块

| 旧模块或入口 | 新位置 |
|---|---|
| `core/config.py`、`numeric_config.py` | 公共物理参数：`config.py`；地面状态/Q/R：`models/ground/config.py`；配平角：`models/ground/trim.py` |
| `core/model.py`、`numeric_model.py` | `models/ground/model.py` |
| `airborne/config.py` | 公共物理参数：`config.py`；腾空状态/Q/R：`models/airborne/config.py` |
| `airborne/model.py` | 物理线性化：`models/airborne/model.py`；原带权重接口：`models/airborne/reduction.py` 的 `build_shape_model()` |
| `airborne/schedule.py`、`airborne_lqr.py` | `scheduling/airborne_lqr.py`、`artifacts/airborne_lqr.py`、`cli/airborne_lqr.py` |
| `airborne/fl_pitch.py` | `analysis/airborne_fl_pitch.py` 与 `cli/airborne_fl_pitch.py` |
| `gain_schedule.py`、`scheduling/leg_length_schedule.py` | `scheduling/ground_lqr.py`、`artifacts/ground_lqr.py`、`cli/ground_lqr.py` |
| 顶层 `rdob.py` | `scheduling/rdob_node.py`、`artifacts/rdob_node.py`、`cli/rdob.py` |
| `scheduling/rdob_schedule.py` | `scheduling/rdob.py` 与 `artifacts/rdob.py` |
| `control/saturation.py`、`saturation.py` | `simulation/saturation.py`；初始用例：`models/ground/config.py` |
| `analysis/step_drop.py` | `models/contact/model.py`、`simulation/step_drop.py`、`cli/step_drop.py` |
| `data_process/com_fit.py` | `calibration/leg_com.py` 与 `cli/leg_com.py` |
| `data_process/清洗后重心.xlsx` | `data/reference/清洗后重心.xlsx` |
| 根目录 `run_*.py` | `scripts/run_*.py` |

上述包内路径均相对于 `src/wheel_leg_lqr/`。不再提供 `import *` 兼容门面；同一功能只有一个实现。
