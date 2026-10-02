# 重构映射

| 模块 | 职责 |
|---|---|
| `src/wheel_leg_lqr/core/config.py` | 物理参数、状态/输入维度、权重与配平计算 |
| `src/wheel_leg_lqr/core/model.py` | 直接基于 `L1/L2` 的几何、质量矩阵与静态线性化 |
| `src/wheel_leg_lqr/airborne/config.py` | 腾空模型独立的物理参数、腿质心拟合和 Q/R 权重 |
| `src/wheel_leg_lqr/airborne/model.py` | 无接触双髋模型、自由平移消元及四维可控形状映射 |
| `src/wheel_leg_lqr/airborne/fl_pitch.py` | 独立五坐标无接触模型的 `F_L→pitch` 瞬时影响分析入口 |
| `src/wheel_leg_lqr/control/lqr.py` | 离散化、LQR 和数值可控性公共算法 |
| `src/wheel_leg_lqr/control/rdob.py` | 六阶输入等效 RDOB 连续设计、梯形离散更新和初始化 |
| `src/wheel_leg_lqr/rdob.py` | RDOB 上位机入口，默认 L1/L2 二维调度，可选单点诊断 |
| `src/wheel_leg_lqr/scheduling/rdob_schedule.py` | RDOB 的 Gamma 三次全张量与 Bq 总阶三次直接拟合、float32 验证、六通道离散常量及纯数据 C++ 头文件导出 |
| `src/wheel_leg_lqr/model_acceleration.py` | 前向模型加速度公开入口与 CLI |
| `src/wheel_leg_lqr/scheduling/model_acceleration_schedule.py` | 前向行 10 个系数提取、总阶三次拟合、镜像及 float32 验证、系数与工作点导出；实测腿长连续调度省去平均腿长偏差项 |
| `docs/model_acceleration.md` | 前向模型加速度标量计算、总阶三次项序、配平与总输入约定 |
| `docs/input_equivalent_rdob.md` | RDOB 数学推导、投影条件及运行时契约 |
| `docs/rdob_design_and_implementation.md` | RDOB 专题：调度对象、完整推导、离散更新与 C++ 示例 |
| `src/wheel_leg_lqr/control/saturation.py` | 执行器饱和仿真和验证 |
| `src/wheel_leg_lqr/scheduling/leg_length_schedule.py` | 地面十四状态 `L1/L2` 张量积增益拟合与产物导出 |
| `src/wheel_leg_lqr/airborne/schedule.py` | 独立离散 LQR、双腿长增益拟合、float32 稳定性验证与头文件导出 |
| `src/wheel_leg_lqr/airborne_lqr.py` | 腾空小模型公开入口与 CLI |
| `docs/06_airborne_lqr.md` | 腾空小模型推导、状态映射、角动量约束与固件使用说明 |
| `src/wheel_leg_lqr/scheduling/polynomial.py` | L1/L2 张量积与总阶三次多项式基础运算 |
| `src/wheel_leg_lqr/analysis/symbolic.py` | 符号能控性分析 |
| `src/wheel_leg_lqr/analysis/bandwidth.py` | 闭环模态与带宽分析 |
| `src/wheel_leg_lqr/analysis/step_drop.py` | 单腿下台阶的释放、地面/机械限位接触恢复与 RDOB 限幅筛查 |
| `src/wheel_leg_lqr/analysis/qr_optimizer.py` | QR 权重离线寻优 |
| `docs/step_drop_simulation.md` | 0.18 m 单腿下台阶的可复现实验条件、结果与模型适用边界 |
| `src/wheel_leg_lqr/artifacts/io.py` | 生成物写盘工具 |

地面 LQR 调度入口为 `src/wheel_leg_lqr/gain_schedule.py`，腾空 LQR 为 `src/wheel_leg_lqr/airborne_lqr.py`，RDOB 为 `src/wheel_leg_lqr/rdob.py`。腾空包自带配置、动力学、离散 LQR 和拟合；十四状态索引只在公开入口和文档中作为接线映射。
