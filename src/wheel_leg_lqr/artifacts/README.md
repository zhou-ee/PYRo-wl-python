# 生成物导出

`io.py` 统一 UTF-8 写盘、输出目录和参考数据写入保护。默认输出到项目 `outputs/<组件>`；安装后无法识别源码根时使用当前目录。显式相对路径相对于调用者当前目录。

`cpp.py` 格式化有限 float32 C++ 数组；`metadata.py` 保存物理参数快照和 SHA-256 指纹。`ground_lqr.py`、`airborne_lqr.py`、`rdob.py`、`rdob_node.py` 维护各自头文件/NPZ/JSON 契约。

导出使用结果携带的物理参数、Q/R、采样时间和域，保留既有 C++ 文件名、命名空间、状态顺序与系数项序。地面平衡角曲线由当前参数计算。导出器不重新设计控制器或运行调度。

`ground10_lqr.py` 导出十状态专用 JSON、NPZ 和 `ground10_lqr_coef.h`；命名空间为 `wheel_leg_ground10`，K 为 4×10，包含 float32 Horner 增益和配平求值器。默认目录为 `outputs/ground10_lqr`，具有独立模型标识与几何假设，不占用原十四状态文件契约。

## 依赖边界

读取已有结果、元数据与 NumPy；不被调度反向导入。

## 使用示例

```python
from wheel_leg_lqr.scheduling.rdob_node import compute_rdob_node
from wheel_leg_lqr.artifacts.rdob_node import export_rdob
from wheel_leg_lqr.artifacts.io import default_output_dir
export_rdob(compute_rdob_node(0.24, 0.32), default_output_dir('rdob_node'))
```

## 文件与接口

| 文件 | 主要接口 |
|---|---|
| [airborne_lqr.py](airborne_lqr.py) | `render_airborne_header`, `render_airborne_summary`, `save_airborne_schedule` |
| [cpp.py](cpp.py) | `cpp_array` |
| [ground_lqr.py](ground_lqr.py) | `render_active_fl_u0_block`, `render_firmware_lqr_body`, `render_header`, `render_summary`, `render_polynomial_matrices`, `save_result` |
| [ground10_lqr.py](ground10_lqr.py) | `render_header`, `save_result`, `save_node` |
| [io.py](io.py) | `project_root`, `default_output_dir`, `resolve_output_dir`, `write_text` |
| [metadata.py](metadata.py) | `artifact_metadata` |
| [rdob.py](rdob.py) | `render_rdob_schedule_header`, `export_rdob_schedule` |
| [rdob_node.py](rdob_node.py) | `export_rdob` |
