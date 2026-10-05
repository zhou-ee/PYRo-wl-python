# Windows 启动脚本

每个 `run_*.py` 从本脚本路径找到项目 `src`，转发到对应 CLI。适合未安装包时运行，保留旧启动器的功能名称。输出目录由 CLI 统一决定，不在脚本中强行覆盖用户参数。

## 依赖边界

只处理未安装导入路径，不承担算法实现。

## 使用示例

```powershell
# PowerShell，从任何目录运行都可以
python E:\Software\Develop\RM\Wheel_leg\PYRo-wl-python\scripts\run_rdob.py --mode node
```

## 文件与接口

| 文件 | 主要接口 |
|---|---|
| [run_airborne_fl_pitch.py](run_airborne_fl_pitch.py) |  |
| [run_airborne_lqr.py](run_airborne_lqr.py) |  |
| [run_analyze_bandwidth.py](run_analyze_bandwidth.py) |  |
| [run_controllability.py](run_controllability.py) |  |
| [run_gain_schedule.py](run_gain_schedule.py) |  |
| [run_optimize_qr.py](run_optimize_qr.py) |  |
| [run_rdob.py](run_rdob.py) |  |
| [run_step_drop.py](run_step_drop.py) |  |
