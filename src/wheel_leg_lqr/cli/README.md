# 命令行流程入口

每个文件负责参数解析、调用计算、指定输出和展示结果。保留 `wheelleg-*` 命令；未安装时可用 `PYTHONPATH=src` 与 `python -m wheel_leg_lqr.cli.<模块>`。

入口：`ground_lqr`、`airborne_lqr`、`rdob`、`symbolic`、`bandwidth`、`qr_optimizer`、`step_drop`、`airborne_fl_pitch`、`leg_com`。`--output-dir` 覆盖默认输出路径，命令行不会等待 Enter。

## 依赖边界

CLI 组装计算、分析与导出；下层不导入 CLI。

## 使用示例

```powershell
# PowerShell
python -m wheel_leg_lqr.cli.airborne_lqr --step 0.02
python -m wheel_leg_lqr.cli.rdob --mode node --l1 0.24 --l2 0.32
```

## 文件与接口

| 文件 | 主要接口 |
|---|---|
| [airborne_fl_pitch.py](airborne_fl_pitch.py) | `main` |
| [airborne_lqr.py](airborne_lqr.py) | `main` |
| [bandwidth.py](bandwidth.py) | `main` |
| [ground_lqr.py](ground_lqr.py) | `main` |
| [leg_com.py](leg_com.py) | `main` |
| [qr_optimizer.py](qr_optimizer.py) | `main` |
| [rdob.py](rdob.py) | `main` |
| [step_drop.py](step_drop.py) | `main` |
| [symbolic.py](symbolic.py) | `main` |
