# python-wl

双轮腿机器人主机侧建模、LQR 增益调度与离线分析代码。该目录是从原项目整理出的独立可运行副本，保留 `wheel_leg_lqr` 的旧导入入口，同时把实现按职责拆分，便于测试、替换和扩展。

## 目录结构

```text
python-wl/
├─ src/wheel_leg_lqr/
│  ├─ core/                 # 物理参数、状态维度、符号模型与静态线性化
│  ├─ control/              # 离散化、LQR、可控性数值计算、饱和仿真
│  ├─ scheduling/           # 唯一的 L1/L2 张量积增益调度和多项式拟合
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
cd E:\Software\Agent\pack\python-wl
python -m pip install -e .
```

常用命令：

```powershell
wheelleg-schedule --step 0.02 --output-dir data/generated
wheelleg-controllability --output-dir data/generated
wheelleg-bandwidth --height 0.24
wheelleg-optimize-qr --max-iter 400
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
- `control` 提供不依赖调度策略的通用控制运算。
- `scheduling` 只组织 L1/L2 配平点扫描、张量积拟合和验证；文件格式由调度模块调用 `artifacts.io` 完成。
- 生成物必须写到显式 `--output-dir`，不要覆盖源码或参考数据。

## 测试

```powershell
python -m pytest -p no:cacheprovider -q
```

当前参数来自最新 CAD/动力学提交；若修改物理参数，应重新生成 `data/generated` 下的产物并运行完整测试。
