# 迁移结果对照

`compare_migration.py` 在独立 Python 进程中加载一个代码库，生成配平、地面/腾空模型与完整调度、RDOB、接触力学、时域轨迹、CAD 标定和分析结果。旧库质量只通过进程内参数对象覆盖，源文件不修改；新库通过 `parameters=` 传入相同质量。

分别对 12 kg 和 18.9255 kg 运行旧库、新库四个进程。`summarize_comparison.py` 比较 NPZ 数组，并区分数值一致、舍入误差、已说明的行为调整和无法解释的差异。无法解释的差异或键/文本状态不一致会返回失败。

输出位于 `outputs/migration_comparison/`，正式结论见 `docs/migration_comparison.md`。比较固定使用地面/RDOB 0.02 m 网格、腾空 0.01 m 网格、当前两个库一致的 Q/R 和 1 ms 采样周期。

```powershell
python docs/validation/compare_migration.py old E:\Software\Agent\pack\PYRo-wl-python 12 outputs/migration_comparison/old_12
python docs/validation/compare_migration.py new E:\Software\Develop\RM\Wheel_leg\PYRo-wl-python 12 outputs/migration_comparison/new_12
# 将以上两行质量和输出目录替换为 18.9255 后再执行一次
python docs/validation/summarize_comparison.py outputs/migration_comparison
```

旧 CAD 拟合器在原代码中固定使用 12 kg，因此 18.9255 kg 情况下其平衡角输出单独列为已知修正，不作为等参回归断言。QR 优化器同样单独记录旧代码硬编码横滚权重 800 与当前配置权重 600 的区别；共同 Q/R 输入下的候选评估另行比较。
