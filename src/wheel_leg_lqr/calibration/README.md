# CAD 腿部质心拟合

`leg_com.py` 的 `fit_leg_com_data(excel_path, parameters=...)` 读取 CAD Excel，拟合升幂三次质心与平衡角曲线并返回误差指标。质量读取公共配置，不再硬编码另一份机体/腿质量。

原始 Excel 在 `data/reference/清洗后重心.xlsx`。安装 `pip install -e ".[cad]"` 启用 pandas/openpyxl；基础建模导入不读取 Excel。拟合结果需要人工核对数据来源后更新公共配置，不自动覆盖正在使用的参数。

## 依赖边界

读取公共参数和参考数据；Excel 依赖延迟到拟合调用时加载。

## 使用示例

```python
from wheel_leg_lqr.calibration.leg_com import fit_leg_com_data
result = fit_leg_com_data()
print(result['metrics'])
```

## 文件与接口

| 文件 | 主要接口 |
|---|---|
| [leg_com.py](leg_com.py) | `fit_leg_com_data` |
