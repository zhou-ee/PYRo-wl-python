# 输入数据

只维护可追溯的输入数据。CAD 输入在 `reference/`，生成物统一进入项目 `outputs/`。

生成器禁止将输出写入 `data/reference` 或其子目录。原始工作区的旧生成物留在源目录作为对照。

## 依赖边界

数据被 calibration 等组件读取。
