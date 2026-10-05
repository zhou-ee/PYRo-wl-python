> 本文保留迁移前的结构构想；当前目标工作区已按此方案实施，实际接口见组件 README，验证结果见 migration_report.md。

# PYRo-wl-python 工作区迁移与代码结构方案

日期：2026-10-05。阶段：结构构想，尚未执行代码迁移、重构或功能删除。

源目录：`E:\Software\Agent\pack\PYRo-wl-python`。

目标目录：`E:\Software\Develop\RM\Wheel_leg\PYRo-wl-python`。检查时目录存在且为空；本阶段继续保持为空。

## 1. 设计目标

1. 同一台机器人的物理参数只有一个维护入口，地面、腾空、接触模型共享这份参数。
2. 各模型保留自己的状态定义、输入定义、约束、假设和控制参数；共享参数不意味着共享动力学方程。
3. 动力学计算、控制器设计、腿长调度、仿真分析、文件导出和命令行分开。
4. 删除已弃用的前向模型加速度估计功能及其配套入口、测试、文档和产物。
5. 保留地面 LQR、腾空 LQR、RDOB、带宽/能控性/QR 优化、下台阶仿真与 CAD 质心拟合能力。
6. 迁移以源目录当前文件内容为准，包含未提交修改；旧目录保留为回退参考。

## 2. 现有代码中的具体问题

| 位置 | 当前问题 | 调整方向 |
|---|---|---|
| `core/config.py` | 混合物理参数、状态顺序、Q/R、执行器限制、调度阶数及配平计算 | 分别进入公共物理配置、模型配置、配平模块、调度设置 |
| `airborne/config.py` | 重复定义质量、机体惯量、质心、腿长范围和腿质心多项式 | 从公共物理参数读取所需分量 |
| `core/model.py` | 符号编译读取全局 `PARAMETERS`，无参数的单项缓存无法区分不同机器人配置 | 给模型实例绑定不可变物理参数，缓存随实例或物理参数指纹区分 |
| `airborne/model.py` | `linearize_airborne()` 同时生成物理矩阵、Q/R 和 Q 相关的形状投影 | 物理线性化与控制空间构造分开 |
| `control/lqr.py` | 默认读取地面模型的 Q/R 和采样周期 | Q/R、采样周期由调用者显式传入，支持不同状态维数 |
| `control/rdob.py` | 直接依赖地面模型的固定维数 | 矩阵算法从输入形状推断维数，六通道部署约束放在地面 RDOB 调度层 |
| `control/saturation.py` | 通用仿真和十四状态初始扰动用例混在一起 | 通用仿真进入 `simulation`，特定场景用例在地面配置或分析中声明 |
| `scheduling/rdob_schedule.py` | 反向导入顶层 `rdob.py` 中的节点设计和默认值 | 节点设计与 RDOB 配置归入调度层，CLI 只向下调用 |
| 地面/腾空调度 | 重复实现离散化、能控性、张量拟合、输出路径和写盘 | 共用无场景依赖的基础算法，分别保留验证规则 |
| `analysis/step_drop.py` | 同时承担九坐标接触模型、仿真和 CLI | 力学进 `models/contact`，时间推进进 `simulation`，入口进 `cli` |
| `data_process/com_fit.py` | 独立于安装包，重复硬编码质量，依赖未进入项目声明的 pandas | 移入 `calibration`，使用公共参数，补充 CAD 可选依赖 |
| 顶层兼容模块和包初始化 | 多处 `import *`、导入包时加载多个功能模块 | 内部统一真实导入路径，包初始化保持轻量 |
| 输出路径 | 相对路径依赖工作目录，腾空另有重复实现，部分启动脚本强行追加输出参数 | 使用一个路径解析入口，用户显式路径优先，默认路径明确 |

源目录当前已有 5 个未提交修改：`README.md`、`docs/model_acceleration.md`、`core/config.py`、`airborne/config.py` 和 `model_acceleration.py`。迁移不能仅从 Git 提交导出，以免丢失这些内容；其中属于弃用功能的文件按本方案不进入最终工作区。

## 3. 建议目标结构

保留现有 Python 包名 `wheel_leg_lqr`，项目目录使用 `PYRo-wl-python`，避免同时更改路径、包名与行为。

```text
PYRo-wl-python/
├─ pyproject.toml
├─ README.md
├─ FILE_MAP.md
├─ src/wheel_leg_lqr/
│  ├─ __init__.py                  # 版本与少量公开对象，避免批量导入
│  ├─ config.py                    # 机器人物理参数唯一入口
│  ├─ models/
│  │  ├─ leg_geometry.py           # 腿质心曲线与公共几何计算
│  │  ├─ ground/
│  │  │  ├─ config.py              # 14 状态/6 输入、地面 Q/R、默认工作点
│  │  │  ├─ model.py               # 地面动力学及连续线性化
│  │  │  └─ trim.py                # 平衡角、配平几何和配平输入
│  │  ├─ airborne/
│  │  │  ├─ config.py              # 6 状态/2 输入、腾空 Q/R、模型假设
│  │  │  ├─ model.py               # 无接触动力学与物理线性化
│  │  │  └─ reduction.py           # 形状空间、守恒约束、加权投影
│  │  └─ contact/
│  │     ├─ config.py              # 接触刚度、阻尼、释放和碰撞场景设置
│  │     └─ model.py               # 九坐标运动学、质量及接触力
│  ├─ control/
│  │  ├─ lqr.py                    # 离散化、DARE/LQR、数值能控性
│  │  └─ rdob.py                   # RDOB 矩阵设计、初始化与离散更新
│  ├─ scheduling/
│  │  ├─ domain.py                 # 腿长网格、归一化、采样和边界检查
│  │  ├─ polynomial.py             # 张量积/总阶拟合与数值求值
│  │  ├─ ground_lqr.py             # 地面节点设计、增益调度和验证
│  │  ├─ airborne_lqr.py           # 腾空形状 LQR、增益调度和验证
│  │  └─ rdob.py                   # 六通道节点设计、配置、调度和验证
│  ├─ simulation/
│  │  ├─ saturation.py             # 饱和闭环时间推进与验证
│  │  └─ step_drop.py              # 释放、碰撞、恢复仿真
│  ├─ analysis/
│  │  ├─ symbolic.py
│  │  ├─ bandwidth.py
│  │  ├─ qr_optimizer.py
│  │  └─ airborne_fl_pitch.py      # 腾空腿力对 pitch 的影响分析
│  ├─ calibration/
│  │  └─ leg_com.py                # CAD 数据清洗、质心拟合、误差报告
│  ├─ artifacts/
│  │  ├─ io.py                    # 路径与统一写盘
│  │  ├─ cpp.py                   # 公共 C++ 数组格式化
│  │  ├─ ground_lqr.py            # 地面 LQR 导出格式
│  │  ├─ airborne_lqr.py          # 腾空 LQR 导出格式
│  │  └─ rdob.py                  # RDOB 导出格式
│  └─ cli/                        # 参数解析、加载配置、调用与结果展示
├─ scripts/                       # 需要保留的 Windows 双击启动器
├─ data/
│  └─ reference/                  # CAD 等输入原始数据，保留来源说明
├─ outputs/                       # 生成物，不提交 Git
│  ├─ ground_lqr/
│  ├─ airborne_lqr/
│  └─ rdob/
├─ tests/                         # 按物理模型/控制/调度/导出分组
└─ docs/
```

目录用于明确现有职责，不要求每项操作都新增抽象。每个功能直接使用函数和少量数据类，不建立插件注册系统或多层基类。

## 4. 统一物理配置入口

### 4.1 公共 `config.py` 放什么

建议先采用 Python 文件与不可变 dataclass，无需额外增加 YAML/TOML 加载器。

| 参数组 | 内容 |
|---|---|
| 机体 | 质量、完整 3×3 惯量张量、相对髋关节的三维质心位置 |
| 腿 | 质量、可用腿长范围、CAD 质心曲线系数及有效范围 |
| 轮 | 质量、半径、滚动惯量 |
| 布局 | 髋半轮距、轮半轮距、坐标轴和正方向约定 |
| 被动机构 | 气弹簧力曲线、归一化区间 |
| 执行器硬件 | 力矩常数、电流上限、髋力矩上限、主动腿力上限 |
| 环境 | 重力，允许实验时显式替换 |

所有参数注明 SI 单位、坐标系、惯量参考点、单侧/整机含义以及数据来源。当前动力学同时使用质心平移项和机体转动惯量；统一时必须确认 CAD 惯量的参考点，不能仅凭相同数值合并语义。

腾空需要的 `body_pitch_inertia` 由完整惯量张量的对应轴分量获得，质心 x/z 分量从同一个三维向量读取，不再手工复制。

公共配置不包含 Q/R、状态编号、调度步长、多项式阶数、RDOB 带宽、补偿开关或接触模型刚度。执行器硬件极限与某个控制器额外采用的软限幅要区分。

### 4.2 派生计算放哪里

- 质心多项式求值在 `models/leg_geometry.py`，系数在公共配置。
- 配平角、配平输入和静态工作点在 `models/ground/trim.py`，显式使用物理参数。
- 气弹簧曲线求值及主动/总输入转换给出一个公共实现；沿用当前约定：气弹簧不加入现有 A/B/K，只修正主动前馈。
- `beta_eq(L)` 属于质量与质心曲线共同决定的派生量。现有固件导出的 `LEG_COM_BETA_EQ_POLY` 应由当前物理配置生成并验证，不继续作为独立手写物理常量。
- 未被使用的 `LEG_COM_ALPHA_POLY` 可在迁移审查中移出运行配置，保留需要的 CAD 拟合报告。

### 4.3 各模型配置放什么

- `models/ground/config.py`：14 状态/6 输入名称、单位、默认工作点、地面 Q/R、采样周期。
- `models/airborne/config.py`：6 状态/2 输入名称、单位、腾空 Q/R、采样周期、无接触假设。
- `models/contact/config.py`：接触假设、接触刚度/阻尼、释放判据、仿真数值边界。
- 调度模块：扫描范围和步长、拟合阶数、误差阈值、验证采样设置。物理允许腿长范围仍来自公共配置。
- RDOB 调度模块：六通道带宽/极点、采样周期、扰动限幅比例、补偿选择等设计设置。

状态定义和 Q/R 放在同一模型的配置中便于调参，但 `model.py` 只使用物理与状态定义，不读取控制权重。公共 LQR 函数必须显式收到 Q/R。

## 5. 参数传递与依赖边界

公共 `config.py` 是唯一物理参数维护入口；入口创建一次配置对象，并把它传给模型和后续计算。避免每个函数自行获取可修改的全局配置。

```python
from wheel_leg_lqr.config import DEFAULT_ROBOT
from wheel_leg_lqr.models.ground.config import DEFAULT_GROUND
from wheel_leg_lqr.models.ground.model import GroundModel
from wheel_leg_lqr.control.lqr import discretize_state_space, solve_discrete_lqr

# 结构示意，尚未实现。
model = GroundModel(DEFAULT_ROBOT)
plant = model.linearize(l1=0.24, l2=0.32)
ad, bd = discretize_state_space(plant.A, plant.B,
                              sample_time=DEFAULT_GROUND.sample_time)
gain, cost, poles = solve_discrete_lqr(ad, bd,
                                     DEFAULT_GROUND.make_q(),
                                     DEFAULT_GROUND.make_r())
```

约束如下：

1. `config` 不依赖任何动力学、控制、调度或导出模块。
2. `models` 依赖公共物理配置和模型内部定义，不反向依赖控制、调度或 CLI。
3. `control` 只处理传入矩阵和数值参数，不导入地面/腾空配置。
4. `scheduling` 调用模型、控制算法和公共拟合，不解析命令行，不写文件。
5. `simulation` 调用模型与控制算法；`analysis` 组织场景和指标，不承担通用动力学实现。
6. `artifacts` 格式化已有结果和元数据，不重新建模或设计控制器。结果对象/数组由入口传入，避免导出器与调度器互相导入。
7. `cli` 负责组装上述模块、参数覆盖、输出路径和用户提示；下层不能导入 `cli` 或启动脚本。

地面和腾空共享物理参数，共用离散化和拟合算法，分别建立接触与无接触方程。腾空仍保留四维可控形状子空间及两维整体转动自由模态；不套用地面的全状态闭环稳定性判据。

地面缓存绑定模型实例，或采用包含全部物理参数及质心曲线的稳定指纹。Q/R 不进入物理编译缓存键。配置中的向量和矩阵优先用 tuple 存储，在计算边界转为 ndarray，避免 frozen dataclass 内的 ndarray 仍可被原地修改。

## 6. 加速度估计删除范围

完整移除以下功能文件：

- `run_model_acceleration.py`
- `src/wheel_leg_lqr/model_acceleration.py`
- `src/wheel_leg_lqr/scheduling/model_acceleration_schedule.py`
- `tests/test_model_acceleration.py`
- `docs/model_acceleration.md`
- `data/generated/model_acceleration/` 下全部功能产物，不迁入目标目录。

同步清理 `pyproject.toml` 中的 `wheelleg-model-acceleration` 入口、包公开列表、README、FILE_MAP 和其他失效引用。

删除范围是前向模型加速度估计及其系数调度。动力学中的加速度项、RDOB、接触仿真、`compute_leg_force_pitch_effect()` 和腾空 FL→pitch 分析仍是保留功能。公共总阶三次拟合也必须保留，因为 RDOB 的 Bq 仍在使用。

源目录保留原状；这里的移除指目标工作区的最终内容。固件仓库 `E:\Software\Develop\RM\Wheel_leg\PYRo-wl` 不在此次 Python 工作区修改范围内。

## 7. 数据、导出与路径约定

- 原始 CAD Excel 移入 `data/reference`；拟合脚本在 `calibration` 中，按需安装 pandas/openpyxl，避免建模基础安装承担 Excel 依赖。
- 旧生成物不作为新参数下的有效产物直接交付。源目录保留它们作为对照；目标目录从新结构重新生成。
- 每类功能使用独立输出目录，避免互相覆盖。保留既有 C++ 头文件名、命名空间、状态/输入顺序、系数项序、float32 求值方式和主动/总输入约定。
- NPZ/JSON 元数据记录物理参数快照或指纹、控制配置、采样周期、模型与导出 schema 版本、拟合归一化范围及验证摘要。离线产物求值使用自己的归一化参数，不能重新读取当前全局配置。
- 仓库启动器默认输出到项目根下的 `outputs/<功能>`，显式 `--output-dir` 优先；已安装 CLI 在非仓库环境使用当前目录下的 `outputs/<功能>`。所有规则由一个路径函数实现并在文档中说明。
- 移除当前以相邻 `host_code` 目录为前提的路径假设，依据新的参考数据路径设置必要的写入保护。
- 启动器只处理未安装运行的导入路径和命令转发。命令行执行不自动阻塞等待 Enter；需要双击暂停时单独提供明确选项。

## 8. 迁移顺序和验收

### 第一步：固定源代码基线

记录文件清单及当前修改，运行保留功能的测试并记录结果。在源目录建立地面/腾空非对称腿长等代表节点的数值基线。遇到源代码已有失败，单独记录，不把它归因于迁移。

### 第二步：建立干净目标副本

迁移当前源码、项目声明、文档、测试和 CAD 输入数据，排除缓存、IDE 文件、旧生成物及弃用加速度功能。源目录保留，迁移完成前不删除。目标 Git 的建立/历史保留方式单独处理；不把 `.git` 当普通源码目录复制，也不修改源目录历史。

### 第三步：统一物理配置

将地面、腾空、接触及 CAD 工具连接到同一配置。保持当前有效参数值，尤其保留未提交的机体质量修改。先比对质量矩阵、A/B、配平、增益和 RDOB 矩阵，再继续拆职责。

### 第四步：逐项解耦

先清除控制算法对地面配置的依赖，再拆腾空物理线性化与加权投影、RDOB 节点设计与 CLI，最后分离调度/导出及接触模型/仿真。每次以数值对照确认行为，再做下一项。

内部代码和测试使用新真实路径，不保留两套实现。根据实际外部使用情况，可暂留少量显式兼容门面；既有 CLI 名称优先保留。弃用加速度入口不提供兼容门面。

### 第五步：重新生成并验收

1. 保留功能测试通过，目标可以独立安装和运行。
2. 原参数下代表点的 M/A/B、配平、K、RDOB 矩阵与源基线按各对象的数值误差容差一致，不要求含时间或路径元数据的文件逐字节一致。
3. 统一修改机体质量后，地面、腾空和接触模型都响应；只改地面 Q/R 时，物理矩阵及腾空控制结果不受影响。
4. 先后使用两份物理配置，地面符号编译缓存不会串用。
5. 地面闭环、腾空形状子空间、RDOB 匹配恒等式/冻结点检查、拟合误差及 float32 验证保持各自原有判据。
6. 各保留 CLI 的帮助与小规模计算可用；从其他工作目录启动时，输出规则和显式路径覆盖正确。
7. 生成 C++ 数据契约保持一致；对状态顺序、系数项序和 float32 求值的已有有效验证继续保留。
8. 加速度估计源码、入口和产物不再出现在目标工作区；除明确说明删除范围的迁移文档外，不留下失效功能引用。
9. README、FILE_MAP、安装路径与运行示例更新为目标工作区；源目录保留可回退。

现有 `test_ground_lqr_parameters_cannot_change_airborne_gain` 将物理参数和 Q/R 独立性合并测试。统一物理配置后应拆成“共享物理参数影响两个模型”与“地面控制权重不影响腾空”两项有实际意义的验证。

## 9. 本阶段结论

优先采用“一个机器人物理配置 + 各模型独立配置 + 通用矩阵算法 + 独立调度与导出”的结构。

本阶段只记录设计，不修改物理参数、执行代码删除或填充目标目录。后续实现按本方案逐步迁移，以现有数值行为和固件数据契约作为验收依据。
