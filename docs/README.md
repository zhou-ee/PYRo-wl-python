# 设计与使用文档

`workspace_restructure_plan.md` 是迁移前的设计记录，执行结果见 `migration_report.md`。RDOB、腾空 LQR、单腿下台阶专题文档保留，代码路径和运行命令更新至新结构。

组件接口以各目录 README 和源码为准；历史专题中的实验数值只对其记录的参数快照有效。

最新等参回归对照见 [migration_comparison.md](migration_comparison.md)，可复现工具见 [validation/README.md](validation/README.md)。

机体前后、左右对称对应的 CAD 惯量导出误差修正见 [body_symmetry_correction.md](body_symmetry_correction.md)。该参数修正发生在迁移对照之后，与保留历史参数的迁移等参验证分别记录。

## 依赖边界

文档链接与 FILE_MAP 共同提供模块导航。
