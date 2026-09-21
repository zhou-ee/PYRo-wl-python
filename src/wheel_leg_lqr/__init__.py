"""双轮腿机器人主机侧建模、LQR 调度与 RDOB 生成包。"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

__all__ = [
    "analysis",
    "artifacts",
    "control",
    "core",
    "numeric_config",
    "numeric_model",
    "rdob",
    "resolve_output_dir",
    "scheduling",
]


def resolve_output_dir(output_dir: str | Path) -> Path:
    """解析产物输出目录，并拒绝写入只读参考快照。

    Args:
        output_dir: 请求的输出目录（相对当前工作目录或绝对路径）。
            Requested output directory (relative to cwd or absolute).

    Returns:
        解析后的绝对 ``Path``。The resolved absolute ``Path``.

    Raises:
        ValueError: 若目录解析后位于冻结快照内部。
            If the directory resolves inside the frozen snapshot.
    """
    path = Path(output_dir).resolve()
    package_root = Path(__file__).resolve().parents[2]
    snapshot_root = package_root.parent / "host_code"
    try:
        path.relative_to(snapshot_root)
    except ValueError:
        return path
    raise ValueError(
        "output-dir must not point into the frozen snapshot "
        f"{snapshot_root}; use data/generated/ or an explicit temporary "
        "directory instead"
    )
