from __future__ import annotations
from ..artifacts.io import default_output_dir
import argparse
from ..artifacts.io import resolve_output_dir
from ..analysis.symbolic import generic_witness, _write_matrix


def main() -> None:
    """运行符号结构能控性检查，并将矩阵转储到 --output-dir。

    Run the symbolic structural check and dump the nominal input matrix and the
    controllability matrix into ``--output-dir`` (default ``outputs/symbolic``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=default_output_dir("symbolic"),
        help="Directory for the dumped matrices (default: outputs/symbolic).",
    )
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)

    result = generic_witness()
    print(
        f"symbolic S shape={result['symbolic_input_matrix'].shape}, "
        f"rank={result['symbolic_input_rank']}"
    )
    print(
        "C shape="
        f"{result['controllability_matrix'].shape}, "
        f"generic witness rank={result['controllability_rank']}"
    )
    print("Full rank 14 means the chosen generic mechanical witness is controllable.")
    print(
        "This is a structural witness, not the final robot result; replace M, D, K, S "
        "with each measured trim point."
    )

    _write_matrix(
        output_dir / "nominal_input_matrix.txt",
        result["symbolic_input_matrix"],
    )
    _write_matrix(
        output_dir / "generic_controllability_matrix.txt",
        result["controllability_matrix"],
    )

if __name__ == "__main__":
    main()
