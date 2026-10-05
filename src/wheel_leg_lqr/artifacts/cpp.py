"""Finite float32 C++ data formatting, no modeling or controller design."""
import numpy as np
def cpp_array(name, values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > np.finfo(np.float32).max):
        raise ValueError(f"{name} is not representable as finite float32")
    values = values.astype(np.float32)
    shape = "".join(f"[{size}]" for size in values.shape)
    def render(array, indent):
        if array.ndim == 1:
            return "{"+", ".join(f"{float(value):.9e}f" for value in array)+"}"
        return "{\n"+",\n".join(indent+"    "+render(row, indent+"    ") for row in array)+"\n"+indent+"}"
    return f"static constexpr float {name}{shape} = {render(values, '')};"
