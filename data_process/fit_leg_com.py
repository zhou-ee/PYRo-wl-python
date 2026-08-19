"""Compatibility launcher for :mod:`data_process.com_fit`."""

try:
    from .com_fit import *
except ImportError:  # pragma: no cover - direct file execution compatibility
    from com_fit import *

if __name__ == "__main__":
    main()
