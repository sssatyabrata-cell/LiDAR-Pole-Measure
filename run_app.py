from pathlib import Path
import sys


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def main() -> int:
    _add_src_to_path()
    from lidar_pole_measure.main import run
    from lidar_pole_measure.ui import run_ui

    if len(sys.argv) > 1:
        return run()
    run_ui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
