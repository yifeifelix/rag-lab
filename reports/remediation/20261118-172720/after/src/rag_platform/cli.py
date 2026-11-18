"""Explicit process bootstrap for source CLI tools."""
from pathlib import Path
import runpy
import sys

from .runtime import configure_runtime


def main():
    configure_runtime()
    if len(sys.argv) < 2:
        raise SystemExit('Expected a script under scripts/')
    name = sys.argv.pop(1)
    if Path(name).name != name or not name.endswith('.py'):
        raise SystemExit('Expected a script filename')
    root = Path(__file__).resolve().parents[2]
    sys.argv[0] = str(root / 'scripts' / name)
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
