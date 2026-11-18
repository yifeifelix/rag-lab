"""Load local settings and route writable runtime paths to the Data volume."""

from pathlib import Path
import os
import shlex
import tempfile


def configure_runtime(project_root: Path | None = None) -> dict[str, str]:
    root = project_root or Path(os.getenv('RAG_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
    root = root.resolve()
    if not root.is_relative_to(Path('/Volumes/Data')):
        raise ValueError('RAG_PROJECT_ROOT must reside under /Volumes/Data')
    env_file = root / '.env'
    if env_file.exists():
        for number, line in enumerate(env_file.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:]
            key, sep, value = line.partition('=')
            if not sep or not key.strip().isidentifier():
                raise ValueError(f'Invalid .env assignment on line {number}')
            parts = shlex.split(value, comments=True)
            if len(parts) > 1:
                raise ValueError(f'Quote .env values containing spaces (line {number})')
            os.environ.setdefault(key.strip(), parts[0] if parts else '')
    defaults = {
        'HF_HOME': root / '.cache/huggingface',
        'TORCH_HOME': root / '.cache/torch',
        'UV_CACHE_DIR': root / '.cache/uv',
        'PIP_CACHE_DIR': root / '.cache/pip',
        'XDG_CACHE_HOME': root / '.cache',
        'TMPDIR': root / '.cache/tmp',
        'PYTHONPYCACHEPREFIX': root / '.cache/pycache',
    }
    paths = {}
    for key, default in defaults.items():
        path = Path(os.environ.get(key, str(default))).expanduser().resolve()
        if not path.is_relative_to(Path('/Volumes/Data')):
            path = default.resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = paths[key] = str(path)
    tempfile.tempdir = paths['TMPDIR']
    return paths
