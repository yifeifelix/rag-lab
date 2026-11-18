"""Ensure distribution contains runtime modules and the browser UI."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
wheels = list((root / 'dist').glob('*.whl'))
assert len(wheels) == 1, 'Clean dist/ before building a new release'
with zipfile.ZipFile(wheels[0]) as wheel:
    names = set(wheel.namelist())
    for module in (root / 'src/rag_platform').glob('*.py'):
        name = 'rag_platform/' + module.name
        assert name in names, f'Missing module: {name}'
        assert wheel.read(name) == module.read_bytes(), f'Stale module: {name}'
    assert 'rag_platform/static/index.html' in names, 'Missing browser UI'
    assert wheel.read('rag_platform/static/index.html') == (root / 'src/rag_platform/static/index.html').read_bytes()
print(f'Wheel source and UI contents verified: {wheels[0].name}')
