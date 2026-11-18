"""Repeatable offline checks and audit artifacts; does not call paid providers."""

import argparse
import ast
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from rag_platform.config import RAGConfig
from rag_platform.provenance import sha256_file


def record_manifest(root, output):
    paths = set()
    for folder in ('src/rag_platform','scripts','tests','docs','deploy','data','reports','dist'):
        for path in (root/folder).rglob('*'):
            relative = path.relative_to(root)
            if (path.is_file() and not any(part in ('__pycache__','index','volumes','audit')
                                          for part in relative.parts)):
                paths.add(path)
    paths.update(root/name for name in ('README.md','pyproject.toml','.env.example','.gitignore'))
    paths.update(p for p in output.rglob('*') if p.is_file() and 'before' not in p.relative_to(output).parts
                 and p.name != 'final_manifest.json')
    files = {str(p.relative_to(root)):sha256_file(p) for p in sorted(paths)}
    baseline_file = output/'baseline_manifest.json'
    baseline = json.loads(baseline_file.read_text()) if baseline_file.exists() else {}
    integrity = all((output/'before'/name).exists() and sha256_file(output/'before'/name) == expected
                    for name,expected in baseline.items())
    changes = {name:'deleted' if not (root/name).exists() else 'modified'
               for name,expected in baseline.items()
               if not (root/name).exists() or sha256_file(root/name) != expected}
    payload = {'timestamp_utc':datetime.now(timezone.utc).isoformat(), 'files':files,
               'baseline_snapshot_integrity':integrity if baseline else None,
               'baseline_files':len(baseline),'changes_against_baseline':changes,
               'note':'Local hashes provide change tracking, not tamper-proof independent attestation'}
    (output/'final_manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    return payload


def main():
    root = RAGConfig().project_root
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=root/'reports/audit'/datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S'))
    parser.add_argument('--manifest-only', action='store_true', help='Refresh hashes after documentation or review edits')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(root):
        parser.error('Audit output must be inside the project')
    output.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        manifest = record_manifest(root, output)
        print('Recorded file hashes:', len(manifest['files']))
        return 1 if manifest['baseline_snapshot_integrity'] is False else 0
    result = {'timestamp_utc':datetime.now(timezone.utc).isoformat(), 'scope':'offline_implementation_checks',
              'real_model_validated':False, 'browser_rendering_validated':False,
              'http_network_load_validated':False}
    python_paths = [p for folder in ('src/rag_platform','scripts','tests') for p in (root/folder).rglob('*.py')]
    for path in python_paths:
        ast.parse(path.read_text(), filename=str(path))
    result['python_syntax'] = {'passed':True, 'files':len(python_paths)}
    page = (root/'src/rag_platform/static/index.html').read_text()
    script = re.search(r'<script>(.*?)</script>',page,re.S).group(1)
    script_path = output/'workbench_ui.js'
    script_path.write_text(script)
    node = shutil.which('node')
    if node:
        checked = subprocess.run([node,'--check',str(script_path)],capture_output=True,text=True)
        (output/'javascript_check.log').write_text(checked.stdout+checked.stderr)
        result['javascript_syntax'] = {'passed':checked.returncode == 0, 'returncode':checked.returncode}
    else:
        result['javascript_syntax'] = {'passed':False,'reason':'node unavailable'}
    command = [sys.executable,'-m','pytest','-q','-p','no:cacheprovider',f'--junitxml={output / "pytest.xml"}']
    tested = subprocess.run(command,cwd=root,capture_output=True,text=True)
    (output/'pytest.log').write_text(tested.stdout+tested.stderr)
    print(tested.stdout+tested.stderr)
    result['pytest'] = {'passed':tested.returncode == 0,'returncode':tested.returncode,'command':command}
    wheels = list((root/'dist').glob('*.whl'))
    result['wheel'] = {'passed':False,'reason':'No wheel built'}
    if wheels:
        wheel = max(wheels,key=lambda p:p.stat().st_mtime_ns)
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            matches = all(archive.read(str(p.relative_to(root/'src'))) == p.read_bytes()
                          for p in (root/'src/rag_platform').rglob('*')
                          if p.is_file() and p.suffix in ('.py','.html') and '__pycache__' not in p.parts)
            clean = not any('/.env' in n or '/.cache/' in n or n.startswith('data/') for n in names)
        result['wheel'] = {'passed':matches and clean,'path':str(wheel.relative_to(root)),
                           'sha256':sha256_file(wheel),'matches_current_source':matches,
                           'static_ui_included':'rag_platform/static/index.html' in names,
                           'no_runtime_data_or_secrets_packaged':clean}
    deps = sorted({(d.metadata['Name'],d.version) for d in distributions() if d.metadata['Name']})
    (output/'installed_dependencies.txt').write_text('\n'.join(f'{name}=={version}' for name,version in deps)+'\n')
    result['passed'] = all(result[key]['passed'] for key in ('python_syntax','javascript_syntax','pytest','wheel'))
    (output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    record_manifest(root, output)
    print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
