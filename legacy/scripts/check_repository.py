"""Reject common credential formats and private runtime files before sharing.

This is a narrow publish check, not a guarantee that all sensitive data is absent.
Only file names and line numbers are printed when a credential is detected.
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'GitHub token': re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
    'Google API key': re.compile(r'AIza[A-Za-z0-9_-]{30,}'),
    'API secret': re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}'),
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
}


def private_path(relative):
    parts = relative.parts
    return (any(part in {'.venv', '.cache', '__pycache__', '.git'} for part in parts)
            or parts[:2] in {('data', 'workspaces'), ('reports', 'private')}
            or parts[:2] == ('deploy', 'volumes')
            or relative.name.startswith('.orbstack.env')
            or relative.name == 'user_feedback.jsonl'
            or (relative.name.startswith('.env') and relative.name != '.env.example')
            or relative.suffix in {'.pem', '.key', '.p12', '.pfx'}
            or ('data' in parts and 'index' in parts))


def main():
    result = subprocess.run(['git', 'ls-files', '--cached', '-z'], cwd=ROOT, capture_output=True)
    if result.returncode == 0:
        paths = [ROOT / name for name in result.stdout.decode().split('\0') if name]
    else:
        # Before git init, inspect the intended source/data/report surface only.
        paths = [p for p in ROOT.iterdir() if p.is_file() and not private_path(p.relative_to(ROOT))]
        for directory in ('src', 'tests', 'scripts', 'docs', 'data', 'reports', 'deploy', 'requirements', '.github'):
            paths.extend(p for p in (ROOT / directory).rglob('*')
                         if p.is_file() and not private_path(p.relative_to(ROOT)))
    errors = []
    for path in paths:
        relative = path.relative_to(ROOT)
        if private_path(relative):
            errors.append(f'{relative}: private/runtime path must not be tracked')
            continue
        if path.stat().st_size > 50 * 1024 * 1024:
            errors.append(f'{relative}: larger than the source-sharing limit (50 MiB)')
            continue
        for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    errors.append(f'{relative}:{number}: possible {label}')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'Publish check passed: {len(paths)} files; private paths and common credential formats checked')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
