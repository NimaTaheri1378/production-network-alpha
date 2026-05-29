from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

FORBIDDEN_PREFIXES = [
    'data/raw/', 'data/interim/', 'data/processed/', 'data/protected/',
    'logs/', 'artifacts/processed/', 'artifacts/model_runs/', 'artifacts/logs/',
    'artifacts/schema_discovery/', 'artifacts/schema_drilldown/',
    'artifacts/github_release/', 'artifacts/robustness_decision/',
]
ALLOWED_ARTIFACT_PREFIXES = ['artifacts/release_public/']
FORBIDDEN_NAMES = {'.pgpass', '.env'}
FORBIDDEN_SUFFIXES = {'.parquet', '.feather', '.arrow', '.pickle', '.pkl', '.sqlite', '.db', '.log', '.pyc', '.pyo'}
MAX_TRACKED_SIZE_MB = 50
LOCAL_PATTERNS = [
    '/' + 'home/' + 'nt' + '612',
    '/' + 'cache/' + 'home/' + 'nt' + '612',
    'nt' + '612',
    'gpun' + '001',
    'amarel' + 'n',
    'nt' + '***' + '2',
]
SECRET_PATTERNS = [
    'BEGIN ' + 'PRIVATE KEY',
    'BEGIN RSA ' + 'PRIVATE KEY',
    'github' + '_pat_',
    'gh' + 'p_',
    's' + 'k-',
]
TEXT_BINARY_OK = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.pdf'}


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def iter_files(root: Path) -> list[str]:
    git = run(['git', '-C', str(root), 'ls-files'])
    if git.returncode == 0:
        return [line.strip() for line in git.stdout.splitlines() if line.strip()]
    print('[WARN] git ls-files failed; checking filesystem policy only.')
    return [p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bad: list[str] = []

    for rel in iter_files(root):
        path = root / rel
        name = path.name
        suffix = path.suffix.lower()

        if name in FORBIDDEN_NAMES:
            bad.append(rel)
        if suffix in FORBIDDEN_SUFFIXES:
            bad.append(rel)
        for pref in FORBIDDEN_PREFIXES:
            if rel.startswith(pref):
                if rel.endswith('/.gitkeep') or rel.startswith('examples/synthetic_demo/'):
                    continue
                if any(rel.startswith(ap) for ap in ALLOWED_ARTIFACT_PREFIXES):
                    continue
                bad.append(rel)
        if path.exists() and path.is_file() and path.stat().st_size > MAX_TRACKED_SIZE_MB * 1024 * 1024:
            bad.append(f'{rel} [large:{path.stat().st_size}]')

        if path.exists() and suffix not in TEXT_BINARY_OK:
            try:
                text = path.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                text = ''
            for pat in LOCAL_PATTERNS + SECRET_PATTERNS:
                if pat in text:
                    bad.append(f'{rel} [contains:{pat}]')

    if bad:
        print('[FATAL] Files violate public-release policy:')
        for item in sorted(set(bad)):
            print('  -', item)
        return 2
    print('[OK] Public release preflight passed. No forbidden tracked files or local leaks detected.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
