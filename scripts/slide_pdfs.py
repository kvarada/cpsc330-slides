"""Plan and export opt-in lecture PDFs; run from the repository root."""
import concurrent.futures
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading

SITE = Path('website/_site')
CACHE = Path('.pdf-cache')
PLAN = Path('.pdf-plan.json')


def digest_files(paths):
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode())
        digest.update(b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def valid_pdf(path):
    return path.is_file() and path.read_bytes().startswith(b'%PDF-')


def plan():
    CACHE.mkdir(exist_ok=True)
    # Shared assets are intentionally conservative; per-deck figure directories
    # are hashed separately so a plot change doesn't invalidate every lecture.
    shared = []
    for folder in (SITE / 'slides', SITE / 'site_libs'):
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix not in ('.qmd', '.html', '.pdf'):
                if not any(p.endswith('_files') for p in path.relative_to(folder).parts):
                    shared.append(path)
    shared += [Path(__file__), Path('.github/workflows/publish.yml')]
    common = digest_files(shared)
    missing = []
    enabled = 0
    for page in sorted(Path('website').glob('lecture-*.qmd')):
        text = page.read_text()
        metadata = text.split('---', 2)[1]
        match = re.search(r'\{\{< revealjs "(slides/[^"/]+)\.html"', text)
        if not match:
            continue
        stem = match[1]
        pdf = SITE / (stem + '.pdf')
        if not re.search(r'^publish-pdf:\s*true\s*(?:#.*)?$', metadata, re.MULTILINE):
            pdf.unlink(missing_ok=True)
            continue
        enabled += 1
        html = SITE / (stem + '.html')
        assets = [html]
        assets += [p for p in (SITE / (stem + '_files')).rglob('*') if p.is_file()]
        key = hashlib.sha256((common + digest_files(assets)).encode()).hexdigest()
        cached = CACHE / (html.stem + '.pdf')
        manifest = CACHE / (html.stem + '.json')
        try:
            hit = json.loads(manifest.read_text()) == key and valid_pdf(cached)
        except (OSError, ValueError):
            hit = False
        if hit:
            shutil.copyfile(cached, pdf)
            print(f'Reusing {pdf}', flush=True)
        else:
            missing.append(dict(stem=stem, key=key))
    PLAN.write_text(json.dumps(missing))
    print(f'{enabled} enabled; {len(missing)} PDFs need export.', flush=True)
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            output.write(f'missing={len(missing)}\n')


def export():
    jobs = json.loads(PLAN.read_text())
    if not jobs:
        return
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def convert(job):
        stem = job['stem']
        pdf = SITE / (stem + '.pdf')
        print(f'Exporting {stem}', flush=True)
        subprocess.run(['decktape', 'reveal', '--chrome-arg=--no-sandbox',
                        '--load-pause', '2000',
                        f'http://127.0.0.1:{server.server_port}/{stem}.html', str(pdf)],
                       check=True, timeout=900)
        if not valid_pdf(pdf):
            raise RuntimeError(f'Invalid PDF: {pdf}')
        shutil.copyfile(pdf, CACHE / pdf.name)
        (CACHE / (pdf.stem + '.json')).write_text(json.dumps(job['key']))
        print(f'Finished {stem}', flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(convert, jobs))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    {'plan': plan, 'export': export}[sys.argv[1]]()
