# SPDX-License-Identifier: MIT
"""Check local Markdown links and heading anchors without network access."""
import re
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def markdown_text(path):
    text = path.read_text(encoding='utf-8-sig')
    return re.sub(r'^(`{3,}|~{3,}).*?^\1\s*$', '', text,
                  flags=re.MULTILINE | re.DOTALL)


def anchors(text):
    result = set(re.findall(r'\b(?:id|name)=["\']([^"\']+)["\']', text))
    counts = Counter()
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', text, re.MULTILINE):
        heading = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', heading)
        heading = re.sub(r'<[^>]+>', '', heading).strip().lower()
        slug = ''.join(c for c in heading if c.isalnum() or c in ' _-').replace(' ', '-')
        suffix = f'-{counts[slug]}' if counts[slug] else ''
        result.add(slug + suffix)
        counts[slug] += 1
    return result


def main():
    listed = subprocess.check_output(
        ['git', '-C', str(ROOT), 'ls-files', '-z', '--cached', '--others', '--exclude-standard'])
    published = {ROOT / name.decode('utf-8') for name in listed.split(b'\0') if name}
    published = {p for p in published if p.is_file()}
    files = sorted(p for p in published if p.suffix == '.md')
    errors = []
    checked = 0
    documents = {p: markdown_text(p) for p in files if p.is_file()}
    for path, text in documents.items():
        for match in re.finditer(r'\]\(([^\s()]+)\)', text):
            url = urlsplit(match.group(1).strip('<>'))
            if url.scheme or url.netloc:
                continue
            checked += 1
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            issue = None
            if not target.is_relative_to(ROOT):
                issue = 'link escapes the repository'
            elif not target.exists():
                issue = 'missing target'
            elif target.is_file() and target not in published:
                issue = 'target is not tracked or publishable'
            elif target.is_dir() and not any(p.is_relative_to(target) for p in published):
                issue = 'directory has no published files'
            elif url.fragment and target.suffix.lower() == '.md':
                body = documents.get(target)
                if body is None:
                    body = markdown_text(target)
                if unquote(url.fragment) not in anchors(body):
                    issue = 'missing heading anchor'
            if issue:
                line = text.count('\n', 0, match.start()) + 1
                errors.append(f'{path.relative_to(ROOT)}:{line}: {issue}: {match.group(1)}')
    for error in errors:
        print(error)
    print(f'{len(documents)} Markdown files, {checked} local links, {len(errors)} errors.')
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
