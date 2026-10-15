#!/usr/bin/env python3
"""Apply curated cross-note backlinks to Literature-Vault (idempotent).

Run from repository root: python tools/apply-literature-links.py
"""
from pathlib import Path
import json,sys,re
ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'tools/literature-relations.json').read_text(encoding='utf-8'))
errors=[]
for key,path in data['notes'].items():
    f=ROOT/path
    if not f.exists(): errors.append('missing note: '+path)
for key,path in data['topics'].items():
    if not (ROOT/path).exists(): errors.append('missing topic: '+path)
if errors:
    print('\n'.join(errors),file=sys.stderr);sys.exit(1)
count=0
for key,path in data['notes'].items():
    f=ROOT/path
    raw=f.read_bytes()
    old=raw.decode('utf-8')
    section=data['footers'][key]
    heading='## 跨文献关联导航（2026-10-15 整理）'
    if heading in old:
        continue
    eol='\r\n' if b'\r\n' in raw[:4096] else '\n'
    additional=('' if raw.endswith(b'\n') else eol)+section.replace('\n',eol)
    f.write_bytes(raw+additional.encode('utf-8'))
    count+=1
print(f'Updated {count} notes; {len(data["notes"])-count} already linked.')
