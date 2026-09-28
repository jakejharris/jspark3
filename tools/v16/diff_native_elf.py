#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Locate byte, section and symbol differences without executing either ELF."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "recipe/scripts"))
import _diagnostics as diagnostics
import argparse
from collections import Counter
import hashlib
import json
import struct


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Elf:
    def __init__(self, path):
        self.data = Path(path).read_bytes()
        if self.data[:6] != b'\x7fELF\x02\x01':
            raise ValueError('requires little-endian ELF64 (AArch64 or x86-64)')
        header = struct.unpack_from('<16sHHIQQQIHHHHHH', self.data)
        self.machine = header[2]
        offset, stride, count, strings = header[6], header[11], header[12], header[13]
        if stride != 64 or not 0 < strings < count or offset + count * stride > len(self.data):
            raise ValueError('invalid or unsupported ELF section table')
        rows = [struct.unpack_from('<IIQQQQIIQQ', self.data, offset + i * stride) for i in range(count)]
        names = self.slice(rows[strings][4], rows[strings][5])
        self.sections = []
        for row in rows:
            self.sections.append(dict(name=self.string(names, row[0]), type=row[1], address=row[3],
                                      offset=row[4], size=row[5], link=row[6], stride=row[9],
                                      data=b'' if row[1] == 8 else self.slice(row[4], row[5])))

    def slice(self, offset, size):
        if offset + size > len(self.data):
            raise ValueError('ELF section exceeds file')
        return self.data[offset:offset + size]

    @staticmethod
    def string(data, offset):
        if offset >= len(data):
            raise ValueError('ELF string exceeds table')
        return data[offset:data.index(b'\0', offset)].decode('utf-8', errors='replace')

    def symbols(self):
        result = Counter()
        for table in self.sections:
            if table['type'] not in (2, 11):
                continue
            if table['stride'] != 24 or table['link'] >= len(self.sections):
                raise ValueError('invalid ELF symbol table')
            strings = self.sections[table['link']]['data']
            for offset in range(0, len(table['data']), 24):
                name, info, other, index, value, size = struct.unpack_from('<IBBHQQ', table['data'], offset)
                name = self.string(strings, name)
                content = None
                if size and 0 < index < len(self.sections):
                    section = self.sections[index]
                    start = value - section['address']
                    if 0 <= start and start + size <= len(section['data']):
                        content = digest(section['data'][start:start + size])
                result[(table['name'], name, info, other, index, value, size, content)] += 1
        return result


def byte_changes(left, right, limit=32):
    count, examples = abs(len(left) - len(right)), []
    for offset, (a, b) in enumerate(zip(left, right)):
        if a != b:
            count += 1
            if len(examples) < limit:
                examples.append({'offset': offset, 'left': a, 'right': b})
    return {'different_bytes': count, 'first_differences': examples,
            'length_delta': len(right) - len(left)}


def compare(left, right):
    a, b = Elf(left), Elf(right)
    changed, identical = [], []
    sa, sb = ({s['name']: s for s in elf.sections} for elf in (a, b))
    for name in sorted(sa.keys() | sb.keys()):
        x, y = sa.get(name), sb.get(name)
        if x and y and x['data'] == y['data'] and x['size'] == y['size']:
            identical.append(name)
            continue
        row = {'name': name}
        for label, section in (('left', x), ('right', y)):
            row[label] = None if section is None else dict(offset=section['offset'], size=section['size'],
                                                           sha256=digest(section['data']))
        if x and y:
            row.update(byte_changes(x['data'], y['data']))
        changed.append(row)
    symbols_a, symbols_b = a.symbols(), b.symbols()
    symbol_changes = {}
    for label, difference in (('left_only', symbols_a - symbols_b), ('right_only', symbols_b - symbols_a)):
        symbol_changes[label + '_count'] = sum(difference.values())
        symbol_changes[label] = [dict(table=row[0], name=row[1], info=row[2], visibility=row[3],
                                     section=row[4], value=row[5], size=row[6], content_sha256=row[7], count=n)
                                 for row, n in sorted(difference.items(), key=lambda item: repr(item[0]))[:64]]
    return {'identical': a.data == b.data,
            'left': {'path': str(left), 'sha256': digest(a.data), 'size': len(a.data), 'machine': a.machine},
            'right': {'path': str(right), 'sha256': digest(b.data), 'size': len(b.data), 'machine': b.machine},
            'file_bytes': byte_changes(a.data, b.data), 'changed_sections': changed,
            'identical_sections': identical, 'symbols': symbol_changes}


def public_comparison(result, output=None):
    diagnostics.retain(json.dumps(result), output)
    result = json.loads(json.dumps(result))
    for side in ('left', 'right'):
        result[side]['path'] = '<private input>'
    for row in result['changed_sections']:
        row['name'] = diagnostics.fingerprint(row['name'])
        for change in row.get('first_differences', []):
            change.pop('left', None); change.pop('right', None)
    result['identical_sections'] = [diagnostics.fingerprint(name) for name in result['identical_sections']]
    for change in result['file_bytes']['first_differences']:
        change.pop('left', None); change.pop('right', None)
    for side in ('left_only', 'right_only'):
        for row in result['symbols'][side]:
            row['name'] = diagnostics.fingerprint(row['name'])
            row['table'] = diagnostics.fingerprint(row['table'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--output', type=Path, help='JSON report (default: stdout); exit 1 means different')
    args = parser.parse_args()
    try:
        result = compare(args.left, args.right)
    except (OSError, ValueError, IndexError, struct.error) as exc:
        diagnostics.report_failure(exc)
        return 2
    report = json.dumps(public_comparison(result, args.output), indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end='')
    return 0 if result['identical'] else 1


if __name__ == '__main__':
    diagnostics.install_exception_hook()
    raise SystemExit(main())
