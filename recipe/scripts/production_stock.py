"""Fail-closed production policy for native ABLIT=0; no model execution code."""
import re

PROFILE = 'production-stock'
DONOR_KEYS = frozenset(('ABLIT_METHOD', 'ABLIT_LAYERS', 'ABLIT_INCLUDE_MTP',
                        'JSPARK_ABLIT_ROOT', 'JSPARK_ABLIT_MANIFEST_SHA256'))
DISABLED_FIELDS = frozenset(('schema_version', 'ablit', 'rank', 'state', 'applied_layers'))


def require(condition, message):
    if not condition:
        raise ValueError('production-stock: ' + message)


def environment(values):
    require(values.get('ABLIT') == '0', 'requires ABLIT=0')
    require(not DONOR_KEYS.intersection(values), 'all five donor keys must be absent')
    require(values.get('JSPARK3_ABLIT_SWAP', '0') == '0', 'fresh stock boot cannot enable swap')


def disabled_receipt(receipt, rank):
    require(type(rank) is int and rank in range(3), 'invalid expected rank')
    require(type(receipt) is dict and set(receipt) == DISABLED_FIELDS,
            'missing native DISABLED receipt or unexpected donor fields')
    require(type(receipt['schema_version']) is int and receipt['schema_version'] == 1 and
            type(receipt['ablit']) is int and receipt['ablit'] == 0 and
            type(receipt['rank']) is int and receipt['rank'] == rank,
            'receipt schema/mode/rank mismatch')
    require(receipt['state'] == 'DISABLED' and receipt['applied_layers'] == [],
            'native receipt contains donor effects')


def stock_identity(stock, rank):
    require(type(stock) is dict and stock.get('status') == 'PASS' and
            type(stock.get('rank')) is int and stock['rank'] == rank and
            type(stock.get('ablit')) is int and stock['ablit'] == 0,
            'native stock verifier identity mismatch')
    disabled_receipt(stock.get('ablation'), rank)


def loader_facts(loaders):
    """Checked facts needed by load_gate, shared by producer and controller.

    Accept historical full loader receipts as well as projected receipts;
    filenames, tensor descriptors and optional fields remain private.
    """
    require(isinstance(loaders, dict), 'missing loader evidence')
    result = {}
    for kind in ('target', 'draft'):
        row = loaders.get(kind)
        require(isinstance(row, dict) and row.get('state') == 'COMPLETE',
                kind + ' loader incomplete')
        digest = row.get('ordered_headers_sha256')
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None,
                kind + ' loader header census missing or malformed')
        result[kind] = {'state': 'COMPLETE', 'ordered_headers_sha256': digest}
    return result


def all_ranks(values, runtime):
    environment(values)
    require(type(runtime) is list and len(runtime) == 3, 'requires all three native receipts')
    for rank, row in enumerate(runtime):
        require(type(row) is dict and type(row.get('rank')) is int and row['rank'] == rank,
                'three-rank census mismatch')
        stock_identity(row.get('stock_v13'), rank)
    return dict(status='PASS', profile=PROFILE, ablit=0, ranks=[0, 1, 2],
                native_receipts='DISABLED', donor_keys='ABSENT')
