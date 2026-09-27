"""Optional TRIAR runtime-N stage on either existing thirds flag value."""
import copy

KEY = 'JSPARK3_TRIAR'
ENTRY = '/recipe/scripts/triar_entry.py'


def install(ns):
    ns['V16_ENV'] += (KEY, 'JSPARK3_TRIAR_P2P')
    base_env, base_create = ns['rank_env'], ns['container_argv']
    base_contract, base_digest = ns['validate_container_contract'], ns['configuration_digest']

    def enabled(values):
        flag, protocol = values.get(KEY, '0'), values.get('JSPARK3_TRIAR_P2P', 'auto')
        if flag not in ('0', '1') or protocol != 'auto':
            raise ns['Refusal']('invalid TRIAR flag/protocol')
        if flag == '0' and protocol != 'auto':
            raise ns['Refusal']('TRIAR protocol requires TRIAR enabled')
        if flag == '1' and values.get('JSPARK3_V16_PROFILE') not in ('qa','production','production-stock'):
            raise ns['Refusal']('TRIAR requires QA, production or production-stock profile')
        return flag == '1'

    def rank_env(values, rank, preflight_sha256='0'*64, recipe_manifest_sha256=None):
        result = base_env(values, rank, preflight_sha256, recipe_manifest_sha256)
        if enabled(values):
            result += ['JSPARK3_TRIAR=1']
            protocol = values.get('JSPARK3_TRIAR_P2P', 'auto')
            if protocol == 'simple': result += ['NCCL_P2P_LL_THRESHOLD=0']
            elif protocol == 'll': result += ['NCCL_P2P_LL_THRESHOLD=1048576', 'NCCL_ALLOC_P2P_NET_LL_BUFFERS=1']
        return result

    def create(values, *args, **kwargs):
        command = base_create(values, *args, **kwargs)
        if enabled(values):
            at = command.index('--entrypoint') + 1
            image = command.index(ns['IMAGE'], at)
            if values.get('JSPARK3_V18_THIRDS', '0') == '1':
                if command[image+1:image+3] != ['-S', '/recipe/scripts/thirds_entry.py']:
                    raise ns['Refusal']('thirds entry seam drift')
                command[image+2] = ENTRY
            else:
                if command[at] != '/recipe/scripts/container_entry.sh':
                    raise ns['Refusal']('base entry seam drift')
                command[at] = '/usr/bin/python3'
                command[image+1:image+1] = ['-S', ENTRY]
        return command

    def contract(values, rank, identity, item, preflight_sha, recipe_sha):
        if enabled(values):
            expected = ['-S', ENTRY, *ns['server_argv'](values, rank)]
            if item['Config']['Entrypoint'] != ['/usr/bin/python3'] or item['Config']['Cmd'] != expected:
                raise ns['Refusal']('TRIAR entry/argv drift')
            item = copy.deepcopy(item)
            if values.get('JSPARK3_V18_THIRDS', '0') == '1':
                item['Config']['Cmd'][1] = '/recipe/scripts/thirds_entry.py'
            else:
                item['Config'].update(Entrypoint=['/recipe/scripts/container_entry.sh'], Cmd=expected[2:])
        return base_contract(values, rank, identity, item, preflight_sha, recipe_sha)

    def digest(values):
        original = base_digest(values)
        if not enabled(values): return original
        return ns['sha_bytes'](ns['canonical'](dict(v18_configuration_sha256=original,
            triar=dict(mode='runtime-N', p2p=values.get('JSPARK3_TRIAR_P2P', 'auto')))))

    ns.update(rank_env=rank_env, container_argv=create, validate_container_contract=contract,
              configuration_digest=digest)
