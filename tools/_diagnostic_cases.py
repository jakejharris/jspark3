# SPDX-License-Identifier: Apache-2.0
"""The independent review's 76 diagnostic inputs, unchanged synthetic values."""
import json

TRACE=('Traceback (most recent call last):\n'
       '  File "remote_check.py", line 23, in check\n'
       '    assert actual == expected, "native identity drift"\n'
       'AssertionError: native identity drift\n')
SECRET='opaqueR3A7b9CredentialValue';SECOND='opaqueR3D4e6ContinuationValue'
cases=[]
def add(name,diagnostic,secrets=(SECRET,),markers=(),known=None,append_trace=True):
    cases.append(dict(name=name,diagnostic=diagnostic,secrets=list(secrets),markers=list(markers),known=known or {},append_trace=append_trace))
for scheme in ['Bearer','Basic','token','Digest','Negotiate','AWS4-HMAC-SHA256','Custom']:
    add('scheme-'+scheme,'Authorization: '+scheme+' '+SECRET)
for name in ['Proxy-Authorization','X-Api-Key','X-Auth-Token','X-Credential','Cookie','Set-Cookie']:
    add('header-'+name,name+': value='+SECRET+'; extra='+SECOND,(SECRET,SECOND))
add('folded-header','Authorization: Custom '+SECRET+'\r\n\t'+SECOND,(SECRET,SECOND))
add('prefixed-folded-header','DEBUG Authorization: Custom '+SECRET+'\nDEBUG\t'+SECOND,(SECRET,SECOND))
add('tuple-header',repr(('Authorization','Custom '+SECRET)))
add('json-header',json.dumps({'Authorization':'Custom '+SECRET}))
add('json-cookie',json.dumps({'Cookie':'session='+SECRET}))
add('json-escaped-header-key','{"Authoriz\\u0061tion":"Custom '+SECRET+'"}')
add('json-escaped-cookie-key','{"Coo\\u006bie":"session='+SECRET+'"}')
for scheme in ['https','ssh','ftp','git+https']:
    for userinfo in [SECRET,'user:'+SECRET,':'+SECRET]:
        add('url-'+scheme+'-'+str(len(cases)),scheme+'://'+userinfo+'@host.invalid/repo')
add('scheme-relative-url','//'+SECRET+'@host.invalid/repo')
add('json-slashes-url','https:\\/\\/'+SECRET+'@host.invalid/repo')
add('percent-userinfo-url','https://' + 'user:' + SECRET + '%2Ftail@host.invalid/repo')
for key in ['api_key','access_token','key','auth','signature','sig','session_id']:
    add('query-'+key,'https://host.invalid/?'+key+'='+SECRET+'&page=2')
for key in ['api%5Fkey','%74oken','X-Amz-Signature','X-Goog-Signature']:
    add('query-'+key,'https://host.invalid/?'+key+'='+SECRET+'&page=2')
add('fragment-key','https://host.invalid/#key='+SECRET)
for key in ['PASSWORD','API_KEY','--api-key']:
    for separator in [',',';','&']:
        add('scalar-'+key+'-'+str(ord(separator)),key+'=front'+separator+SECRET)
add('quoted-scalar','PASSWORD="front,'+SECRET+';tail"')
for prefix in ['ghp_','github_pat_','hf_','sk-','xoxb-','AIza']:
    value=prefix+'X'*32;add('token-'+prefix,'X-Diagnostic: '+value,(value,))
key='-----BEGIN ' + 'PRIVATE KEY-----\n'+SECRET+'\n-----END PRIVATE KEY-----'
add('private-key',key)
add('truncated-private-key',key.split('-----END')[0],markers=(),append_trace=False)
add('known-plain','diagnostic value='+SECRET,known={'TEST_SECRET':SECRET})
known='opaque+slash/value=42'
add('known-percent-encoded','diagnostic value=opaque%2Bslash%2Fvalue%3D42',('opaque%2Bslash%2Fvalue%3D42',),known={'TEST_SECRET':known})
# Long opaque values must never appear in shared output, including tail fragments.
add('truncation-secret-value','TOKEN='+'s'*40000,('s'*100,))
add('truncation-secret-header','Authorization: Custom '+'s'*40000,('s'*100,))
add('truncation-clean-tail','older-data\n'+'x'*40000+'\n',(),markers=('AssertionError: native identity drift',))
add('json-error-with-auth',json.dumps({'Authorization':'Custom '+SECRET,'error':'rank2 model checksum drift','file':'remote_check.py','expected':'abc','actual':'def'}),markers=('rank2 model checksum drift','expected','actual'))
add('json-error-with-api-key',json.dumps({'api_key':SECRET,'error':'rank2 model checksum drift','file':'remote_check.py'}),markers=('rank2 model checksum drift',))
add('json-error-with-cookie',json.dumps({'Cookie':'session='+SECRET,'error':'rank2 model checksum drift'}),markers=('rank2 model checksum drift',))
add('indented-trace','remote failure:\n  Authorization: Custom '+SECRET+'\n'+''.join('  '+s+'\n' for s in TRACE.splitlines()),markers=('remote_check.py','AssertionError: native identity drift'),append_trace=False)
add('tokenizer-diagnostic','TokenizerError: missing vocab.json at /models/model-1',(),markers=('missing vocab.json at /models/model-1',))
add('native-diagnostic',TRACE,(),markers=('remote_check.py','AssertionError: native identity drift'),append_trace=False)
