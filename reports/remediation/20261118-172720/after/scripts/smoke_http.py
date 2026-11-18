"""Check a running image through its public HTTP API (no paid models)."""
import argparse
import json
import os
from pathlib import Path
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--env-file', type=Path)
    args = parser.parse_args()
    token = os.getenv('RAG_ADMIN_TOKEN', '')
    if args.env_file:
        values = dict(line.split('=', 1) for line in args.env_file.read_text().splitlines()
                      if line and not line.startswith('#'))
        token = values['RAG_ADMIN_TOKEN']
    def request(path, payload=None, authorized=True, access_token=None):
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {'Content-Type': 'application/json'}
        if authorized:
            headers['Authorization'] = f'Bearer {access_token if access_token is not None else token}'
        req = urllib.request.Request(args.url + path, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)
    health = request('/health')
    assert health['status'] == 'healthy' and health['indexed_chunks'] > 0, health
    try:
        request('/api/v1/documents', authorized=False)
    except urllib.error.HTTPError as exc:
        assert exc.code == 401, exc.code
    else:
        raise AssertionError('Container API must require its owner token')
    result = request('/api/v1/query', {'query': 'What is the current EU restore recovery target for Cedar?', 'use_cache': False})
    assert isinstance(result.get('answer'), str) and result['answer'].strip(), result
    assert result['generation_mode'] in ('local_extractive', 'refusal'), result
    tester = values.get('RAG_TESTER_TOKEN') if args.env_file else os.getenv('RAG_TESTER_TOKEN')
    if tester:
        assert request('/api/v1/capabilities', access_token=tester)['role'] == 'tester'
        try:
            request('/api/v1/documents', {}, access_token=tester)
        except urllib.error.HTTPError as exc:
            assert exc.code == 403, exc.code
        else:
            raise AssertionError('Tester must not be able to edit documents')
    print('HTTP smoke passed: health, owner authentication, query')


if __name__ == '__main__':
    main()
