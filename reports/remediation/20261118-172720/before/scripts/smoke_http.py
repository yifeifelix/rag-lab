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
    def request(path, payload=None, authorized=True):
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {'Content-Type': 'application/json'}
        if authorized:
            headers['Authorization'] = f'Bearer {token}'
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
    result = request('/api/v1/query', {'query': 'What is the data retention policy?', 'use_cache': False})
    assert isinstance(result.get('answer'), str) and result['answer'].strip(), result
    assert result['generation_mode'] == 'local_extractive', result
    print('HTTP smoke passed: health, owner authentication, query')


if __name__ == '__main__':
    main()
