"""Frozen grouped challenge set: explicit fictional policies + licensed CPython docs.

All questions are agent-authored, including test. This is a development fixture,
not a blind independent benchmark. Human users must supply independent test cases.
"""
from pathlib import Path
import hashlib
import json

from rag_platform.config import RAGConfig


def prepare():
    root = RAGConfig().project_root
    target = root / 'data/learning_v2'
    target.mkdir(parents=True, exist_ok=True)
    documents, rows = [], {'dev': [], 'test': []}
    services = ['Cedar', 'Birch', 'Maple', 'Oak', 'Pine', 'Elm',
                'Willow', 'Aspen', 'Hazel', 'Alder', 'Spruce', 'Beech']

    def add(split, group, question, relevant, spans, category, keywords=None):
        rows[split].append({'question_id': f'V2-{split.upper()}-{len(rows[split]) + 1:03}',
            'group_id': group, 'split': split, 'question': question, 'relevant_doc_ids': relevant,
            'supporting_spans': spans, 'category': category, 'expected_keywords': keywords or [],
            'paraphrase_variants': [], 'label_status': 'agent_authored_pending_human_review'})

    for index, service in enumerate(services):
        split = 'dev' if index < 6 else 'test'
        group = f'policy-{service.lower()}'
        ids, facts = {}, {}
        for region_index, region in enumerate(['EU', 'US']):
            recovery = 17 + index * 3 + region_index * 11
            retention = 37 + index * 5 + region_index * 13
            for kind in ['restore', 'export']:
                doc_id = f'{service.lower()}.{region.lower()}.{kind}.v2'
                fact = (f'The {service} {region} restore recovery target is {recovery} minutes.' if kind == 'restore'
                        else f'The {service} {region} export retention period is {retention} hours.')
                text = (f'# {service} {region} {kind} policy\nStatus: CURRENT. Effective version: 2026.\n'
                        f'{fact}\nThis policy applies only to the {region} region and the {kind} operation. '
                        'It specifies no contract renewal fee, refund guarantee, or staffing requirement. '
                        'Policies for other regions or operations do not override this one.')
                documents.append({'doc_id': doc_id, 'title': f'{service} {region} {kind} policy 2026',
                    'domain': 'fictional_operations', 'version': '2', 'text': text,
                    'source': f'fixture://learning-v2/{doc_id}', 'license': 'MIT',
                    'provenance': {'kind': 'fictional_policy_fixture'}})
                ids[region, kind], facts[region, kind] = doc_id, fact
                if kind == 'restore':
                    questions = [f'Under the current policy, how many minutes should {service} {region} restoration take?',
                                 f'What is the 2026 recovery target for a {service} restore in {region}?']
                    keywords = [str(recovery), 'minutes']
                else:
                    questions = [f'For {service} in {region}, how long are exported files retained under the current policy?',
                                 f'In 2026 what is the {region} export retention window for {service}, in hours?']
                    keywords = [str(retention), 'hours']
                for question in questions:
                    add(split, group, question, [doc_id], {doc_id: [fact]}, 'conceptual_semantic', keywords)
            # Same vocabulary, conflicting values, clearly labelled historical version.
            archived_id = f'{service.lower()}.{region.lower()}.restore.v1'
            documents.append({'doc_id': archived_id, 'title': f'{service} {region} restore policy ARCHIVED 2024',
                'domain': 'fictional_operations', 'version': '1', 'source': f'fixture://learning-v2/{archived_id}',
                'license': 'MIT', 'text': f'# {service} {region} restore policy\nStatus: ARCHIVED. Effective version: 2024.\n'
                f'The historical {service} {region} restore recovery target was {recovery + 90} minutes. '
                'This document is not the current policy. Use the 2026 policy for current recovery targets.'})
        for question, keys in [
            (f'Under the 2026 {service} EU policies, give both the restore recovery target and export retention period.', [('EU', 'restore'), ('EU', 'export')]),
            (f'Compare current {service} restore recovery targets between the EU and US regions.', [('EU', 'restore'), ('US', 'restore')])]:
            relevant = [ids[key] for key in keys]
            add(split, group, question, relevant, {ids[key]: [facts[key]] for key in keys}, 'multi_hop')
        add(split, group, f'{service} EU restore target in 2026?', [ids['EU', 'restore']],
            {ids['EU', 'restore']: [facts['EU', 'restore']]}, 'short_query')
        add(split, group, f'What is the current {service} US renewal fee in dollars?', [], {}, 'unanswerable_guardrail')
        add(split, group, f'Which {service} EU policy guarantees a refund when restoration takes too long?', [], {}, 'unanswerable_guardrail')
    source = root / 'data/python_support/documents.jsonl'
    python_docs = [json.loads(line) for line in source.read_text().splitlines()]
    documents.extend(python_docs)
    examples = [
        ('dev', 'PY-GATHER', 'When aggregating successful coroutines, are results listed by arrival time or by the original input order?', 'in the order of the original sequence'),
        ('dev', 'PY-WAIT-FOR', 'What happens to a coroutine that suppresses cancellation inside a timeout wrapper and returns a value?', 'that value is returned'),
        ('dev', 'PY-WAIT', 'Does waiting with a timeout raise an exception for unfinished futures, or return them separately?', 'does not raise TimeoutError'),
        ('dev', 'PY-SEMAPHORE', 'Semaphore', 'decremented by each acquire()'),
        ('test', 'PY-SHIELD', 'Can protecting an inner task from cancellation prevent its caller from receiving CancelledError?', 'its caller is still cancelled'),
        ('test', 'PY-TO-THREAD', 'Do context variables follow a blocking function dispatched from the event loop into a worker thread?', 'propagated'),
        ('test', 'PY-TASKGROUP', 'When one grouped task raises a non-cancellation exception, what happens to its siblings?', 'cancel all remaining tasks'),
        ('test', 'PY-CREATE-TASK', 'What kind of object is returned after scheduling execution of a coroutine in a new task?', 'Return a Task object.'),
        ('test', 'PY-QUEUE', 'Queue', 'qsize()'),
    ]
    for split, doc_id, question, quotation in examples:
        document = next(doc for doc in python_docs if doc['doc_id'] == doc_id)
        assert ' '.join(quotation.split()) in ' '.join(document['text'].split()), (doc_id, quotation)
        add(split, f'python-{doc_id}', question, [doc_id], {doc_id: [quotation]}, 'short_query' if len(question.split()) == 1 else 'conceptual_semantic')
    artifacts = {'documents.jsonl': documents, **{f'qa_{split}.jsonl': data for split, data in rows.items()}}
    hashes = {}
    for filename, values in artifacts.items():
        content = ''.join(json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n' for value in values)
        path = target / filename
        if path.exists() and path.read_text() != content:
            raise RuntimeError('Frozen dataset differs; use a new version: ' + filename)
        path.write_text(content)
        hashes[filename] = hashlib.sha256(content.encode()).hexdigest()
    groups = {split: sorted({row['group_id'] for row in values}) for split, values in rows.items()}
    assert not set(groups['dev']) & set(groups['test'])
    manifest = {'schema': 2, 'documents': len(documents), 'questions': {k: len(v) for k, v in rows.items()},
        'sha256': hashes, 'groups': groups, 'source_python_dataset_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'question_hashes': {split: sorted(hashlib.sha256(' '.join(row['question'].lower().split()).encode()).hexdigest()
                                         for row in values) for split, values in rows.items()},
        'label_status': 'agent_authored_pending_independent_human_review', 'blind_test': False,
        'limitations': ['Policy documents are fictional', 'Test author has seen the implementation',
                        'Dev/test groups are separate; semantic independence requires human review',
                        'Do not tune on test results; collect a new human-authored test set for external claims']}
    (target / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['questions']), 'documents:', len(documents))


if __name__ == '__main__':
    from rag_platform.runtime import configure_runtime
    configure_runtime()
    prepare()
