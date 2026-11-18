"""Summarize completed annotations without turning pending rows into successes."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(path):
    groups = defaultdict(lambda: {'total':0, 'reviewed':0, 'correct':0, 'partial':0,
                                  'incorrect':0, 'citations_supported':0})
    for line in path.read_text().splitlines():
        row = json.loads(line)
        counts = groups[row['architecture']]
        counts['total'] += 1
        review = row.get('review', {})
        label = review.get('correctness')
        if label is None:
            continue
        if label not in ('correct', 'partial', 'incorrect') or not review.get('reviewer'):
            raise ValueError('Completed reviews require correct/partial/incorrect and a reviewer')
        if review.get('citation_support') not in ('supported', 'unsupported', 'not_applicable'):
            raise ValueError('Completed reviews require a citation_support label')
        counts['reviewed'] += 1
        counts[label] += 1
        counts['citations_supported'] += review['citation_support'] == 'supported'
    for counts in groups.values():
        counts['pending'] = counts['total']-counts['reviewed']
        counts['correct_rate_among_reviewed'] = counts['correct']/counts['reviewed'] if counts['reviewed'] else None
    return dict(groups)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.input)
    text = json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.write_text(text)
    print(text)
