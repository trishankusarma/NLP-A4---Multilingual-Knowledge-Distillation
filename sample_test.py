import json
import random

random.seed(123)

with open('outputs/train.jsonl') as f:
    records = [json.loads(line) for line in f if line.strip()]

# Keep only correct samples
correct = [r for r in records if r.get('correct', False)]

# Sample per language for balanced test set
from collections import defaultdict
lang_buckets = defaultdict(list)
for r in correct:
    lang_buckets[r['language']].append(r)

print('Correct samples per language:')
for lang, rows in sorted(lang_buckets.items()):
    print(f'  {lang}: {len(rows)}')

# Sample 100 per language (or all if fewer)
test_records = []
for lang, rows in lang_buckets.items():
    sample = random.sample(rows, min(20, len(rows)))
    test_records.append(sample)
    for s in sample:
        test_records

# flatten
test_records = []
for lang, rows in lang_buckets.items():
    sample = random.sample(rows, min(20, len(rows)))
    test_records.extend(sample)

random.shuffle(test_records)
print(f'Total test records: {len(test_records)}')

with open('outputs/test.jsonl', 'w') as f:
    for r in test_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print('Saved to outputs/test.jsonl')