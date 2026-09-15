# SPDX-License-Identifier: Apache-2.0
"""Summarize intact CUDA-chain GPU timestamp intervals from the hidden demo."""
import argparse, collections, csv, json, statistics
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('trace', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--after-frame', type=int, default=600)
p.add_argument('--width', type=int)
p.add_argument('--height', type=int)
a = p.parse_args()
rows = list(csv.DictReader(a.trace.open(newline='')))
frames = collections.defaultdict(list)
for row in rows:
    if int(row['frame']) > a.after_frame:
        frames[int(row['frame'])].append(row)
if not frames:
    raise SystemExit('No samples after the selected warmup frame.')
expected = max(len(frame) for frame in frames.values())
complete = {i: frame for i, frame in frames.items() if len(frame) == expected and
    [int(row['chain']) for row in frame] == list(range(1,expected+1)) and
    all(int(row['status']) == 0 for row in frame)}
if not complete:
    raise SystemExit('No complete successful frames.')
fingerprints = {tuple((row['names'],int(row['kernels'])) for row in frame) for frame in complete.values()}
if len(fingerprints) != 1:
    raise SystemExit('Different graphs were captured; summarize each graph separately.')
samples = collections.defaultdict(list)
counts = collections.Counter(row['names'] for row in next(iter(complete.values())))
for frame in complete.values():
    groups = collections.defaultdict(float)
    for row in frame:
        groups[row['names']] += float(row['gpu_ms'])
    for name, ms in groups.items():
        samples[name].append(ms)
totals = [sum(float(row['gpu_ms']) for row in frame) for frame in complete.values()]
data = {'schema':1,'frames':sorted(complete), 'complete_samples':len(complete),
        'discarded_samples':len(frames)-len(complete), 'chains_per_evaluation':expected,
        'kernels_per_evaluation':sum(int(row['kernels']) for row in next(iter(complete.values()))),
        'sum_of_chain_intervals_ms': {'median':statistics.median(totals),'min':min(totals),'max':max(totals)},
        'instrumentation_overhead_not_subtracted':True,
        'model_output_resolution':[a.width,a.height] if a.width and a.height else None,
        'resolution_source':'Must be confirmed separately in this trial OptiScaler log',
        'architecture_selection':'Unknown: runtime submitted fatbins, not bare ELF modules',
        'by_kernel':sorted([{'name':name,'launches_per_evaluation':counts[name],
            'median_total_ms':statistics.median(ms), 'min_total_ms':min(ms),'max_total_ms':max(ms)}
            for name,ms in samples.items()],key=lambda row:row['median_total_ms'],reverse=True)}
a.output.write_text(json.dumps(data,indent=2))
print(json.dumps({**{k:v for k,v in data.items() if k!='by_kernel'},'top_12':data['by_kernel'][:12]},indent=2))
