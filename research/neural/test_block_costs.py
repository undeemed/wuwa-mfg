# SPDX-License-Identifier: Apache-2.0
"""Check schedule rejection using in-memory corruptions of a real native trace."""
import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path

from summarize_block_costs import summarize


class MemoryTrace:
    def __init__(self,rows):
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows);self.text=stream.getvalue()
    def open(self,**kwargs):return io.StringIO(self.text,newline=kwargs.get('newline'))
    def read_bytes(self):return self.text.encode()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    actual=summarize(args.trace)
    with args.trace.open(newline='') as handle:all_rows=list(csv.DictReader(handle))
    rows=[r for r in all_rows if int(r['frame'])==actual['frames'][0]]
    baseline=summarize(MemoryTrace(rows))
    assert len(baseline['frames'])==1 and len(baseline['blocks'])==71
    checks=[]
    for label,index,key,value in [('wrong kernel',10,'names','different_kernel'),
            ('wrong chain',10,'chain','999'),('multi-kernel chain',10,'kernels','2'),
            ('nonfinite interval',10,'gpu_ms','nan'),('negative interval',10,'gpu_ms','-1'),
            ('failed launch',10,'status','-1')]:
        changed=copy.deepcopy(rows);changed[index][key]=value
        try:summarize(MemoryTrace(changed))
        except ValueError:checks.append({'case':label,'rejected':True})
        else:raise AssertionError(label)
    try:summarize(MemoryTrace(rows[:-1]))
    except ValueError:checks.append({'case':'truncated frame','rejected':True})
    else:raise AssertionError('Truncated frame accepted.')
    failed=copy.deepcopy(rows)
    for row in failed:row['frame']=str(int(rows[0]['frame'])+1)
    failed[10]['status']='-1'
    mixed=summarize(MemoryTrace(rows+failed))
    assert mixed['frames']==baseline['frames'] and len(mixed['discarded_frames'])==1
    assert mixed['total_intervals']==baseline['total_intervals']
    report={'schema':1,'trace_sha256':hashlib.sha256(args.trace.read_bytes()).hexdigest(),
      'real_trace_complete_frames':len(actual['frames']),'single_complete_frame_accepted':True,
      'rejections':checks,'failed_frame_excluded_from_mixed_trace':True,
      'scope':'Parser guards and schedule consistency; not proof of inferred block correspondence or image quality.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
