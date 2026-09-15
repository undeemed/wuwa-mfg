# SPDX-License-Identifier: Apache-2.0
"""Alternate warmed CUDA Graph modes to reduce timing-order sensitivity."""
import argparse,json,statistics
from pathlib import Path

import numpy as np
import torch

from compare_output_grade import read_capture
from fused_norm import FusedNorm
from output_grade import GradedStudent
from student_probe import HierarchicalStudent
from student_training_pairs import read,sha
from test_student_output import configure,same


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--model',nargs=2,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--conditioning-fusion',action='store_true',help='Compare all existing fusions with all plus decoder conditioning.')
    args=p.parse_args()
    modes=['all','all-conditioning'] if args.conditioning_fusion else ['combined','all']
    if args.output.exists() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Use a fresh private report.')
    assert 1<=len(args.model)<=4 and len({n for n,_ in args.model})==len(args.model)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    controls,arrays,hashes=read_capture(args.capture)
    assert arrays['color'].shape==(1080,1920,3)
    source=torch.from_numpy(arrays['color']).permute(2,0,1).unsqueeze(0).cuda().half().contiguous(memory_format=torch.channels_last)
    kernel=FusedNorm();rows={}
    with torch.inference_mode():
        for name,folder in args.model:
            folder=Path(folder);record=read(folder/'result.json');a=record['architecture']
            assert record['controls']==controls and a['variant'] in ('hierarchical','hierarchical-film') and a['width'] in (16,32)
            assert a['blocks']==2 and a['noise_channels']==0
            state=torch.load(folder/'student-private.pt',map_location='cpu',weights_only=True)
            assert state['architecture']==a
            model=GradedStudent(HierarchicalStudent(a['width'],2,conditioned=a['variant']=='hierarchical-film'),a['explicit_output_grading'])
            model.load_state_dict(state['state_dict'],strict=True)
            model=model.cuda().half().eval().to(memory_format=torch.channels_last);model.fused_backend=kernel
            configure(model,'baseline',kernel);reference=model(source)
            graphs,outputs={},{}
            for mode in modes:
                configure(model,mode,kernel)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(5):model(source)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):out=model(source)
                graphs[mode]=graph;outputs[mode]=out
            for _ in range(40):
                for graph in graphs.values():graph.replay()
            torch.cuda.synchronize()
            assert all(same(out,reference) for out in outputs.values())
            measurements=[]
            for index in range(30):
                order=modes[:] if index%2==0 else modes[::-1]
                times={}
                for mode in order:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):graphs[mode].replay()
                    end.record();end.synchronize()
                    times[mode]=start.elapsed_time(end)/10
                measurements.append({'order':order,'ms_per_replay':times})
            assert all(same(out,reference) for out in outputs.values())
            summary={mode:{'median_ms':statistics.median(r['ms_per_replay'][mode] for r in measurements),
                'p95_ms':float(np.percentile([r['ms_per_replay'][mode] for r in measurements],95))} for mode in graphs}
            rows[name]={'checkpoint_sha256':sha(folder/'student-private.pt'),'width':a['width'],
                'all_graphs_match_baseline_bitwise':True,'measurements':measurements,'summary':summary,
                'median_reduction_percent':100*(1-summary[modes[1]]['median_ms']/summary[modes[0]]['median_ms'])}
            del graphs,outputs,graph,out,reference,model,state
    result={'complete':True,'target_achieved':False,'quality_gate_passed':False,'native_runtime_accelerated':False,
        'capture_hashes':hashes,'warmup_replays_per_mode':40,'replays_per_interval':10,
        'alternating_pairs':30,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'models':rows,
        'compared_modes':modes,
        'limitations':['Intervals average ten consecutive graph replays; p95 is an interval-average percentile, not per-frame tail latency.',
            'Measured execution of experimental students plus grade, excluding D3D12 integration. No native-runtime acceleration.',
            'One static input and one device; other desktop GPU work is not controlled.']}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({n:{'summary':r['summary'],'reduction_percent':r['median_reduction_percent']} for n,r in rows.items()},indent=2))


if __name__=='__main__':
    main()
