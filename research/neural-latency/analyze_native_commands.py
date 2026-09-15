# SPDX-License-Identifier: Apache-2.0
"""Audit observed command methods and owned packed-argument forwarding."""
import argparse,collections,json
from pathlib import Path
from collect_demo_photo_training import read,sha,dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('state','paired','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists() and not a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    paths={n:a.state/n for n in ('nr-command-probe.jsonl','nr-buffer-probe.jsonl','nr-launch-contract.jsonl')}
    command=[json.loads(s) for s in paths['nr-command-probe.jsonl'].read_text().splitlines()]
    barriers=[json.loads(s) for s in paths['nr-buffer-probe.jsonl'].read_text().splitlines()]
    contracts=[json.loads(s) for s in paths['nr-launch-contract.jsonl'].read_text().splitlines()]
    assert [r['event'] for r in command]==list(range(1,len(command)+1))
    cap=command[0]['event_cap'];assert command[0]['kind']=='configuration'
    hooks=[r for r in command if r['kind']=='hook'];expected=set(range(9,86))-{26,80}
    slots={r['slot'] for r in hooks};assert len(slots)==len(hooks)
    methods={r['slot']:r['method'] for r in hooks}
    all_hooks=slots==expected and all(r['status']==0 for r in hooks)
    rows=[]
    for frame in (1,2,3,4,64,128,256):
        events=[r for r in command if r['frame']==frame]
        entered=[r for r in events if r['kind']=='evaluation_enter'];exited=[r for r in events if r['kind']=='evaluation_exit']
        assert len(entered)==len(exited)==1 and events[0]['kind']=='evaluation_enter' and events[-1]['kind']=='evaluation_exit'
        launches=[r for r in events if r['kind']=='launch_enter'];ends=[r for r in events if r['kind']=='launch_exit']
        copies=[r for r in events if r['kind']=='owned_arguments']
        assert len(launches)==len(ends)==158 and [r['after_chain'] for r in launches]==list(range(1,159))
        assert all(r['count']==1 and not r['extended'] for r in launches)
        depth=0
        for r in events:
            if r['kind']=='launch_enter':assert depth==0;depth=1
            elif r['kind']=='launch_exit':assert depth==1;depth=0
            elif r['kind']=='owned_arguments':assert depth==1 and r['copy_exact']
        assert depth==0
        native=[r for r in contracts if r.get('frame')==frame]
        assert len(native)==158 and [r['chain'] for r in native]==list(range(1,159))
        assert all(r['status']==0 and r['kernel']==0 and r['api']=='chain' for r in native)
        calls=[r for r in events if r['kind']=='command']
        between=[r for r in calls if not r['inside_launch'] and 1<=r['after_chain']<158]
        before=[r for r in calls if not r['inside_launch'] and r['after_chain']==0]
        after=[r for r in calls if not r['inside_launch'] and r['after_chain']==158]
        inside=[r for r in calls if r['inside_launch']]
        legacy=[r for r in barriers if r.get('frame')==frame and r['kind']=='legacy_barrier_call']
        enhanced=[r for r in barriers if r.get('frame')==frame and r['kind']=='enhanced_barrier_group']
        def counts(data):return dict(collections.Counter(methods[r['slot']] for r in data))
        rows.append({'frame':frame,'native_single_kernel_calls':len(native),'owned_argument_calls':len(copies),
            'all_packed_arguments_copied':len(copies)==158 and [r['after_chain'] for r in copies]==list(range(1,159)),
            'owned_argument_bytes':sum(r['bytes'] for r in copies),'reused_source_storage':sum(r['reused_source_storage'] for r in copies),
            'reused_storage_changed':sum(r['reused_storage_changed'] for r in copies),
            'copy_records':[{'chain':r['after_chain'],'bytes':r['bytes'],'reused_source_storage':r['reused_source_storage'],
                'reused_storage_changed':r['reused_storage_changed']} for r in copies],
            'commands':{'inside_native_launch':counts(inside),'between_launches':counts(between),'before_first_launch':counts(before),'after_last_launch':counts(after),
                'off_evaluation_thread':sum(not r['on_evaluation_thread'] for r in calls),'different_command_list':sum(not r['same_command_list'] for r in calls),
                'nested_method_calls':sum(r['nested_method'] for r in calls)},
            'outside_command_sequence':[{'chain':r['after_chain'],'method':methods[r['slot']],'nested':r['nested_method'],
                'on_evaluation_thread':r['on_evaluation_thread'],'same_command_list':r['same_command_list']} for r in calls if not r['inside_launch']],
            'legacy_barriers':len(legacy),'enhanced_barrier_groups':len(enhanced),
            'barriers_inside_launch':sum(r['inside_launch'] for r in legacy+enhanced),
            'coverage_gap':entered[0]['coverage_gap'] or exited[0]['coverage_gap'],
            'retained_argument_bytes_at_exit':exited[0]['retained_argument_bytes']})
    pair=read(a.paired);assert pair['complete'] and pair['normal_state_restored']
    matched=[]
    for r in pair['paired_frames']:
        if all(r['resources'][n]['byte_equal'] for n in ('color','depth','motion')):matched.append(r['frame'])
    history_matched=[];history_ok=False
    for r in pair['paired_frames']:
        if r['reset']:history_ok=True
        history_ok=history_ok and r['frame'] in matched
        if history_ok:history_matched.append(r['frame'])
    report={'complete':True,'target_achieved':False,'quality_gate_passed':False,'native_runtime_accelerated':False,
        'new_sample_launches':2,'new_game_launches':0,'input_sha256':{n:sha(f) for n,f in paths.items()},'paired_sha256':sha(a.paired),
        'events_retained':len(command),'cap_reached':len(command)>=cap,'expected_method_hooks':75,'all_method_hooks_attached':all_hooks,
        'hook_results':[{k:r[k] for k in ('slot','method','status')} for r in hooks],
        'rows':rows,'matched_input_frames':matched,'matched_input_outputs_exact':all(pair['paired_frames'][i]['resources']['output']['byte_equal'] for i in matched),
        'history_matched_frames':history_matched,
        'history_matched_outputs_exact':all(pair['paired_frames'][i]['resources']['output']['byte_equal'] for i in history_matched) if history_matched else None,
        'paired_capture':pair,'native_batching_safety_established':False,
        'limitations':['Hooks cover the observed ID3D12GraphicsCommandList10 implementations, excluding IUnknown/object methods; existing hooks cover both barrier methods.',
            'Device/resource methods, cached alternative implementations, future interfaces and implicit driver operations are not a complete captured API stream.',
            'Packed CPU argument bytes are copied and retained through process exit; this does not extend embedded GPU-resource lifetimes.',
            'Original argument storage is inspected only while each caller supplies it. Reuse is determined by identity and comparison with an owned earlier copy.',
            'A frame with identical current inputs but different preceding inputs is not an output-equivalence test of identical temporal history.',
            'No kernel calls are merged and no native speedup is claimed. Unmatched-input frames cannot establish output equivalence.']}
    dump(a.output,report)
    print(json.dumps({'events':len(command),'all_method_hooks_attached':all_hooks,'matched_input_frames':matched,
        'matched_input_outputs_exact':report['matched_input_outputs_exact'],'rows':[{k:v for k,v in r.items() if k not in ('copy_records','outside_command_sequence')} for r in rows]},indent=2))


if __name__=='__main__':main()
