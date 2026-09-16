# SPDX-License-Identifier: Apache-2.0
"""Read-only inspection of polling loops in the pinned first native module.

Generated code/disassembly remains private. Numeric opcode counts establish
polling structure, not a complete dependency graph or a proven hang cause.
"""
import argparse,collections,hashlib,json,re,subprocess
from pathlib import Path
from inspect_pre_kernel import CUBIN_SHA,symbols


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cubin','nvdisasm','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    assert not a.output.exists() and not a.output.resolve().is_relative_to(repo)
    data=a.cubin.read_bytes();assert hashlib.sha256(data).hexdigest()==CUBIN_SHA
    a.output.mkdir(parents=True)
    report={'complete':False,'native_files_modified':False,'target_achieved':False,'quality_gate_passed':False,
        'cubin_sha256':CUBIN_SHA,'nvdisasm_sha256':hashlib.sha256(a.nvdisasm.read_bytes()).hexdigest(),'kernels':[]}
    names=['cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8']+[
        'cc_tinlayout_fused_swin_1h_32_1_'+suffix+'_fp8' for suffix in ('inpview_tilesync','chained','ds_wait')]
    for index,name in enumerate(names):
        symbol=symbols(data,name);destination=a.output/f'kernel-{index}.sass'
        with destination.open('wb') as file:
            result=subprocess.run([str(a.nvdisasm.resolve()),'-c','-fun',str(symbol['symbol_index']),'-hex',str(a.cubin.resolve())],
                stdout=file,stderr=subprocess.PIPE,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=60)
        source=destination.read_text();instructions=[];labels={}
        for line in source.splitlines():
            label=re.match(r'\s*(\.L_[A-Za-z0-9_]+):',line)
            if label:labels[label[1]]=len(instructions)
            instruction=re.search(r'/\*([0-9a-fA-F]+)\*/\s+([^;\n]+);',line)
            if instruction:instructions.append(instruction[2].strip())
        assert instructions
        counts=collections.Counter(re.sub(r'^@!?U?P(?:[0-9]+|T)\s+','',s).split()[0].split('.')[0] for s in instructions)
        loops=[]
        for i,text in enumerate(instructions):
            target=re.search(r'\bBRA\s+`?\((\.L_[A-Za-z0-9_]+)\)',text)
            if not target or target[1] not in labels:continue
            start=labels[target[1]]
            if start>=i or i-start>64:continue
            body=instructions[start:i+1]
            if any('NANOSLEEP' in s for s in body) and any('LDG.' in s for s in body):
                loops.append({'instruction_span':len(body),'global_loads':sum('LDG.' in s for s in body),
                    'strong_gpu_loads':sum('LDG.' in s and '.STRONG.GPU' in s for s in body),'sleep_instructions':sum('NANOSLEEP' in s for s in body)})
        report['kernels'].append({'name':name,'symbol_index':symbol['symbol_index'],'kernel_bytes':symbol['size'],
            'instruction_count':len(instructions),'instruction_families':dict(sorted(counts.items())),
            'bounded_span_polling_back_edges':loops,'strong_gpu_stores':sum('STG.' in s and '.STRONG.GPU' in s for s in instructions),
            'disassembler_diagnostics':result.stderr.decode(errors='replace')})
    report['complete']=True
    report['scope']='Static polling-loop evidence only. No source operands, instruction encodings, addresses, images or weights are published; no producer/consumer dependency-cycle proof.'
    (a.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{'name':r['name'],'polling_loops':len(r['bounded_span_polling_back_edges']),'strong_gpu_stores':r['strong_gpu_stores']} for r in report['kernels']],indent=2))


if __name__=='__main__':main()
