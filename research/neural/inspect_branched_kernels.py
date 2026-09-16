# SPDX-License-Identifier: Apache-2.0
"""Inspect known multi-head kernels from the pinned private SM89 module.

The disassembly stays outside the repository. This read-only utility does not
launch an application, modify a runtime or redistribute native code.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from inspect_pre_kernel import DLL_SHA,read_span,symbols

MODULES=[(1,2,4858400,2199280,'5cc669ec5f78a58def2166195daf4516ecb39fdc01462fe84ec1c77d6f2dda46'),
         (2,4,7332096,2070528,'63fb6c2cbe3285a7a96d2782d0be456065011372c172dbf4ff0d23da78526c53'),
         (3,8,9751392,2581712,'cd1317d6307b06bbf51760ecfbcf5473ccee2274423aa1f8cd1a9d80c0f252ac')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll',type=Path,required=True)
    parser.add_argument('--cuobjdump',type=Path,required=True)
    parser.add_argument('--nvdisasm',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args()
    if args.output_directory.exists() or args.output_directory.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Use a fresh private output directory for vendor code.')
    data=args.dll.read_bytes()
    if hashlib.sha256(data).hexdigest()!=DLL_SHA:raise ValueError('Unknown native runtime.')
    args.output_directory.mkdir(parents=True)
    report={'schema':1,'dll_sha256':DLL_SHA,'runtime_modified':False,
        'nvdisasm_sha256':hashlib.sha256(args.nvdisasm.read_bytes()).hexdigest(),
        'cuobjdump_sha256':hashlib.sha256(args.cuobjdump.read_bytes()).hexdigest(),
        'private_code_not_for_redistribution':True,'kernels':[]}
    for module,heads,offset,length,digest in MODULES:
        fatbin=read_span(data,offset,length)
        if hashlib.sha256(fatbin).hexdigest()!=digest:raise ValueError('Unexpected module bytes.')
        filename=f'module-{module:02d}.fatbin'
        (args.output_directory/filename).write_bytes(fatbin)
        cubin=args.output_directory/f'module-{module:02d}.3.sm_89.cubin'
        extract=subprocess.run([str(args.cuobjdump.resolve()),'--extract-elf',cubin.name,filename],
            cwd=args.output_directory,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=60)
        if extract.returncode:raise RuntimeError(extract.stderr.decode(errors='replace'))
        image=cubin.read_bytes()
        name=f'cc_tinlayout_fused_swin_{heads}h_{heads*32}_{heads}_chained_fp8'
        symbol=symbols(image,name)
        destination=args.output_directory/f'window-{heads}h.sass'
        with destination.open('wb') as output:
            result=subprocess.run([str(args.nvdisasm.resolve()),'-c','-fun',str(symbol['symbol_index']),
                '-hex',str(cubin.resolve())],stdout=output,stderr=subprocess.PIPE,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=60)
        if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
        source=destination.read_text()
        instructions=re.findall(r'/\*([0-9a-fA-F]+)\*/\s+([^;\n]+);',source)
        if not instructions:raise ValueError('No instructions decoded.')
        counts=Counter(re.sub(r'^@!?U?P(?:[0-9]+|T)\s+','',text.strip()).split()[0].split('.')[0] for _,text in instructions)
        record={'heads':heads,'symbol':symbol,'module':module,'fatbin_sha256':digest,
            'cubin_sha256':hashlib.sha256(image).hexdigest(),'instruction_families':dict(sorted(counts.items())),
            'diagnostics':result.stderr.decode(errors='replace')}
        report['kernels'].append(record)
        (args.output_directory/f'window-{heads}h-instructions.txt').write_text(
            '\n'.join(offset+' '+text.strip() for offset,text in instructions)+'\n')
        print(name,'symbol',symbol['symbol_index'],'bytes',symbol['size'],'instructions',len(instructions),flush=True)
    (args.output_directory/'inspection.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
