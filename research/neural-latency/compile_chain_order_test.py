# SPDX-License-Identifier: Apache-2.0
"""Compile the original chain-order workload to a private SM89 cubin."""
import argparse,ctypes as C,hashlib,json,os
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stress',action='store_true',help='Compile the original varied-latency/shared-memory workload.')
    p.add_argument('--overlap',action='store_true',help='Append the bounded producer/consumer rendezvous to the stress workload.');a=p.parse_args()
    assert not a.overlap or a.stress
    assert not a.output.exists() and not a.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])
    filename='chain_order_stress.cu' if a.stress else 'chain_order_test.cu'
    source=Path(__file__).with_name(filename).read_bytes();directory=Path(torch.__file__).parent/'lib'
    input_hashes={filename:hashlib.sha256(source).hexdigest()}
    if a.overlap:
        additional=Path(__file__).with_name('chain_overlap_test.cu').read_bytes()
        input_hashes['chain_overlap_test.cu']=hashlib.sha256(additional).hexdigest();source+=b'\n'+additional
    with os.add_dll_directory(str(directory)):
        lib=C.WinDLL(str(directory/'nvrtc64_120_0.dll'))
        def fn(name,types):
            f=getattr(lib,name);f.restype=C.c_int;f.argtypes=types;return f
        create=fn('nvrtcCreateProgram',[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p])
        compile_=fn('nvrtcCompileProgram',[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)])
        logsize=fn('nvrtcGetProgramLogSize',[C.c_void_p,C.POINTER(C.c_size_t)]);log=fn('nvrtcGetProgramLog',[C.c_void_p,C.c_void_p])
        size=fn('nvrtcGetCUBINSize',[C.c_void_p,C.POINTER(C.c_size_t)]);get=fn('nvrtcGetCUBIN',[C.c_void_p,C.c_void_p])
        destroy=fn('nvrtcDestroyProgram',[C.POINTER(C.c_void_p)])
        program=C.c_void_p();assert create(C.byref(program),source,filename.encode(),0,None,None)==0
        options=(C.c_char_p*3)(b'--gpu-architecture=sm_89',b'--std=c++17',b'--fmad=false')
        try:
            status=compile_(program,len(options),options);length=C.c_size_t();assert logsize(program,C.byref(length))==0
            message=C.create_string_buffer(length.value);assert log(program,message)==0
            if status:raise RuntimeError(message.value.decode())
            assert size(program,C.byref(length))==0 and 0<length.value<1024*1024
            buffer=C.create_string_buffer(length.value);assert get(program,buffer)==0;binary=buffer.raw
        finally:assert destroy(C.byref(program))==0
    assert binary[:4]==b'\x7fELF';a.output.write_bytes(binary)
    report={'complete':True,'source_sha256':hashlib.sha256(source).hexdigest(),'cubin_sha256':hashlib.sha256(binary).hexdigest(),'cubin_bytes':len(binary),
        'options':[x.decode() for x in options],'gpu_execution':False,'original_workload_only':True,'stress':a.stress,
        'overlap':a.overlap,'input_sha256':input_hashes}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
