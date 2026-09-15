# SPDX-License-Identifier: Apache-2.0
"""Load a locally compiled kernel into the current PyTorch CUDA process only."""
import ctypes as C
import os
from pathlib import Path
import torch

class FusedNorm:
    def __init__(self):
        if os.name != 'nt' or torch.cuda.get_device_capability() != (8,9):
            raise RuntimeError('This experiment is validated only on Windows SM89.')
        torch.cuda.init()
        # torch.cuda.init alone can leave the driver context unbound on Windows.
        torch.empty(1, device='cuda')
        directory = Path(torch.__file__).parent / 'lib'
        self.dll_directory = os.add_dll_directory(str(directory))
        nvrtc = C.WinDLL(str(directory / 'nvrtc64_120_0.dll'))
        self.driver = C.WinDLL('nvcuda.dll')
        source = Path(__file__).with_suffix('.cu').read_bytes()
        program = C.c_void_p()
        def setup(library, name, types):
            fn = getattr(library,name); fn.restype = C.c_int; fn.argtypes = types
            return fn
        create = setup(nvrtc,'nvrtcCreateProgram',[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p])
        compile_ = setup(nvrtc,'nvrtcCompileProgram',[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)])
        size = setup(nvrtc,'nvrtcGetProgramLogSize',[C.c_void_p,C.POINTER(C.c_size_t)])
        log = setup(nvrtc,'nvrtcGetProgramLog',[C.c_void_p,C.c_void_p])
        ptx_size = setup(nvrtc,'nvrtcGetPTXSize',[C.c_void_p,C.POINTER(C.c_size_t)])
        ptx = setup(nvrtc,'nvrtcGetPTX',[C.c_void_p,C.c_void_p])
        destroy = setup(nvrtc,'nvrtcDestroyProgram',[C.POINTER(C.c_void_p)])
        def check(code, where):
            if code: raise RuntimeError(f'{where}: status {code}')
        check(create(C.byref(program),source,b'fused_norm.cu',0,None,None),'nvrtcCreateProgram')
        options = (C.c_char_p*3)(b'--gpu-architecture=compute_89',b'--std=c++17',b'--fmad=false')
        status = compile_(program,len(options),options)
        length = C.c_size_t(); size(program,C.byref(length))
        message = C.create_string_buffer(length.value); log(program,message)
        if status:
            destroy(C.byref(program)); raise RuntimeError(message.value.decode())
        check(ptx_size(program,C.byref(length)),'nvrtcGetPTXSize')
        code = C.create_string_buffer(length.value); check(ptx(program,code),'nvrtcGetPTX')
        destroy(C.byref(program))
        load = setup(self.driver,'cuModuleLoadDataEx',[C.POINTER(C.c_void_p),C.c_void_p,C.c_uint,C.c_void_p,C.c_void_p])
        lookup = setup(self.driver,'cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
        self.launch = setup(self.driver,'cuLaunchKernel',[C.c_void_p,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p])
        self.module = C.c_void_p(); check(load(C.byref(self.module),code,0,None,None),'cuModuleLoadDataEx')
        self.functions = {}
        self.softmax_functions = {}
        for dtype, name in [(torch.float16,b'cosine_norm_f16'),(torch.float32,b'cosine_norm_f32')]:
            function = C.c_void_p(); check(lookup(C.byref(function),self.module,name),'cuModuleGetFunction')
            self.functions[dtype] = function
        for dtype, name in [(torch.float16,b'bit_affine_softmax_f16'),(torch.float32,b'bit_affine_softmax_f32')]:
            function = C.c_void_p(); check(lookup(C.byref(function),self.module,name),'cuModuleGetFunction')
            self.softmax_functions[dtype] = function
        # Module stays alive until process exit, including any captured graphs.

    def __call__(self, x):
        if x.device.type != 'cuda' or x.shape[-1] != 32 or x.dtype not in self.functions or x.requires_grad:
            raise ValueError('Expected an inference-only CUDA float16/float32 tensor ending in 32 channels.')
        x = x.contiguous()
        output = torch.empty_like(x)
        rows = x.numel()//32
        if not rows: return output
        if rows > 2**30: raise ValueError('Tensor exceeds the bounded launch size.')
        input_arg, output_arg, count_arg = C.c_void_p(x.data_ptr()), C.c_void_p(output.data_ptr()), C.c_uint(rows)
        params = (C.c_void_p*3)(C.addressof(input_arg),C.addressof(output_arg),C.addressof(count_arg))
        stream = torch.cuda.current_stream(x.device)
        status = self.launch(self.functions[x.dtype],(rows*4+255)//256,1,1,256,1,1,0,C.c_void_p(stream.cuda_stream),params,None)
        if status: raise RuntimeError(f'cuLaunchKernel failed: {status}')
        return output

    def softmax(self, x):
        columns = x.shape[-1]
        if x.device.type != 'cuda' or x.dtype not in self.softmax_functions or x.requires_grad or not 2 <= columns <= 2048 or columns%2:
            raise ValueError('Expected inference-only CUDA float16/32 rows of even length 2..2048.')
        x = x.contiguous()
        output = torch.empty_like(x)
        rows = x.numel()//columns
        if not rows: return output
        if rows >= 2**31: raise ValueError('Too many rows.')
        input_arg, output_arg = C.c_void_p(x.data_ptr()), C.c_void_p(output.data_ptr())
        count_arg, column_arg = C.c_uint(rows), C.c_uint(columns)
        params = (C.c_void_p*4)(C.addressof(input_arg),C.addressof(output_arg),C.addressof(count_arg),C.addressof(column_arg))
        threads = min(256,max(32,1 << (columns//2-1).bit_length()))
        stream = torch.cuda.current_stream(x.device)
        status = self.launch(self.softmax_functions[x.dtype],rows,1,1,threads,1,1,0,C.c_void_p(stream.cuda_stream),params,None)
        if status: raise RuntimeError(f'cuLaunchKernel failed: {status}')
        return output
