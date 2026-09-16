# SPDX-License-Identifier: Apache-2.0
"""Load a locally compiled kernel into the current PyTorch CUDA process only."""
import ctypes as C
import math
import os
from pathlib import Path
import torch

class TensorLayout(C.Structure):
    _fields_=[('sizes',C.c_ulonglong*8),('strides',C.c_ulonglong*8),('rank',C.c_uint)]

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
        self.gate_functions = {}
        self.publish_functions = {}
        self.roundtrip_functions = {}
        self.grade_functions = {}
        for dtype, name in [(torch.float16,b'cosine_norm_f16'),(torch.float32,b'cosine_norm_f32')]:
            function = C.c_void_p(); check(lookup(C.byref(function),self.module,name),'cuModuleGetFunction')
            self.functions[dtype] = function
        for dtype, name in [(torch.float16,b'bit_affine_softmax_f16'),(torch.float32,b'bit_affine_softmax_f32')]:
            function = C.c_void_p(); check(lookup(C.byref(function),self.module,name),'cuModuleGetFunction')
            self.softmax_functions[dtype] = function
        for dtype, name in [(torch.float16,b'quadratic_activation_f16'),(torch.float32,b'quadratic_activation_f32')]:
            function = C.c_void_p(); check(lookup(C.byref(function),self.module,name),'cuModuleGetFunction')
            self.gate_functions[dtype] = function
        for dtype, name in [(torch.float16,b'cosine_publish_f16'),(torch.float32,b'cosine_publish_f32')]:
            function = C.c_void_p(); check(lookup(C.byref(function),self.module,name),'cuModuleGetFunction')
            self.publish_functions[dtype] = function
        for dtype, name in [(torch.float16,b'fp8_roundtrip_f16'),(torch.float32,b'fp8_roundtrip_f32')]:
            function = C.c_void_p(); check(lookup(C.byref(function),self.module,name),'cuModuleGetFunction')
            self.roundtrip_functions[dtype] = function
        for dtype,name in [(torch.float16,b'output_grade_f16'),(torch.float32,b'output_grade_f32')]:
            function=C.c_void_p();check(lookup(C.byref(function),self.module,name),'cuModuleGetFunction')
            self.grade_functions[dtype]=function
        self.noise_function = C.c_void_p()
        check(lookup(C.byref(self.noise_function),self.module,b'gaussian_noise_f32'),'cuModuleGetFunction')
        self.mma_function = C.c_void_p()
        check(lookup(C.byref(self.mma_function),self.module,b'fp8_mma_f16'),'cuModuleGetFunction')
        self.half_mma_function = C.c_void_p()
        check(lookup(C.byref(self.half_mma_function),self.module,b'half_mma_f16'),'cuModuleGetFunction')
        self.affine_function=C.c_void_p()
        check(lookup(C.byref(self.affine_function),self.module,b'affine_compose_f16'),'cuModuleGetFunction')
        self.pack_function=C.c_void_p()
        check(lookup(C.byref(self.pack_function),self.module,b'fp8_pack_f16'),'cuModuleGetFunction')
        self.decoder_function=C.c_void_p()
        check(lookup(C.byref(self.decoder_function),self.module,b'decoder_upscale_add_f16'),'cuModuleGetFunction')
        self.student_output_function=C.c_void_p()
        check(lookup(C.byref(self.student_output_function),self.module,b'student_output_f16'),'cuModuleGetFunction')
        self.residual_function=C.c_void_p()
        check(lookup(C.byref(self.residual_function),self.module,b'residual_scale_add_f16'),'cuModuleGetFunction')
        self.dense_residual_functions={}
        for channels in (16,32,64,96,128,192):
            function=C.c_void_p()
            check(lookup(C.byref(function),self.module,('residual_dense_'+str(channels)).encode()),'cuModuleGetFunction')
            self.dense_residual_functions[channels]=function
        self.conditioning_function=C.c_void_p()
        check(lookup(C.byref(self.conditioning_function),self.module,b'decoder_conditioned_f16'),'cuModuleGetFunction')
        self.dense_conditioning_functions={}
        for channels in (16,32,64,128):
            function=C.c_void_p()
            check(lookup(C.byref(function),self.module,('conditioned_dense_'+str(channels)).encode()),'cuModuleGetFunction')
            self.dense_conditioning_functions[channels]=function
        # Module stays alive until process exit, including any captured graphs.

    def decoder_conditioned_add(self,x,skip,scale,shift):
        tensors=(x,skip,scale,shift)
        if any(t.device.type!='cuda' or t.device!=x.device or t.dtype!=torch.float16
               or t.ndim!=4 or t.requires_grad for t in tensors):
            raise ValueError('Expected same-device inference-only CUDA NCHW FP16 tensors.')
        batch,channels,height,width=skip.shape
        if not skip.numel() or skip.numel()>=2**31:
            raise ValueError('Tensor exceeds the bounded launch extent.')
        if x.shape==skip.shape:ratio=1
        elif height%2==0 and width%2==0 and x.shape==(batch,channels,height//2,width//2):ratio=2
        else:raise ValueError('Input must match the skip or be exactly half its spatial extent.')
        if scale.shape!=(batch,channels,1,1) or shift.shape!=scale.shape:
            raise ValueError('Expected one scale and shift per image/channel.')
        output=torch.empty((batch,height,width,channels),device=x.device,dtype=x.dtype).permute(0,3,1,2)
        arguments=[C.c_void_p(t.data_ptr()) for t in (*tensors,output)]
        dense=(channels in self.dense_conditioning_functions
               and all(t.is_contiguous(memory_format=torch.channels_last) for t in (x,skip))
               and all(t.data_ptr()%4==0 for t in tensors)
               and all(t.stride(1)==1 and t.stride(0)%2==0 for t in (scale,shift)))
        count=skip.numel()//2 if dense else skip.numel()
        arguments += [C.c_uint(v) for v in (count,channels,height,width,ratio)]
        if dense:
            arguments += [C.c_ulonglong(t.stride(0)) for t in (scale,shift)]
            function=self.dense_conditioning_functions[channels]
        else:
            arguments += [C.c_ulonglong(s) for t in (x,skip) for s in t.stride()]
            arguments += [C.c_ulonglong(s) for t in (scale,shift) for s in t.stride()[:2]]
            function=self.conditioning_function
        params=(C.c_void_p*len(arguments))(*(C.addressof(v) for v in arguments))
        status=self.launch(function,(count+255)//256,1,1,256,1,1,0,
                           C.c_void_p(torch.cuda.current_stream(x.device).cuda_stream),params,None)
        if status:raise RuntimeError(f'cuLaunchKernel failed: {status}')
        return output

    def residual_scale_add(self,x,residual,scale):
        if any(t.device.type!='cuda' or t.device!=x.device or t.dtype!=torch.float16
               or t.ndim!=4 or t.requires_grad for t in (x,residual,scale)):
            raise ValueError('Expected same-device inference-only CUDA NCHW FP16 tensors.')
        batch,channels,height,width=x.shape
        if residual.shape!=x.shape or scale.shape!=(1,channels,1,1):
            raise ValueError('Expected matching residual extent and one scale per channel.')
        count=x.numel()
        if not count or count>=2**31:
            raise ValueError('Tensor exceeds the bounded launch extent.')
        output=torch.empty((batch,height,width,channels),device=x.device,dtype=x.dtype).permute(0,3,1,2)
        arguments=[C.c_void_p(t.data_ptr()) for t in (x,residual,scale,output)]
        dense=(channels in self.dense_residual_functions and scale.stride(1)==1
               and all(t.data_ptr()%4==0 for t in (x,residual,scale))
               and all(t.is_contiguous(memory_format=torch.channels_last) for t in (x,residual)))
        if dense:
            arguments += [C.c_uint(count//2)]
            function=self.dense_residual_functions[channels]
            threads=(count//2+255)//256
        else:
            arguments += [C.c_uint(v) for v in (count,channels,height,width)]
            arguments += [C.c_ulonglong(s) for t in (x,residual) for s in t.stride()]
            arguments += [C.c_ulonglong(scale.stride(1))]
            function=self.residual_function
            threads=(count+255)//256
        params=(C.c_void_p*len(arguments))(*(C.addressof(v) for v in arguments))
        stream=torch.cuda.current_stream(x.device)
        status=self.launch(function,threads,1,1,256,1,1,0,
                           C.c_void_p(stream.cuda_stream),params,None)
        if status:raise RuntimeError(f'cuLaunchKernel failed: {status}')
        return output

    def student_output(self,source,head,parameters):
        if any(t.device.type!='cuda' or t.device!=source.device or t.dtype!=torch.float16
               or t.ndim!=4 or t.requires_grad for t in (source,head)):
            raise ValueError('Expected same-device inference-only CUDA NCHW FP16 tensors.')
        batch,channels,height,width=source.shape
        if channels!=3 or head.shape!=(batch,48,(height+31)//32*8,(width+31)//32*8):
            raise ValueError('Expected RGB and a 48-channel head covering the image padded to multiples of 32.')
        exposure,contrast,saturation=parameters
        if not (all(math.isfinite(v) for v in parameters) and 0<exposure<=16
                and -1<=contrast<=1 and 0<=saturation<=1):
            raise ValueError('Unsupported finite grading parameters.')
        pixels=batch*height*width
        if not pixels or pixels>=2**31:
            raise ValueError('Image exceeds the bounded launch extent.')
        output=torch.empty((batch,height,width,3),device=source.device,dtype=source.dtype).permute(0,3,1,2)
        arguments=[C.c_void_p(t.data_ptr()) for t in (source,head,output)]
        arguments += [C.c_uint(v) for v in (pixels,height,width)]
        arguments += [C.c_ulonglong(s) for t in (source,head) for s in t.stride()]
        arguments += [C.c_float(v) for v in parameters]
        params=(C.c_void_p*len(arguments))(*(C.addressof(v) for v in arguments))
        stream=torch.cuda.current_stream(source.device)
        status=self.launch(self.student_output_function,(pixels+255)//256,1,1,256,1,1,0,
                           C.c_void_p(stream.cuda_stream),params,None)
        if status:raise RuntimeError(f'cuLaunchKernel failed: {status}')
        return output

    def decoder_upscale_add(self,x,skip):
        if any(t.device.type!='cuda' or t.device!=x.device or t.dtype!=torch.float16
               or t.ndim!=4 or t.requires_grad for t in (x,skip)):
            raise ValueError('Expected same-device inference-only CUDA NCHW FP16 tensors.')
        batch,channels,height,width=skip.shape
        if x.shape!=(batch,channels,height//2,width//2) or height%2 or width%2:
            raise ValueError('The skip extent must be exactly twice the input in both spatial dimensions.')
        count=skip.numel()
        if not count or count>=2**31:
            raise ValueError('Tensor exceeds the bounded launch extent.')
        output=torch.empty((batch,height,width,channels),device=x.device,dtype=x.dtype).permute(0,3,1,2)
        arguments=[C.c_void_p(t.data_ptr()) for t in (x,skip,output)]
        arguments += [C.c_uint(v) for v in (count,channels,height,width)]
        arguments += [C.c_ulonglong(s) for t in (x,skip) for s in t.stride()]
        params=(C.c_void_p*len(arguments))(*(C.addressof(v) for v in arguments))
        stream=torch.cuda.current_stream(x.device)
        status=self.launch(self.decoder_function,(count+255)//256,1,1,256,1,1,0,
                           C.c_void_p(stream.cuda_stream),params,None)
        if status:raise RuntimeError(f'cuLaunchKernel failed: {status}')
        return output

    def affine_compose(self,source,detail,coefficients):
        tensors=(source,detail,coefficients)
        if any(x.device!=source.device or x.device.type!='cuda' or x.dtype!=torch.float16
               or x.requires_grad or x.ndim!=4 for x in tensors):
            raise ValueError('Expected same-device inference-only CUDA NCHW float16 tensors.')
        if source.shape[1]!=3 or detail.shape!=source.shape:
            raise ValueError('Source and detail must have matching RGB extents.')
        batch,_,height,width=source.shape
        if coefficients.shape!=(batch,12,(height+31)//32,(width+31)//32):
            raise ValueError('Expected twelve coefficients per padded 32x32 cell.')
        pixels=batch*height*width
        if not pixels or pixels>=2**31:raise ValueError('Image exceeds the bounded launch extent.')
        output=torch.empty((batch,height,width,3),device=source.device,dtype=source.dtype).permute(0,3,1,2)
        arguments=[*(C.c_void_p(x.data_ptr()) for x in tensors),C.c_void_p(output.data_ptr()),
                   C.c_uint(pixels),C.c_uint(height),C.c_uint(width),
                   *(C.c_ulonglong(s) for x in tensors for s in x.stride())]
        params=(C.c_void_p*len(arguments))(*(C.addressof(v) for v in arguments))
        stream=torch.cuda.current_stream(source.device)
        status=self.launch(self.affine_function,(pixels+255)//256,1,1,256,1,1,0,
                           C.c_void_p(stream.cuda_stream),params,None)
        if status:raise RuntimeError(f'cuLaunchKernel failed: {status}')
        return output

    def output_grade(self,x,parameters):
        if (x.device.type!='cuda' or x.ndim!=4 or x.shape[1]!=3
                or x.dtype not in self.grade_functions or x.requires_grad):
            raise ValueError('Expected inference-only CUDA NCHW RGB float16/32.')
        exposure,contrast,saturation=parameters
        if not (all(math.isfinite(v) for v in parameters) and 0<exposure<=16
                and -1<=contrast<=1 and 0<=saturation<=1):
            raise ValueError('Unsupported finite grading parameters.')
        batch,_,height,width=x.shape
        pixels=batch*height*width
        if not pixels or pixels>=2**31:raise ValueError('Image exceeds the bounded launch extent.')
        output=torch.empty((batch,height,width,3),device=x.device,dtype=x.dtype).permute(0,3,1,2)
        arguments=[C.c_void_p(x.data_ptr()),C.c_void_p(output.data_ptr()),
                   C.c_uint(pixels),C.c_uint(height),C.c_uint(width),
                   *(C.c_ulonglong(s) for s in x.stride()),
                   C.c_float(exposure),C.c_float(contrast),C.c_float(saturation)]
        params=(C.c_void_p*len(arguments))(*(C.addressof(v) for v in arguments))
        stream=torch.cuda.current_stream(x.device)
        status=self.launch(self.grade_functions[x.dtype],(pixels+255)//256,1,1,256,1,1,0,
                           C.c_void_p(stream.cuda_stream),params,None)
        if status:raise RuntimeError(f'cuLaunchKernel failed: {status}')
        return output

    def mma(self, a, b, seed=None):
        if (a.device.type!='cuda' or b.device!=a.device or a.ndim<2 or b.ndim<2
                or a.dtype not in (torch.float8_e4m3fn,torch.float16) or b.dtype!=a.dtype
                or a.requires_grad or b.requires_grad):
            raise ValueError('Expected inference-only, same-device FP16 or E4M3 tensors with matching dtypes.')
        m,k=a.shape[-2:];bk,n=b.shape[-2:]
        batches=a.numel()//(m*k) if m*k else 0
        step=16 if a.dtype==torch.float16 else 32
        if not (m>0 and n>0 and k>0 and k%step==0 and bk==k and batches>0):
            raise ValueError(f'Positive M/N and matching K divisible by {step} are required.')
        if b.ndim!=2 and b.shape[:-2]!=a.shape[:-2]:
            raise ValueError('B must be shared 2D or have matching batch dimensions.')
        shape=(*a.shape[:-1],n)
        if seed is not None and (seed.device!=a.device or seed.dtype!=torch.float16 or seed.requires_grad
                                  or not 2<=seed.ndim<=len(shape) or tuple(seed.shape)!=shape[-seed.ndim:]):
            raise ValueError('Seed must be FP16, same-device, and match a trailing output shape including MxN.')
        tiles=((m+15)//16)*((n+7)//8)*batches
        if max(m,n,k,batches)>=2**31 or tiles>=2**32:
            raise ValueError('MMA launch exceeds its bounded indexing contract.')
        a=a.contiguous();b=b.contiguous()
        # Packed A loads require four-byte alignment, even for contiguous views.
        if a.data_ptr()%4:a=a.clone()
        if seed is not None:seed=seed.contiguous()
        output=torch.empty(shape,device=a.device,dtype=torch.float16)
        arguments=[C.c_void_p(a.data_ptr()),C.c_void_p(b.data_ptr()),
                   C.c_void_p(seed.data_ptr()) if seed is not None else C.c_void_p(),
                   C.c_void_p(output.data_ptr()),C.c_uint(m),C.c_uint(n),C.c_uint(k),C.c_uint(batches),
                   C.c_ulonglong(0 if b.ndim==2 else k*n),
                   C.c_ulonglong(0 if seed is None or seed.ndim==2 else m*n),
                   C.c_uint(1 if seed is None else seed.numel()//(m*n))]
        params=(C.c_void_p*len(arguments))(*(C.addressof(v) for v in arguments))
        stream=torch.cuda.current_stream(a.device)
        function=self.half_mma_function if a.dtype==torch.float16 else self.mma_function
        status=self.launch(function,(tiles+3)//4,1,1,128,1,1,0,
                           C.c_void_p(stream.cuda_stream),params,None)
        if status:raise RuntimeError(f'cuLaunchKernel failed: {status}')
        return output

    def noise(self, height, width, frame_index=0):
        if not (0 < height <= 8192 and 0 < width <= 8192 and 0 <= frame_index < 2**32):
            raise ValueError('Expected bounded positive extents and a uint32 frame index.')
        output = torch.empty((height,width,3),device='cuda',dtype=torch.float32)
        arguments = [C.c_void_p(output.data_ptr()),C.c_uint(width),C.c_uint(height),C.c_uint(frame_index)]
        params = (C.c_void_p*4)(*(C.addressof(v) for v in arguments))
        stream = torch.cuda.current_stream(output.device)
        status = self.launch(self.noise_function,(height*width+255)//256,1,1,256,1,1,0,
                             C.c_void_p(stream.cuda_stream),params,None)
        if status: raise RuntimeError(f'cuLaunchKernel failed: {status}')
        return output

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

    def gate(self, x):
        if x.device.type != 'cuda' or x.dtype not in self.gate_functions or x.requires_grad:
            raise ValueError('Expected inference-only CUDA float16/float32.')
        x=x.contiguous()
        output=torch.empty_like(x)
        count=x.numel()
        if not count:return output
        if count>=2**38:raise ValueError('Tensor exceeds bounded launch size.')
        input_arg,output_arg,count_arg=C.c_void_p(x.data_ptr()),C.c_void_p(output.data_ptr()),C.c_ulonglong(count)
        params=(C.c_void_p*3)(C.addressof(input_arg),C.addressof(output_arg),C.addressof(count_arg))
        stream=torch.cuda.current_stream(x.device)
        status=self.launch(self.gate_functions[x.dtype],(count+255)//256,1,1,256,1,1,0,C.c_void_p(stream.cuda_stream),params,None)
        if status:raise RuntimeError(f'cuLaunchKernel failed: {status}')
        return output

    def publish(self, x, scale=None):
        if x.device.type!='cuda' or x.ndim!=4 or x.shape[-1]!=32 or x.dtype not in self.publish_functions or x.requires_grad:
            raise ValueError('Expected inference-only CUDA [batch,heads,tokens,32] float16/32.')
        batch,heads,tokens,_=x.shape
        rows=batch*heads*tokens
        if rows>2**30 or heads*tokens>=2**32:raise ValueError('Tensor exceeds bounded launch size.')
        if scale is not None:
            if scale.shape!=(heads,) or scale.device!=x.device or scale.requires_grad:
                raise ValueError('Expected a same-device inference scale per head.')
            scale=scale.to(x.dtype).contiguous()
        output=torch.empty(x.shape,device=x.device,dtype=x.dtype)
        if not rows:return output
        # Read the original strided Q/K view directly: no contiguous input copy.
        arguments=[C.c_void_p(x.data_ptr()),C.c_void_p(output.data_ptr()),C.c_uint(rows),
                   C.c_void_p(scale.data_ptr()) if scale is not None else C.c_void_p(),
                   C.c_uint(heads),C.c_uint(tokens),*(C.c_ulonglong(s) for s in x.stride())]
        params=(C.c_void_p*len(arguments))(*(C.addressof(v) for v in arguments))
        stream=torch.cuda.current_stream(x.device)
        status=self.launch(self.publish_functions[x.dtype],(rows*4+255)//256,1,1,256,1,1,0,
                           C.c_void_p(stream.cuda_stream),params,None)
        if status:raise RuntimeError(f'cuLaunchKernel failed: {status}')
        return output

    def pack(self,x,*,activate=False):
        if x.device.type!='cuda' or x.ndim>8 or x.dtype!=torch.float16 or x.requires_grad:
            raise ValueError('Expected inference-only CUDA float16 with rank <=8.')
        count=x.numel()
        if count>=2**38:raise ValueError('Tensor exceeds bounded launch size.')
        output=torch.empty(x.shape,device=x.device,dtype=torch.float8_e4m3fn)
        if not count:return output
        layout=TensorLayout()
        if x.is_contiguous():
            layout.rank=1;layout.sizes[0]=count;layout.strides[0]=1
        else:
            layout.rank=x.ndim
            for i,(size,stride) in enumerate(zip(x.shape,x.stride())):
                layout.sizes[i]=size;layout.strides[i]=stride
        arguments=[C.c_void_p(x.data_ptr()),C.c_void_p(output.data_ptr()),C.c_ulonglong(count),layout,C.c_uint(bool(activate))]
        params=(C.c_void_p*len(arguments))(*(C.addressof(v) for v in arguments))
        stream=torch.cuda.current_stream(x.device)
        status=self.launch(self.pack_function,(count+511)//512,1,1,256,1,1,0,
                           C.c_void_p(stream.cuda_stream),params,None)
        if status:raise RuntimeError(f'cuLaunchKernel failed: {status}')
        return output

    def roundtrip(self,x):
        if x.device.type!='cuda' or x.ndim>8 or x.dtype not in self.roundtrip_functions or x.requires_grad:
            raise ValueError('Expected inference-only CUDA float16/32 with rank <=8.')
        count=x.numel()
        if count>=2**38:raise ValueError('Tensor exceeds bounded launch size.')
        output=torch.empty(x.shape,device=x.device,dtype=x.dtype)
        if not count:return output
        layout=TensorLayout()
        if x.is_contiguous():
            layout.rank=1;layout.sizes[0]=count;layout.strides[0]=1
        else:
            layout.rank=x.ndim
            for i,(size,stride) in enumerate(zip(x.shape,x.stride())):
                layout.sizes[i]=size;layout.strides[i]=stride
        arguments=[C.c_void_p(x.data_ptr()),C.c_void_p(output.data_ptr()),C.c_ulonglong(count),layout]
        params=(C.c_void_p*4)(*(C.addressof(v) for v in arguments))
        stream=torch.cuda.current_stream(x.device)
        status=self.launch(self.roundtrip_functions[x.dtype],(count+255)//256,1,1,256,1,1,0,
                           C.c_void_p(stream.cuda_stream),params,None)
        if status:raise RuntimeError(f'cuLaunchKernel failed: {status}')
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
