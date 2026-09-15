# SPDX-License-Identifier: Apache-2.0
"""Offline numerical experiments; never loads a game or vendor DLL.

Model outputs and weights stay local. Synthetic fixtures are smoke tests, not
evidence of perceptual equivalence to the NVIDIA implementation.
"""
from pathlib import Path
import argparse, contextlib, hashlib, json, statistics, sys, time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['audit', 'rounding', 'model', 'graph'], required=True)
    p.add_argument('--image', type=Path)
    p.add_argument('--skip', default='31')
    p.add_argument('--fused-norm', action='store_true', help='Use the experimental local CUDA normalization kernel in graph mode.')
    p.add_argument('--fused-softmax', action='store_true', help='Use the experimental bit-affine row kernel in graph mode.')
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('Output exists; use a fresh experiment path.')
    sys.path.insert(0, str(a.source / 'python'))
    import numpy as np
    from safetensors import safe_open
    spec = json.loads((a.source / 'python/mlxdlss/weight_spec.json').read_text())['tensors']
    with safe_open(a.weights, framework='numpy') as f:
        shapes = {key: list(f.get_slice(key).get_shape()) for key in f.keys()}
        metadata = f.metadata()
    missing = sorted(set(spec) - shapes.keys())
    mismatched = [key for key in spec if key in shapes and spec[key]['shape'] != shapes[key]]
    result = {'schema': 1, 'mode': a.mode,
              'weights_sha256': hashlib.sha256(a.weights.read_bytes()).hexdigest(),
              'weights_count': len(shapes), 'missing': missing, 'shape_mismatches': mismatched,
              'weights_fully_logical': metadata.get('fully_logical'),
              'synthetic_is_not_quality_validation': True, 'game_changed': False}
    if missing or mismatched:
        raise SystemExit(json.dumps(result))
    if a.mode == 'audit':
        result['parameters'] = sum(int(np.prod(shape)) for shape in shapes.values())
    else:
        import torch
        from mlxdlss import model as reference
        if not torch.cuda.is_available():
            raise SystemExit('CUDA unavailable; refusing a misleading CPU timing.')
        torch.manual_seed(1234)
        torch.backends.cuda.matmul.allow_tf32 = False
        result.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                      compute_capability=list(torch.cuda.get_device_capability()))

        def native_roundtrip(x):
            if x.device.type == 'cuda' and x.dtype in (torch.float16, torch.float32):
                return x.clamp(-448, 448).to(torch.float8_e4m3fn).to(x.dtype)
            return original_roundtrip(x)

        original_roundtrip = reference.e4m3_round_trip

        @torch.inference_mode()
        def timed(fn, repeats=10):
            values = []
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            for _ in range(repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                y = fn()
                end.record(); end.synchronize()
                values.append(start.elapsed_time(end))
            return y, {'median_ms': statistics.median(values), 'samples_ms': values}

        if a.mode == 'rounding':
            all_half = np.arange(65536, dtype=np.uint16).view(np.float16)
            finite = all_half[np.isfinite(all_half)].copy()
            x = torch.from_numpy(finite).cuda()
            with torch.inference_mode():
                old, new = original_roundtrip(x), native_roundtrip(x)
            mismatch = int(torch.count_nonzero(old != new).item())
            result['exhaustive_finite_fp16'] = {'count': x.numel(), 'unequal': mismatch,
                'max_abs': float((old - new).abs().max()),
                'signed_zero_bit_mismatches': int(torch.count_nonzero(old.view(torch.int16) != new.view(torch.int16)))}
            x = torch.randn((1, 1088, 1920, 32), device='cuda', dtype=torch.float16)
            old, old_times = timed(lambda: original_roundtrip(x), 12)
            new, new_times = timed(lambda: native_roundtrip(x), 12)
            result['cuda_fp16_conversion'] = {'shape': list(x.shape), 'old': old_times, 'native': new_times,
                'unequal': int(torch.count_nonzero(old != new)), 'applies_to': 'PyTorch reference only'}
        else:
            from mlxdlss.pipeline import NeuralRenderingPipeline
            skip = {int(x) for x in a.skip.split(',') if x}
            allowed = set(range(1,4)) | set(range(5,8)) | set(range(9,14)) | set(range(15,22)) | set(range(23,30)) | set(range(31,39)) | set(range(40,48)) | set(range(49,56)) | set(range(57,62)) | set(range(63,66)) | set(range(67,70))
            if not skip <= allowed:
                raise SystemExit('Cannot remove structural transition, input, or output blocks.')
            if a.image:
                from PIL import Image
                source = np.asarray(Image.open(a.image).convert('RGB').resize((320,320)), dtype=np.float32) / 255
                result['fixture'] = 'user-supplied image resized to 320x320; not a full-resolution quality test'
            else:
                yy, xx = np.indices((320,320), dtype=np.float32)
                source = np.stack((xx/319, yy/319, 0.5+0.3*np.sin(xx*0.21)*np.cos(yy*0.13)), axis=-1)
                result['fixture'] = 'deterministic synthetic gradient and sinusoidal detail, 320x320'
            pipeline = NeuralRenderingPipeline.from_safetensors(a.weights, device='cuda', precision='fast')
            prepared = pipeline.prepare(source)
            tensor = torch.from_numpy(prepared.features[None]).cuda().half()
            original, baseline_timing = timed(lambda: pipeline.model(tensor), 2)
            reference.e4m3_round_trip = native_roundtrip
            exact, exact_timing = timed(lambda: pipeline.model(tensor), 2)
            result['native_roundtrip_model'] = {'baseline': baseline_timing, 'candidate': exact_timing,
                'head_unequal': int(torch.count_nonzero(original != exact)),
                'head_max_abs': float((original-exact).abs().max())}
            if a.mode == 'graph':
                if a.fused_norm or a.fused_softmax:
                    from fused_norm import FusedNorm
                    fused = FusedNorm()
                if a.fused_norm:
                    previous_norm = reference.vendor_cosine_normalize
                    def custom_norm(x):
                        return fused(x) if x.device.type == 'cuda' and x.shape[-1] == 32 else previous_norm(x)
                    reference.vendor_cosine_normalize = custom_norm
                result['uses_fused_norm'] = a.fused_norm
                if a.fused_softmax:
                    previous_softmax = reference.vendor_approximate_softmax
                    def custom_softmax(x):
                        return fused.softmax(x) if x.device.type == 'cuda' and 2 <= x.shape[-1] <= 2048 and x.shape[-1]%2 == 0 else previous_softmax(x)
                    reference.vendor_approximate_softmax = custom_softmax
                result['uses_fused_softmax'] = a.fused_softmax
                # Weights are immutable in this inference experiment. Hoist the
                # CPU-created index and constant bias permutation out of capture.
                bias_layout = reference.recover_attention_bias_layout
                bias_cache = {}
                def cached_bias(bias):
                    key = (bias.data_ptr(), tuple(bias.shape), bias.dtype, bias.device)
                    if key not in bias_cache:
                        bias_cache[key] = bias_layout(bias)
                    return bias_cache[key]
                reference.recover_attention_bias_layout = cached_bias
                with torch.inference_mode():
                    warmup = torch.cuda.Stream()
                    warmup.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(warmup):
                        for _ in range(3):
                            pipeline.model(tensor)
                    torch.cuda.current_stream().wait_stream(warmup)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_output = pipeline.model(tensor)
                    def replay():
                        graph.replay()
                        return graph_output
                    replayed, replay_timing = timed(replay, 12)
                    result['cuda_graph'] = {'timing': replay_timing,
                        'head_unequal': int(torch.count_nonzero(exact != replayed)),
                        'head_max_abs': float((exact-replayed).abs().max()),
                        'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                        'applies_to': 'PyTorch reconstruction; fixed shape and fixed inputs in this smoke test'}
                result['network_shape'] = list(tensor.shape)
                a.output.parent.mkdir(parents=True, exist_ok=True)
                a.output.write_text(json.dumps(result, indent=2))
                print(json.dumps(result, indent=2), flush=True)
                return
            wrappers = {}
            for name in ('_window', '_split_window', '_global'):
                old = getattr(pipeline.model, name)
                wrappers[name] = old
                def wrapper(value, index, *args, _old=old, **kwargs):
                    return value if index in skip else _old(value, index, *args, **kwargs)
                setattr(pipeline.model, name, wrapper)
            ablated, ablated_timing = timed(lambda: pipeline.model(tensor), 2)
            delta = (exact.float()-ablated.float()).abs()
            result['ablation'] = {'removed_blocks': sorted(skip), 'timing': ablated_timing,
                'head_mae': float(delta.mean()), 'head_max_abs': float(delta.max()),
                'head_rmse': float(torch.mean(delta.square()).sqrt()),
                'candidate_accepted': False, 'reason': 'Diagnostic only; no vendor or temporal quality gate.'}
            result['network_shape'] = list(tensor.shape)
            reference.e4m3_round_trip = original_roundtrip
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
