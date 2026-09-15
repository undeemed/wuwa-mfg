# SPDX-License-Identifier: Apache-2.0
"""Single-head block-0 diagnostic using direct FP8 MMA and optional initial C.

The recovered MLX-DLSS schedule supplies the pointwise math and window mapping.
This implementation is for numerical diagnosis, not a tuned production kernel.
"""
import torch


class MmaFirstBlock:
    def __init__(self, reference, model, kernel):
        self.reference, self.model, self.kernel = reference, model, kernel
        self.weights = {}
        for name in ('weight1', 'weight2', 'qkv_weight', 'projection_weight'):
            value = self.weight(name)
            packed = value.to(torch.float8_e4m3fn)
            if not torch.equal(packed.to(value.dtype), value):
                raise ValueError(f'{name} is not exactly representable in E4M3.')
            self.weights[name] = packed

    def weight(self, name):
        return self.model.weight('block0.layer0.' + name)

    @staticmethod
    def pack(value):
        return value.clamp(-448, 448).to(torch.float8_e4m3fn)

    def __call__(self, value, *, attention_mma=False, seed_residual=False, seed_logits=False):
        ref, kernel = self.reference, self.kernel
        if value.ndim != 4 or value.shape[-1] != 32 or value.dtype != torch.float16:
            raise ValueError('Expected FP16 [batch,height,width,32] block-0 adapter output.')
        if value.shape[1] % 8 or value.shape[2] % 8:
            raise ValueError('Block-0 diagnostic requires complete 8x8 windows.')

        def feed_forward(tokens):
            flat = tokens.reshape(-1, 32)
            expanded = kernel.mma(self.pack(flat), self.weights['weight1'])
            activated = self.pack(ref.quadratic_gate_activation(expanded))
            seed = flat * self.weight('ffn_cos_skip') if seed_residual else None
            projected = kernel.mma(activated, self.weights['weight2'], seed)
            if not seed_residual:
                projected = ref.cosine_residual(flat, projected, self.weight('ffn_cos_skip'))
            return projected.reshape(tokens.shape)

        residual = ref._per_token(feed_forward, value)
        bias = self.weight('attn_bias')
        if ref.uses_fragment_swizzle(0, 1):
            bias = ref.recover_attention_bias_layout(bias)
        if not attention_mma:
            branch = ref.window_attention(
                ref.e4m3_round_trip(residual), qkv_weight=self.weight('qkv_weight'),
                attention_scale=self.weight('attn_scale'), attention_bias=bias,
                projection_weight=self.weight('projection_weight'), head_count=1,
                window_size=8, window_origin=(0, 0))
            return ref._residual_per_token(residual, branch, self.weight('attn_cos_skip'))

        def attend(strip):
            windows = ref.partition_windows(strip, 8)
            projected = kernel.mma(self.pack(windows), self.weights['qkv_weight'])
            query, key, values = projected.chunk(3, dim=-1)
            query = ref.vendor_cosine_publish(query.unsqueeze(1), self.weight('attn_scale'))
            key = ref.vendor_cosine_publish(key.unsqueeze(1))
            values = self.pack(values.unsqueeze(1))
            scores = kernel.mma(self.pack(query), self.pack(key.transpose(-2, -1)),
                                bias[0] if seed_logits else None)
            if not seed_logits:
                scores = scores + bias
            probabilities = self.pack(ref.vendor_approximate_softmax(scores))
            attended = kernel.mma(probabilities, values).squeeze(1)
            seed = windows * self.weight('attn_cos_skip') if seed_residual else None
            output = kernel.mma(self.pack(attended), self.weights['projection_weight'], seed)
            if not seed_residual:
                output = ref.cosine_residual(windows, output, self.weight('attn_cos_skip'))
            return ref.reverse_windows(output, batch_count=strip.shape[0], height=strip.shape[1],
                                       width=strip.shape[2], window_size=8)

        batch, height, width, _ = residual.shape
        rows_per_strip = max(8, (ref.CHUNK_TOKENS // (batch * width * 8)) * 8) if ref.CHUNK_TOKENS else height
        output = torch.empty_like(residual)
        for y0 in range(0, height, rows_per_strip):
            output[:, y0:y0+rows_per_strip] = attend(residual[:, y0:y0+rows_per_strip])
        return output
