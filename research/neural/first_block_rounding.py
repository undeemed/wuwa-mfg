# SPDX-License-Identifier: Apache-2.0
"""Experimental block-0 branch-input rounding, validated on captured prefixes.

Adapted from the pinned MLX-DLSS single-head window block. The residual operands
remain in their original precision; only inputs to the two branches are E4M3.
This is not a general correction for every block family or a quality acceptance.
"""


def block_zero(reference, model, value, *, round_ffn=True, round_qkv=True):
    weight = lambda name: model.weight('block0.layer0.' + name)
    def feed_forward(tokens):
        branch_input = reference.e4m3_round_trip(tokens) if round_ffn else tokens
        branch = reference.e4m3_round_trip(reference.quadratic_gate_activation(
            branch_input @ weight('weight1'))) @ weight('weight2')
        return reference.cosine_residual(tokens, branch, weight('ffn_cos_skip'))
    residual = reference._per_token(feed_forward, value)
    bias = weight('attn_bias')
    if reference.uses_fragment_swizzle(0, 1):
        bias = reference.recover_attention_bias_layout(bias)
    branch_input = reference.e4m3_round_trip(residual) if round_qkv else residual
    branch = reference.window_attention(
        branch_input, qkv_weight=weight('qkv_weight'), attention_scale=weight('attn_scale'),
        attention_bias=bias, projection_weight=weight('projection_weight'), head_count=1,
        window_size=8, window_origin=reference.recovered_window_origin(0))
    return reference._residual_per_token(residual, branch, weight('attn_cos_skip'))


def install_block_zero(reference, model):
    """Install on one inference model instance, keeping all other blocks unchanged."""
    original = model._window
    def window(value, index, *, head_count, publish=True):
        if index != 0:
            return original(value, index, head_count=head_count, publish=publish)
        if head_count != 1:
            raise ValueError('Only the single-head block-0 contract is supported.')
        return block_zero(reference, model, value)
    model._window = window
    return original
