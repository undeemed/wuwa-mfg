# SPDX-License-Identifier: Apache-2.0
"""Direct-MMA diagnostic for the recovered 2/4/8-head branched window blocks.

The pinned MLX-DLSS reference supplies the mathematical graph and window map.
Accumulation changes are hypotheses checked against complete native images;
this is not a tuned production kernel or a native runtime patch.
"""
import torch

BRANCHED=set(range(5,23))|set(range(48,66))


class MmaBranchedBlock:
    def __init__(self,reference,model,kernel,*,block_index,compact_bias=False):
        if block_index not in BRANCHED:raise ValueError('Unknown branched window block.')
        self.reference,self.model,self.kernel=reference,model,kernel
        self.index=block_index
        self.compact_bias=compact_bias
        expansion=self.weight('ffn_expand_weight')
        self.heads=expansion.shape[0];self.channels=self.heads*32
        if self.heads not in (2,4,8) or expansion.shape!=(self.heads,4,self.heads,32,32):
            raise ValueError('Unexpected branched expansion layout.')
        contraction=self.weight('ffn_branch_projection_weight')
        if contraction.shape!=(self.heads,4,32,32):raise ValueError('Unexpected branch contraction layout.')
        def pack_weight(value):
            packed=value.to(torch.float8_e4m3fn)
            if not torch.equal(packed.half(),value):raise ValueError('Weight is not exactly representable in E4M3.')
            return packed.contiguous()
        self.expansion=pack_weight(expansion.permute(2,3,0,1,4).reshape(self.channels,self.heads*128))
        self.contraction=pack_weight(contraction.reshape(self.heads,128,32))
        self.output_projection=pack_weight(self.weight('ffn_output_projection_weight'))
        self.qkv=pack_weight(self.weight('qkv_weight'))
        self.projection=pack_weight(self.weight('projection_weight'))
        if self.output_projection.shape!=(self.channels,self.channels) or self.qkv.shape!=(self.channels,self.channels*3):
            raise ValueError('Unexpected projection extent.')

    def weight(self,name):return self.model.weight(f'block{self.index}.layer0.'+name)

    def __call__(self,value,*,ffn_mma=True,attention_mma=True):
        ref,kernel=self.reference,self.kernel
        channels,heads=self.channels,self.heads
        if value.dtype!=torch.float16 or value.ndim!=4 or value.shape[-1]!=channels:
            raise ValueError('Expected FP16 NHWC features with the block channel count.')
        def feed_forward(tokens):
            if not ffn_mma:
                branch=ref.branched_feed_forward(tokens,
                    expansion_weight=self.weight('ffn_expand_weight'),
                    branch_projection_weight=self.weight('ffn_branch_projection_weight'),
                    output_projection_weight=self.weight('ffn_output_projection_weight'))
                return ref.e4m3_round_trip(ref.cosine_residual(tokens,branch,self.weight('ffn_cos_skip')))
            flat=tokens.reshape(-1,channels)
            expanded=kernel.mma(kernel.pack(flat),self.expansion)
            activated=kernel.pack(expanded,activate=True).reshape(-1,heads,128).permute(1,0,2)
            contracted=kernel.mma(activated,self.contraction).permute(1,0,2).reshape(-1,channels)
            projected=kernel.mma(kernel.pack(contracted),self.output_projection,flat*self.weight('ffn_cos_skip'))
            return ref.e4m3_round_trip(projected.reshape(tokens.shape))
        residual=ref._per_token(feed_forward,value)
        bias=self.weight('attn_bias')
        if ref.uses_fragment_swizzle(self.index,heads):bias=ref.recover_attention_bias_layout(bias)
        origin=ref.recovered_window_origin(self.index)
        if not attention_mma:
            branch=ref.window_attention(residual,qkv_weight=self.weight('qkv_weight'),
                attention_scale=self.weight('attn_scale'),attention_bias=bias,
                projection_weight=self.weight('projection_weight'),head_count=heads,
                window_size=8,window_origin=origin)
            return ref._residual_per_token(residual,branch,self.weight('attn_cos_skip'))
        def attend(strip):
            windows=ref.partition_windows(strip,8)
            projected=kernel.mma(kernel.pack(windows),self.qkv)
            query,key,values=projected.chunk(3,dim=-1)
            def per_head(x):return x.reshape(windows.shape[0],64,heads,32).permute(0,2,1,3)
            query=ref.vendor_cosine_publish(per_head(query),self.weight('attn_scale'))
            key=ref.vendor_cosine_publish(per_head(key))
            values=kernel.pack(per_head(values))
            seed=bias if self.compact_bias else bias.reshape(1,heads,64,64).expand(windows.shape[0],-1,-1,-1)
            scores=kernel.mma(kernel.pack(query),kernel.pack(key.transpose(-2,-1)),seed)
            probabilities=kernel.pack(ref.vendor_approximate_softmax(scores))
            attended=kernel.mma(probabilities,values).permute(0,2,1,3).reshape(windows.shape)
            output=kernel.mma(kernel.pack(attended),self.projection,windows*self.weight('attn_cos_skip'))
            return ref.reverse_windows(output,batch_count=strip.shape[0],height=strip.shape[1],width=strip.shape[2],window_size=8)
        height,width=residual.shape[1:3]
        top,left=-origin[0],-origin[1]
        bottom=(-(height+top))%8;right=(-(width+left))%8
        if top or left or bottom or right:residual=torch.nn.functional.pad(residual,(0,0,left,right,top,bottom))
        rows=max(8,(ref.CHUNK_TOKENS//(residual.shape[0]*residual.shape[2]*8))*8) if ref.CHUNK_TOKENS else residual.shape[1]
        output=torch.empty_like(residual)
        for start in range(0,residual.shape[1],rows):output[:,start:start+rows]=attend(residual[:,start:start+rows])
        return output[:,top:top+height,left:left+width]
