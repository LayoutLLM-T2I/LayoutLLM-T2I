import torch as th
import torch.nn as nn
import random
import torch.nn.functional as F
from copy import deepcopy
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.attention import CrossAttention, FeedForward
from einops import rearrange, repeat
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class GligenCombineLayout(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            num_heads=8,
            use_scale_shift_norm=False,
            transformer_depth=1,
            context_dim=None,
            fuser_type=None,
            inpaint_mode=False,
            grounding_downsampler=None,
            grounding_tokenizer=None,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.context_dim = context_dim
        self.fuser_type = fuser_type
        self.inpaint_mode = inpaint_mode
        assert fuser_type in ["gatedSA", "gatedSA2", "gatedCA"]

        self.gligen = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            use_scale_shift_norm=use_scale_shift_norm,
            transformer_depth=transformer_depth,
            context_dim=context_dim,
            fuser_type=fuser_type,
            inpaint_mode=inpaint_mode,
            grounding_downsampler=grounding_downsampler,
            grounding_tokenizer=grounding_tokenizer,
        )

        self.rela_fuse = RelationCrossAttention(model_channels, context_dim, context_dim, num_heads, num_heads)

    def forward(self, input):

        if ("grounding_input" in input):
            grounding_input = input["grounding_input"]
        else:
            # Guidance null case
            grounding_input = self.gligen.grounding_tokenizer_input.get_null_input()

        if self.gligen.training and random.random() < 0.1 and self.gligen.grounding_tokenizer_input.set:  # random drop for guidance
            grounding_input = self.gligen.grounding_tokenizer_input.get_null_input()

        # Grounding tokens: B*N*C
        objs = self.gligen.position_net(**grounding_input)

        # Time embedding
        t_emb = timestep_embedding(input["timesteps"], self.model_channels, repeat_only=False)
        emb = self.gligen.time_embed(t_emb)

        # input tensor
        h = input["x"]
        if self.gligen.downsample_net != None and self.first_conv_type == "GLIGEN":
            temp = self.gligen.downsample_net(input["grounding_extra_input"])
            h = th.cat([h, temp], dim=1)
        if self.gligen.inpaint_mode:
            if self.gligen.downsample_net != None:
                breakpoint()  # TODO: think about this case
            h = th.cat([h, input["inpainting_extra_input"]], dim=1)

        # Text input
        context = input["context"]

        # Start forwarding
        hs = []
        for module in self.gligen.input_blocks:
            h = module(h, emb, context, objs)
            hs.append(h)

        h = self.gligen.middle_block(h, emb, context, objs)

        for module in self.gligen.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, objs)

        b, c, h, w = input["x"].shape
        _tmp = self.rela_fuse(rearrange(input["x"], 'b c h w -> b (h w) c'), input["relations"],
                              input["grounding_input"]["boxes"], input["grounding_input"]["masks"])
        return self.gligen.out(h)


        # x = self.gligen(input)
        # return x


class RelationCrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head, h=64, w=64):
        super().__init__()

        self.w = w
        self.h = h

        self.attn = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads,
                                   dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(th.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(th.tensor(0.)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one
        self.scale = 1

    def forward(self, x, relations, boxes, masks):
        b, a, c = x.shape
        h, w = self.w, self.h
        hidden = rearrange(x, 'b a c -> b c h w')
        mo = boxes.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, mo, 1, 1, 1)
        obj_mask = th.full(hidden.size(), fill_value=0).to(h)

        box_x_1 = (boxes[:, :, 0] * self.w).to(th.int)
        box_y_1 = (boxes[:, :, 1] * self.h).to(th.int)
        box_x_2 = box_x_1 + (boxes[:, :, 2] * self.w).to(th.int)  # x + w
        box_y_2 = box_y_1 + (boxes[:, :, 3] * self.h).to(th.int)  # y + h

        obj_features = []
        for k in range(b):
            _tmp = []
            for i in range(mo):
                left = box_x_1[k][i]
                right = box_x_2[k][i]
                top = box_y_1[k][i]
                bottom = box_y_2[k][i]
                obj_mask[k, i, :, top:bottom, left:right] = 1
                _tmp.append(th.sum(rearrange(hidden[k, i, :, top:bottom, left:right], 'c h w -> c h*w'), dim=-1))
            obj_features.append(th.stack(_tmp))

        obj_features = obj_features + self.scale * th.tanh(self.alpha_attn) * self.attn(self.norm1(obj_features), relations, relations)
        # b, max_obj, 768
        obj_features = obj_features + self.scale * th.tanh(self.alpha_dense) * self.ff(self.norm2(obj_features))

        # b mo 768 h w
        obj_features = rearrange(obj_features.unsqueeze(3).repeat(1, 1, 1, self.w*self.h), "b mo c h*w -> b mo c h w")
        hidden = hidden + obj_mask * obj_features
        hidden = th.sum(hidden, dim=1)
        return hidden

