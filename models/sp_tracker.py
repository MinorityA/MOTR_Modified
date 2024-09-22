
import copy
import math
import torch

from models.deformable_transformer_plus import DeformableTransformerDecoderLayer
from torch import nn
import torch.nn.functional as F 

from .deformable_transformer_plus import MLP
from util.misc import inverse_sigmoid

from models.ops.modules import MSDeformAttn

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class SPTracker(nn.Module):
    def __init__(self, num_layers, d_ffn, dropout, activation, n_heads, n_levels, n_points, num_classes, d_model=256):
        super().__init__()
        layer = DeformableTransformerDecoderLayer(d_model, d_ffn, dropout, activation, n_levels, n_heads, n_points)
        self.layers = _get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model

        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.bbox_embed = _get_clones(self.bbox_embed, num_layers)

        self.class_embed = nn.Linear(d_model, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.class_embed = _get_clones(self.class_embed, num_layers)

        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.ref_point_head = MLP(2*d_model, d_model, d_model, 2)

        self.return_intermediate = True

    def forward(self, memory_bank, detect_instances, tracked_instances):
        
        # process memory_bank to get src, src_spatial_shapes, src_level_start_index, src_padding_mask
        num_memory = len(memory_bank)
        src_flatten = []
        spatial_shapes = []
        for _, memories in enumerate(zip(*memory_bank)):
            _, _, h, w = memories[0].shape
            flatten_memories = torch.cat([memory.flatten(2) for memory in memories], dim=2).transpose(1, 2)

            src_flatten.append(flatten_memories)
            spatial_shapes.append((h*num_memory, w))
        
        src = torch.cat(src_flatten, dim=1)
        src_spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src.device)
        src_level_start_index = torch.cat((src_spatial_shapes.new_zeros((1, )), src_spatial_shapes.prod(1).cumsum(0)[:-1]))
        src_padding_mask = torch.zeros(src.shape[:2], dtype=bool, device=src.device)

        d_output = detect_instances.output_embedding # shape [query_size, query_dim(256)] 
        d_ref_pts = detect_instances.ref_pts  # shape [query_size, 4(x,y)]

        bs = src.shape[0]
        if tracked_instances is None:
            output = d_output[None].repeat(bs, 1, 1)
            reference_points = d_ref_pts[None].repeat(bs, 1, 1)
            attn_mask = None
        else:
            t_output = tracked_instances.output_embedding
            t_ref_pts = tracked_instances.ref_pts

            output = torch.cat((d_output, t_output), dim=0)[None].repeat(bs, 1, 1)
            reference_points = torch.cat((d_ref_pts, t_ref_pts), dim=0)[None].repeat(bs, 1, 1)

            n_dq = d_output.shape[0]
            n_tq = t_output.shape[0]
            n_totalq = n_dq + n_tq
            attn_mask = torch.zeros(n_totalq, n_totalq).to(src.device)
            attn_mask[n_dq:, :n_dq] = float('-inf')

        init_reference = reference_points
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):

            assert reference_points.shape[-1] == 4
            reference_points_input = reference_points[:, :, None].repeat(1, 1, 4, 1)

            # same as dab_detr decoder part
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(output) if lid != 0 else 1
            query_pos = pos_scale * raw_query_pos


            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, attn_mask)

            # update reference_points (box refinement)
            tmp = self.bbox_embed[lid](output)
            new_reference_points = tmp + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points.detach()

            # append new intermediate
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        if self.return_intermediate:
            hs, inter_references = torch.stack(intermediate), torch.stack(intermediate_reference_points)
        
        # post processing part for outputs of decoder
        output_classes = []
        output_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            output_class = self.class_embed[lvl](hs.clone()[lvl])
            tmp_boxes = self.bbox_embed[lvl](hs.clone()[lvl])
            tmp_boxes += reference

            output_coord = tmp_boxes.sigmoid()
            output_classes.append(output_class)
            output_coords.append(output_coord)
        output_class = torch.stack(output_classes)
        output_coord = torch.stack(output_coords)

        ref_pts_all = torch.cat([init_reference[None], inter_references], dim=0)
        out = {'pred_logits': output_class[-1], 'pred_boxes': output_coord[-1], 'ref_pts_all': ref_pts_all, 'hs': hs[-1]}
        out['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b,} for a, b in zip(output_class[:-1], output_coord[:-1])]
        
        # we only update the detect_instances output_embed and query_pos here because track_instances may disappear
        # and in that case, we should not update the output_embed and ref_pts..
        dq_size = d_output.shape[0]
        detect_instances.output_embedding = hs[-1][0][:dq_size].detach()
        detect_instances.ref_pts = inter_references[-1][0][:dq_size].detach()

        return out


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

def build_sptracker_decoder(args, num_classes):
    return SPTracker(
        d_model=args.hidden_dim,
        num_layers=3,
        d_ffn=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        n_heads=args.nheads,
        n_levels=args.num_feature_levels,
        n_points=args.dec_n_points,
        num_classes=num_classes
    )