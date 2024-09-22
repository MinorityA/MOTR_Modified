# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
DETR model and criterion classes.
"""
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List

from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from .backbone import build_backbone
from .matcher import build_matcher
from .deformable_transformer_plus import build_deforamble_transformer
from .qim import build as build_query_interaction_layer
from .memory_bank import build_memory_bank
from .deformable_detr import SetCriterion, MLP
from .segmentation import sigmoid_focal_loss

from .sp_tracker import build_sptracker_decoder


class ClipMatcher(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        frame_id = self._current_frame_idx - 1
        gt_instances = self.gt_instances[frame_id]
        outputs = {
            'pred_logits': track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss('labels',
                                     outputs=outputs,
                                     gt_instances=[gt_instances],
                                     indices=[(src_idx, tgt_idx)],
                                     num_boxes=1)
        self.losses_dict.update(
            {'frame_{}_track_{}'.format(frame_id, key): value for key, value in
             track_losses.items()})

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(self, outputs, gt_instances: List[Instances], indices: List[tuple], num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        #TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_labels(self, outputs, gt_instances: List[Instances], indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J)
            # set labels of track-appear slots to 0.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o
        if self.focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = sigmoid_focal_loss(src_logits.flatten(1),
                                             gt_labels_target.flatten(1),
                                             alpha=0.25,
                                             gamma=2,
                                             num_boxes=num_boxes, mean_in_dim1=False)
            loss_ce = loss_ce.sum()
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses
    
    def match_for_single_detection(self, outputs: dict, aux_outputs):
        
        gt_instance_i = self.gt_instances[self._current_frame_idx]
        indices = self.matcher(outputs, [gt_instance_i])
        indices = [(t1.to(outputs['pred_logits'].device), t2.to(outputs['pred_logits'].device)) for (t1, t2) in indices]

        # calcualte losses
        self.num_samples += len(gt_instance_i)
        for loss in self.losses:
            detection_loss = self.get_loss(loss, outputs, [gt_instance_i], indices, num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in detection_loss.items()})
            
        if aux_outputs is not None:
            for i, aux_output in enumerate(aux_outputs):
                layer_outputs = {
                    'pred_logits': aux_output['pred_logits'][0].unsqueeze(0),
                    'pred_boxes': aux_output['pred_boxes'][0].unsqueeze(0),
                } 
                layer_matched_indices = self.matcher(layer_outputs, [gt_instance_i])
                layer_matched_indices = [(t1.to(outputs['pred_logits'].device), t2.to(outputs['pred_logits'].device)) for (t1, t2) in layer_matched_indices]

                for loss in self.losses:
                    l_dict = self.get_loss(loss, layer_outputs, [gt_instance_i], layer_matched_indices, num_boxes=1)
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in l_dict.items()})
        
        obj_idxes = gt_instance_i.obj_ids
        obj_indices = obj_idxes[indices[0][1]]
        indices = [(indices[0][0], obj_indices)]
        return indices
    
    def match_for_single_tracking(self, frame_res, detect_instances, tracked_instances):
        # match for first frame
        gt_instance_i = self.gt_instances[self._current_frame_idx]
        gt_obj_ids = gt_instance_i.obj_ids
        
        outputs = {
            'pred_logits': frame_res['pred_logits'][0].unsqueeze(0),
            'pred_boxes': frame_res['pred_boxes'][0].unsqueeze(0),
        }
        # the key here is how to operate the indices...
        if tracked_instances is None:
            # in this case, the number of gts would be same with the number of detect_queries
            query_indices = torch.arange(0, len(gt_obj_ids), device=frame_res['pred_logits'].device)
            detect_obj_ids = detect_instances.obj_idxes
            ids_in_gt = torch.tensor([torch.nonzero(gt_obj_ids == id_val, as_tuple=True)[0].item() for id_val in detect_obj_ids], dtype=torch.int64, device=frame_res['pred_logits'].device)
            indices = [(query_indices, ids_in_gt)]

            tracked_instances = detect_instances
        else:
            detect_obj_ids = detect_instances.obj_idxes
            tracked_obj_ids = tracked_instances.obj_idxes
            # get the idxes of detect_instances and tracked_instances
            dq_indices = torch.arange(0, len(detect_obj_ids), device=frame_res['pred_logits'].device)
            tq_indices = torch.arange(len(detect_obj_ids), len(detect_obj_ids)+len(tracked_obj_ids), device=frame_res['pred_logits'].device)
            # remove the duplicate ones in detect_instances
            #   find which detect_obj_ids exist in tracked_obj_ids
            id_mask = torch.isin(detect_obj_ids, tracked_obj_ids)
            non_duplicate_mask = ~id_mask
            #   filter the duplicate ones
            filtered_dq_indices = dq_indices[non_duplicate_mask]
            #   concat the obj_ids and query_indices
            obj_ids = torch.cat((detect_obj_ids, tracked_obj_ids))
            query_indices = torch.cat((filtered_dq_indices, tq_indices))
            obj_ids = obj_ids[query_indices]
            # find the objects that disappear in tracked_instances and match query_indices with gt_ids 
            gt_mask = torch.isin(obj_ids, gt_obj_ids)
            ids_in_gt = torch.tensor([torch.nonzero(gt_obj_ids == id_val, as_tuple=True)[0].item() for id_val in obj_ids[gt_mask]], dtype=torch.int64, device=frame_res['pred_logits'].device)
            indices = [(query_indices[gt_mask], ids_in_gt)]
            
            # update tracked_instances
            #   for track_instances, update the output_embed and query_pos of existing queries
            tq_update_indices = query_indices[torch.logical_and(gt_mask, query_indices >= len(detect_obj_ids))]
            tracked_instances.output_embedding[tq_update_indices - len(detect_obj_ids)] = frame_res['hs'][0][tq_update_indices].detach()
            tracked_instances.ref_pts[tq_update_indices - len(detect_obj_ids)] = frame_res['ref_pts_all'][-1, 0][tq_update_indices].detach()
            #   for detect_instances, only concat the non-duplicate one
            dq_update_indices = query_indices[torch.logical_and(gt_mask, query_indices < len(detect_obj_ids))]
            detect_instances = detect_instances[dq_update_indices]
            tracked_instances = Instances.cat((detect_instances, tracked_instances))

        # calculate losses
        # the only different is the indices..
        for loss in self.losses:
            track_loss = self.get_loss(loss, outputs, [gt_instance_i], indices, num_boxes=1)
            self.losses_dict.update(
            {'frame_track_{}_{}'.format(self._current_frame_idx, key): value for key, value in track_loss.items()})

        if frame_res['aux_outputs'] is not None:
            for i, aux_output in enumerate(frame_res['aux_outputs']):
                layer_outputs = {
                    'pred_logits': aux_output['pred_logits'][0].unsqueeze(0),
                    'pred_boxes': aux_output['pred_boxes'][0].unsqueeze(0),
                } 
                for loss in self.losses:
                    l_dict = self.get_loss(loss, layer_outputs, [gt_instance_i], indices, num_boxes=1)
                    self.losses_dict.update(
                        {'frame_track_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in l_dict.items()})
            
        self._step()
        
        return tracked_instances
    
    def match_for_single_frame(self, outputs: dict):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux['track_instances']
        pred_logits_i = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_i = track_instances.pred_boxes  # predicted boxes of i-th image.

        obj_idxes = gt_instances_i.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}
        outputs_i = {
            'pred_logits': pred_logits_i.unsqueeze(0),
            'pred_boxes': pred_boxes_i.unsqueeze(0),
        }

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        for j in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[j].item()
            # set new target idx.
            if obj_id >= 0:
                if obj_id in obj_idx_to_gt_idx:
                    track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[obj_id]
                else:
                    num_disappear_track += 1
                    track_instances.matched_gt_idxes[j] = -1  # track-disappear case.
            else:
                track_instances.matched_gt_idxes[j] = -1

        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(pred_logits_i.device)
        matched_track_idxes = (track_instances.obj_idxes >= 0)  # occu 
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes], track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(
            pred_logits_i.device)

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances_i)).to(pred_logits_i.device)
        tgt_state[tgt_indexes] = 1
        untracked_tgt_indexes = torch.arange(len(gt_instances_i)).to(pred_logits_i.device)[tgt_state == 0]
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]

        def match_for_single_decoder_layer(unmatched_outputs, matcher):
            new_track_indices = matcher(unmatched_outputs,
                                             [untracked_gt_instances])  # list[tuple(src_idx, tgt_idx)]

            src_idx = new_track_indices[0][0]
            tgt_idx = new_track_indices[0][1]
            # concat src and tgt.
            new_matched_indices = torch.stack([unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]],
                                              dim=1).to(pred_logits_i.device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
        }
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances_i.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_i,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_i],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})
        self._step()
        return track_instances

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses


class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.7, filter_score_thresh=0.6, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] == -1 and track_instances.scores[i] >= self.score_thresh:
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif track_instances.obj_idxes[i] >= 0 and track_instances.scores[i] < self.filter_score_thresh:
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # Set the obj_id to -1.
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        prob = out_logits.sigmoid()
        # prob = out_logits[...,:1].sigmoid()
        scores, labels = prob.max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_size
        scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
        boxes = boxes * scale_fct[None, :]

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = labels
        track_instances.remove('pred_logits')
        track_instances.remove('pred_boxes')
        return track_instances


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MOTR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, criterion, track_embed,
                 aux_loss=True, with_box_refine=False, two_stage=False, memory_bank=None, use_checkpoint=False,
                 use_dab=True, random_refpoints_xy=False, tracker_decoder=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR

            --modified part:
            tracker_decoder: the decoder performing tracking
        """
        super().__init__()
        self.num_queries = num_queries
        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint

        self.use_dab = use_dab
        self.random_refpoints_xy = random_refpoints_xy

        if not two_stage:
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
            else:
                self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries, 4)
                if random_refpoints_xy:
                    self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
                    self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                    self.refpoint_embed.weight.data[:, :2].requires_grad = False
                tgt_embed = self.tgt_embed.weight       # nq, 256
                refanchor = self.refpoint_embed.weight  # nq, 4
                self.query_embed = torch.cat((tgt_embed, refanchor), dim=1) 

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
            # self.track_embed.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.post_process = TrackerPostProcess()
        self.track_base = RuntimeTrackerBase()
        self.criterion = criterion
        self.memory_bank = memory_bank
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length
        
        # for modify the last n layers to perform track
        self.tracker_decoder = tracker_decoder
        self.feat_dropout = nn.Dropout(0.1)

        self.max_obj_id = 0

    def _generate_empty_tracks(self):
        tgt_embed = self.tgt_embed.weight       # nq, 256
        refanchor = self.refpoint_embed.weight  # nq, 4
        query_embed = torch.cat((tgt_embed, refanchor), dim=1) 
        
        track_instances = Instances((1, 1))
        num_queries, dim = query_embed.shape  # (300, 260) 260 = 256+4
        device = query_embed.device
        track_instances.ref_pts = query_embed[:, dim-4:]
        track_instances.query_pos = query_embed
        track_instances.output_embedding = torch.zeros((num_queries, dim-4), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
        track_instances.iou = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)

        # mem_bank_len = self.mem_bank_len
        # track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, dim // 2), dtype=torch.float32, device=device)
        # track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        # track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device)

        return track_instances.to(device)

    def clear(self):
        self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, }
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _forward_single_image(self, samples):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.use_dab:
            tgt_embed = self.tgt_embed.weight
            refanchor = self.refpoint_embed.weight
            query_embeds = torch.cat((tgt_embed, refanchor), dim=1)

        # hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, query_pos = self.transformer(srcs, masks, pos, track_instances.query_pos, ref_pts=track_instances.ref_pts)
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, query_pos = self.transformer(srcs, masks, pos, query_embeds)

        # handle the selective outputs from transformer
        index_hs = [0, 1, 3, 6]
        hs_list = []
        inter_references_list = []

        init_reference_expand = init_reference.unsqueeze(0).repeat(src.shape[0], 1, 1)
        inter_references_list.append(init_reference_expand)

        for i in range(len(index_hs) - 1):
            hs_list.append(torch.flatten(hs[index_hs[i]:index_hs[i+1]].permute(1,0,2,3), 1, 2))
            inter_references_list.append(torch.flatten(inter_references[index_hs[i]:index_hs[i+1]].permute(1,0,2,3), 1, 2))

        outputs_classes = []
        outputs_coords = []
        for lvl in range(len(hs_list)):
            if lvl == 0:
                reference = init_reference
            else:
                reference = torch.cat((inter_references_list[lvl], inter_references_list[lvl-1]), dim=1)
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs_list[lvl])
            tmp = self.bbox_embed[lvl](hs_list[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        # outputs_class = torch.stack(outputs_classes)
        # outputs_coord = torch.stack(outputs_coords)

        # and in sqr_detr, the query_pos shape also need to be modified
        query_pos = torch.flatten(query_pos.view(src.shape[0], query_pos.shape[0]//src.shape[0], query_pos.shape[1], query_pos.shape[2]), 1, 2)
        
        # ref_pts_all = torch.cat([init_reference[None, None, :, :2], inter_references[:, :, :, :2]], dim=0)
        # need to change the inter_references here for selective queries..
        out = {'pred_logits': outputs_classes[-1], 'pred_boxes': outputs_coords[-1], 'ref_pts': inter_references_list[-1][:, :, :4]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_classes, outputs_coords)
        out['hs'] = hs_list[-1]
        out['query_pos'] = query_pos[-1]
        out['src'] = srcs
        return out
    
    def _post_process_single_image(self, frame_res, track_instances, is_last):
        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = frame_res['pred_logits'][0, :, 0].sigmoid()

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        track_instances.output_embedding = frame_res['hs'][0]
        track_instances.ref_pts = frame_res['query_pos']
        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            # each track will be assigned an unique global id by the track base.
            self.track_base.update(track_instances)
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
            # track_instances.track_scores = track_instances.track_scores[..., 0]
            # track_instances.scores = track_instances.track_scores.sigmoid()
            if self.training:
                self.criterion.calc_loss_for_track_scores(track_instances)
        tmp = {}
        tmp['init_track_instances'] = self._generate_empty_tracks()
        tmp['track_instances'] = track_instances
        if not is_last:
            out_track_instances = self.track_embed(tmp)
            frame_res['track_instances'] = out_track_instances
        else:
            frame_res['track_instances'] = None
        return frame_res
    
    def _post_process_detection(self, frame_res):
        outputs = {
            'pred_logits': frame_res['pred_logits'][0].unsqueeze(0),
            'pred_boxes': frame_res['pred_boxes'][0].unsqueeze(0),
        }
        aux_outputs = frame_res['aux_outputs']
        hs = frame_res['hs'][0]
        query_pos = frame_res['ref_pts'][0]
        if self.training:
            indices = self.criterion.match_for_single_detection(outputs, aux_outputs)
            query_idx, tgt_idx = indices[0]
            obj_detected_queries = hs[query_idx]
            obj_detected_query_pos = query_pos[query_idx] # the dimension is weird now, need to be fixed...
            
        else:
            # need to select queries that find objects
            detect_scores = outputs['pred_logits'][0, :, 0].sigmoid()
            indices = torch.nonzero(detect_scores > 0.5, as_tuple=True)[0]
            obj_detected_queries = hs[indices]
            obj_detected_query_pos = query_pos[indices]
            tgt_idx = None

            
        noisy_queries, noisy_query_pos = self._add_noises_to_obj(obj_detected_queries.detach(), obj_detected_query_pos.detach())
        detect_instances = self._init_detect_instances(noisy_queries, noisy_query_pos, tgt_idx)

        return detect_instances

    def _init_detect_instances(self, noisy_queries, noisy_query_pos, obj_ids):

        detect_instances = Instances((1,1))
        detect_instances.output_embedding = noisy_queries
        detect_instances.ref_pts = noisy_query_pos
        detect_instances.obj_idxes = obj_ids
        
        return detect_instances
        

    def _add_noises_to_obj(self, obj_queries, obj_query_pos):
        # feat_noise_sclae = 0.1, set in the self.feat_dropout
        noisy_queries = self.feat_dropout(obj_queries)

        pos_noise_scale = 0.01
        pos_noise = obj_query_pos * pos_noise_scale * torch.randn_like(obj_query_pos)
        noisy_pos = obj_query_pos + pos_noise
        noisy_pos = torch.clamp(noisy_pos, 1e-3, 1-1e-3)

        return noisy_queries, noisy_pos
    
    def _post_process_tracking(self, frame_res, detect_instances, tracked_instances):

        if self.training:
            tracked_instances = self.criterion.match_for_single_tracking(frame_res, detect_instances, tracked_instances)
        else:
            # check the outputs to find the ones with predicted_cls >= 0
            # select ones with high confidence and then send them into track_base
            track_scores = frame_res['pred_logits'][0, :, 0].sigmoid()
            indices = torch.nonzero(track_scores > 0.7, as_tuple=True)[0]
            # for selected detect_instances.. assign id to them (also for the case that no track_instances.. just )
            detect_indices = indices[indices < len(detect_instances.obj_idxes)]
            for i in range(len(detect_indices)):
                detect_indices.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            detect_instances = detect_instances[detect_indices]
            # for track_instances.. just keep them..
            tracked_instances = Instances.cat((detect_indices, tracked_instances))

        return tracked_instances


    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, track_instances=None):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks()
        res = self._forward_single_image(img,
                                         track_instances=track_instances)
        res = self._post_process_single_image(res, track_instances, False)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        ret = {'track_instances': track_instances}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def forward(self, data: dict):
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        frames = data['imgs']  # list of Tensor.
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }

        track_instances = self._generate_empty_tracks()
        keys = list(track_instances._fields.keys())
        memory_bank = []
        tracked_instances = None
        for frame_index, frame in enumerate(frames):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1
            if self.use_checkpoint and frame_index < len(frames) - 2:
                def fn(frame, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    frame_res = self._forward_single_image(frame)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['ref_pts'],
                        frame_res['hs'],
                        frame_res['query_pos'],
                        *[src for src in frame_res['src']],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']]
                    )

                args = [frame]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'ref_pts': tmp[2],
                    'hs': tmp[3],
                    'query_pos': tmp[4],
                    'src': [tmp[5+i] for i in range(4)],
                    'aux_outputs': [{
                        'pred_logits': tmp[9+i],
                        'pred_boxes': tmp[9+2+i],
                    } for i in range(2)],
                }
            else:
                frame = nested_tensor_from_tensor_list([frame])
                # step 1, forward single image to get only detection results(including objects in previous frame)
                frame_res = self._forward_single_image(frame)
            # step 2, add srcs into memory bank for tracker
            if len(memory_bank) >= 4:
                memory_bank.pop(0) # remove the earliest frame
            memory_bank.append(frame_res['src']) # add new frame into memory bank
            # step 3, process detection results to get noisy_instances for tracking
            detect_instances = self._post_process_detection(frame_res)
            # step 4, send detect_instances and tracked_instances into  
            tracker_outputs = self.tracker_decoder(memory_bank, detect_instances, tracked_instances)
            tracked_instances = self._post_process_tracking(tracker_outputs, detect_instances, tracked_instances)


            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])

        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs
    
def build(args):
    dataset_to_num_classes = {
        'coco': 91,
        'coco_panoptic': 250,
        'e2e_mot': 1,
        'e2e_dance': 1,
        'e2e_joint': 1,
        'e2e_static_mot': 1,
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    query_interaction_layer = build_query_interaction_layer(args, args.query_interaction_layer, d_model, hidden_dim, d_model*2)

    tracker_decoder = build_sptracker_decoder(args, num_classes)

    img_matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            "frame_track_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_track_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_track_{}_loss_giou'.format(i): args.giou_loss_coef,
                            })

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    "frame_track_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_track_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_track_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })
    if args.memory_bank_type is not None and len(args.memory_bank_type) > 0:
        memory_bank = build_memory_bank(args, d_model, hidden_dim, d_model * 2)
        for i in range(num_frames_per_batch):
            weight_dict.update({"frame_{}_track_loss_ce".format(i): args.cls_loss_coef})
    else:
        memory_bank = None
    losses = ['labels', 'boxes']
    criterion = ClipMatcher(num_classes, matcher=img_matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(device)
    postprocessors = {}
    model = MOTR(
        backbone,
        transformer,
        track_embed=None,
        tracker_decoder=tracker_decoder,
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        memory_bank=memory_bank,
        use_checkpoint=args.use_checkpoint,
        use_dab=True,
        random_refpoints_xy=args.random_refpoints_xy
    )
    return model, criterion, postprocessors
