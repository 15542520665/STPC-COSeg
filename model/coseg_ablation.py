"""
COSeg Ablation Study Model.
支持独立开启/关闭每个优化模块，用于消融实验。

用法示例:
    model = COSegAblation(args, ablation_config={
        'use_cross_attention': True,      # 交叉注意力原型精炼
        'use_contrastive_loss': True,     # 原型对比学习
        'use_dice_loss': True,            # Dice损失
        'use_lovasz_loss': True,          # Lovász损失
        'use_enhanced_aggregator': True,  # 增强聚合层
        'use_pos_encoding': True,         # 3D位置编码
        'use_consistency_loss': True,     # 特征一致性损失
    })
"""

from typing import Optional, Tuple, Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.stratified_transformer import Stratified
from model.common import MLPWithoutResidual, KPConvResBlock, AggregatorLayer

# 增强模块
from model.enhanced_modules import (
    PrototypeContrastiveLoss,
    CrossAttentionPrototypeRefinement,
    EnhancedAggregatorLayer,
    PositionalEncoding3D,
    FeatureConsistencyLoss,
    QueryGuidedPrototypeRefinement,
)

# 损失函数
from model.loss_functions import (
    DiceLoss,
    LovaszSoftmaxLoss,
    FocalLoss,
    BoundaryLoss,
    LabelSmoothingLoss,
    SupportQueryConsistencyLoss,
)

import torch_points_kernels as tp
from util.logger import get_logger
from lib.pointops2.functions import pointops


# 默认消融配置（全部关闭 = 原始模型）
DEFAULT_ABLATION_CONFIG = {
    'use_cross_attention': False,      # 交叉注意力原型精炼
    'use_contrastive_loss': False,     # 原型对比学习
    'use_dice_loss': False,            # Dice损失
    'use_lovasz_loss': False,          # Lovász损失
    'use_enhanced_aggregator': False,  # 增强聚合层
    'use_pos_encoding': False,         # 3D位置编码
    'use_consistency_loss': False,     # 特征一致性损失
    # 新增优化模块
    'use_focal_loss': False,           # Focal损失（关注难分样本）
    'use_boundary_loss': False,        # 边界感知损失
    'use_label_smoothing': False,      # 标签平滑
    # 结构创新模块
    'use_multiscale_correlation': False,  # 温度多尺度相关性
    # 数据增强模块
    'use_enhanced_augmentation': False,   # 增强数据增强
    # 原型细化模块
    'use_prototype_refinement': False,    # 查询引导原型细化
    # 一致性约束
    'use_sq_consistency': False,          # 支持集-查询集一致性约束
    # 正则化方案
    'use_proto_noise': False,             # 原型噪声增强
    'use_feat_dropout': False,            # 特征Dropout正则化
}


class COSegAblation(nn.Module):
    """
    支持消融实验的COSeg模型。
    可以通过ablation_config独立开启/关闭每个优化模块。
    """
    def __init__(self, args, ablation_config: Dict[str, bool] = None):
        super(COSegAblation, self).__init__()
        
        # 合并配置
        self.ablation_config = DEFAULT_ABLATION_CONFIG.copy()
        if ablation_config:
            self.ablation_config.update(ablation_config)
        
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_subprototypes = args.n_subprototypes
        self.n_queries = args.n_queries
        self.n_classes = self.n_way + 1
        self.args = args
        
        # ========= 基础损失 =========
        self.criterion_ce = nn.CrossEntropyLoss(
            weight=torch.tensor([0.1] + [1 for _ in range(self.n_way)]),
            ignore_index=args.ignore_label,
        )
        self.criterion_base = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        
        # ========= 可选损失模块 =========
        if self.ablation_config['use_dice_loss']:
            self.dice_loss = DiceLoss(ignore_index=args.ignore_label)
            self.dice_weight = float(getattr(args, 'dice_weight', 0.5))
        
        if self.ablation_config['use_lovasz_loss']:
            self.lovasz_loss = LovaszSoftmaxLoss(ignore_index=args.ignore_label)
            self.lovasz_weight = 0.3
        
        if self.ablation_config['use_contrastive_loss']:
            self.contrastive_loss = PrototypeContrastiveLoss(temperature=0.07)
            self.contrastive_weight = 0.1
        
        if self.ablation_config['use_consistency_loss']:
            self.consistency_loss = FeatureConsistencyLoss(temperature=0.1)
            self.consistency_weight = 0.05
        
        # 新增损失模块
        if self.ablation_config.get('use_focal_loss', False):
            self.focal_loss = FocalLoss(gamma=2.0, ignore_index=args.ignore_label)
            self.focal_weight = 0.5
        
        if self.ablation_config.get('use_boundary_loss', False):
            self.boundary_loss = BoundaryLoss(ignore_index=args.ignore_label, theta=0.5)
            self.boundary_weight = 0.3
        
        if self.ablation_config.get('use_label_smoothing', False):
            self.label_smoothing_loss = LabelSmoothingLoss(smoothing=0.1, ignore_index=args.ignore_label)
            self.label_smoothing_weight = 0.5
        
        # 原型细化模块
        if self.ablation_config.get('use_prototype_refinement', False):
            self.prototype_refiner = QueryGuidedPrototypeRefinement(feat_dim=args.channels[2])
        
        # 支持集-查询集一致性约束
        if self.ablation_config.get('use_sq_consistency', False):
            self.sq_consistency_loss = SupportQueryConsistencyLoss()
            self.sq_consistency_weight = 0.1
        
        # 原型噪声增强参数
        default_proto_noise_std = 0.1
        self.proto_noise_std = (
            float(getattr(args, 'proto_noise_std', default_proto_noise_std))
            if self.ablation_config.get('use_proto_noise', False)
            else 0.0
        )
        
        # 特征Dropout
        default_feat_dropout_p = 0.2
        if self.ablation_config.get('use_feat_dropout', False):
            feat_dropout_p = float(getattr(args, 'feat_dropout_p', default_feat_dropout_p))
            self.feat_dropout = nn.Dropout(p=feat_dropout_p)

        # 配置
        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [
            args.patch_size * args.window_size * (2**i)
            for i in range(args.num_layers)
        ]
        args.grid_sizes = [
            args.patch_size * (2**i) for i in range(args.num_layers)
        ]
        args.quant_sizes = [
            args.quant_size * (2**i) for i in range(args.num_layers)
        ]

        if args.data_name == "s3dis":
            self.base_classes = 6
            if args.cvfold == 1:
                self.base_class_to_pred_label = {
                    0: 1, 3: 2, 4: 3, 8: 4, 10: 5, 11: 6,
                }
            else:
                self.base_class_to_pred_label = {
                    1: 1, 2: 2, 5: 3, 6: 4, 7: 5, 9: 6,
                }
        else:
            self.base_classes = 10
            if args.cvfold == 1:
                self.base_class_to_pred_label = {
                    2: 1, 3: 2, 5: 3, 6: 4, 7: 5,
                    10: 6, 12: 7, 13: 8, 14: 9, 19: 10,
                }
            else:
                self.base_class_to_pred_label = {
                    1: 1, 4: 2, 8: 3, 9: 4, 11: 5,
                    15: 6, 16: 7, 17: 8, 18: 9, 20: 10,
                }

        if self.main_process():
            self.logger = get_logger(args.save_path)
            self.logger.info(f"Ablation Config: {self.ablation_config}")

        # Backbone
        self.encoder = Stratified(
            args.downsample_scale,
            args.depths,
            args.channels,
            args.num_heads,
            args.window_size,
            args.up_k,
            args.grid_sizes,
            args.quant_sizes,
            rel_query=args.rel_query,
            rel_key=args.rel_key,
            rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz,
            num_classes=self.args.classes // 2 + 1,
            ratio=args.ratio,
            k=args.k,
            prev_grid_size=args.grid_size,
            sigma=1.0,
            num_layers=args.num_layers,
            stem_transformer=args.stem_transformer,
            backbone=True,
            logger=get_logger(args.save_path),
        )

        self.feat_dim = args.channels[2]

        # ========= 可选增强模块 =========
        
        # 交叉注意力原型精炼
        if self.ablation_config['use_cross_attention']:
            self.prototype_refinement = CrossAttentionPrototypeRefinement(
                self.feat_dim, num_heads=4, dropout=0.1
            )
        
        # 3D位置编码
        if self.ablation_config['use_pos_encoding']:
            self.pos_encoding = PositionalEncoding3D(self.feat_dim)
        
        # 温度多尺度相关性（无额外参数）
        if self.ablation_config.get('use_multiscale_correlation', False):
            # 温度参数：控制相似度的锐利程度
            self.temperatures = [0.5, 1.0, 2.0]  # 锐利、正常、平滑
            # 可学习的尺度融合权重
            self.scale_weights = nn.Parameter(torch.ones(3) / 3)

        # ========= 核心模块 =========
        
        self.lin1 = nn.Sequential(
            nn.Linear(self.n_subprototypes, self.feat_dim),
            nn.ReLU(inplace=True),
        )

        self.kpconv = KPConvResBlock(
            self.feat_dim, self.feat_dim, 0.04, sigma=2
        )

        self.cls = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(self.feat_dim, self.n_classes),
        )

        self.bk_ffn = nn.Sequential(
            nn.Linear(self.feat_dim + self.feat_dim // 2, 4 * self.feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * self.feat_dim, self.feat_dim),
        )

        # 聚合层（可选增强版）
        agglayers = 2 if self.args.data_name == "s3dis" else 4
        
        if self.ablation_config['use_enhanced_aggregator']:
            self.agglayers = nn.ModuleList([
                EnhancedAggregatorLayer(
                    hidden_dim=self.feat_dim,
                    num_heads=4,
                    dropout=0.1,
                )
                for _ in range(agglayers)
            ])
        else:
            self.agglayers = nn.ModuleList([
                AggregatorLayer(
                    hidden_dim=self.feat_dim,
                    guidance_dim=0,
                    nheads=4,
                    attention_type="linear",
                )
                for _ in range(agglayers)
            ])

        if self.n_way == 1:
            self.class_reduce = nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Conv1d(self.n_classes, 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.class_reduce = MLPWithoutResidual(
                self.feat_dim * (self.n_way + 1), self.feat_dim
            )

        self.bg_proto_reduce = MLPWithoutResidual(
            self.n_subprototypes * self.n_way, self.n_subprototypes
        )

        self.init_weights()

        self.register_buffer(
            "base_prototypes", torch.zeros(self.base_classes, self.feat_dim)
        )

    def init_weights(self):
        for name, m in self.named_parameters():
            if "class_attention.base_merge" in name:
                continue
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def main_process(self):
        return not self.args.multiprocessing_distributed or (
            self.args.multiprocessing_distributed
            and self.args.rank % self.args.ngpus_per_node == 0
        )

    def forward(
        self,
        support_offset: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_offset: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        epoch: int,
        support_base_y: Optional[torch.Tensor] = None,
        query_base_y: Optional[torch.Tensor] = None,
        sampled_classes: Optional[np.array] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播，支持消融配置"""
        
        # 获取support特征
        (
            support_feat,
            support_x_low,
            support_offset_low,
            support_y_low,
            _,
            support_base_y,
        ) = self.getFeatures(support_x, support_offset, support_y, support_base_y)
        
        support_offset_low = support_offset_low[:-1].long().cpu()
        support_feat = torch.tensor_split(support_feat, support_offset_low)
        support_x_low = torch.tensor_split(support_x_low, support_offset_low)
        if support_base_y is not None:
            support_base_y = torch.tensor_split(support_base_y, support_offset_low)

        # ========= 可选：特征Dropout（Support特征） =========
        if self.ablation_config.get('use_feat_dropout', False):
            support_feat = tuple(self.feat_dropout(sf) for sf in support_feat)

        # 获取原型
        fg_mask = support_y_low
        bg_mask = torch.logical_not(support_y_low)
        fg_mask = torch.tensor_split(fg_mask, support_offset_low)
        bg_mask = torch.tensor_split(bg_mask, support_offset_low)

        fg_prototypes = self.getPrototypes(
            support_x_low, support_feat, fg_mask,
            k=self.n_subprototypes // self.k_shot,
        )
        bg_prototype = self.getPrototypes(
            support_x_low, support_feat, bg_mask,
            k=self.n_subprototypes // self.k_shot,
        )

        if bg_prototype.shape[0] > self.n_subprototypes:
            bg_prototype = self.bg_proto_reduce(
                bg_prototype.permute(1, 0)
            ).permute(1, 0)

        # 获取query特征
        (
            query_feat,
            query_x_low,
            query_offset_low,
            query_y_low,
            q_base_pred,
            query_base_y,
        ) = self.getFeatures(query_x, query_offset, query_y, query_base_y)

        # ========= 可选：特征Dropout（Query特征） =========
        if self.ablation_config.get('use_feat_dropout', False):
            query_feat = self.feat_dropout(query_feat)

        # ========= 可选：交叉注意力原型精炼 =========
        if self.ablation_config['use_cross_attention']:
            fg_prototypes = self.prototype_refinement(fg_prototypes, query_feat)
            bg_prototype = self.prototype_refinement(bg_prototype, query_feat)
        
        # ========= 可选：查询引导原型细化 =========
        if self.ablation_config.get('use_prototype_refinement', False):
            fg_prototypes = self.prototype_refiner(fg_prototypes, query_feat)
            bg_prototype = self.prototype_refiner(bg_prototype, query_feat)
        
        # ========= 可选：原型噪声增强 =========
        if self.training and self.proto_noise_std > 0:
            fg_prototypes = fg_prototypes + torch.randn_like(fg_prototypes) * self.proto_noise_std
            bg_prototype = bg_prototype + torch.randn_like(bg_prototype) * self.proto_noise_std
        
        sparse_embeddings = torch.cat([bg_prototype, fg_prototypes])

        # 分割query特征
        query_feat_all = query_feat  # 保存用于对比损失
        query_offset_low_cpu = query_offset_low[:-1].long().cpu()
        query_feat = torch.tensor_split(query_feat, query_offset_low_cpu)
        query_x_low_list = torch.tensor_split(query_x_low, query_offset_low_cpu)
        if query_base_y is not None:
            query_base_y_list = torch.tensor_split(query_base_y, query_offset_low_cpu)

        # 更新base prototypes
        if self.training:
            for base_feat, base_y in zip(
                list(query_feat) + list(support_feat),
                list(query_base_y_list) + list(support_base_y),
            ):
                cur_baseclsses = base_y.unique()
                cur_baseclsses = cur_baseclsses[cur_baseclsses != 0]
                for class_label in cur_baseclsses:
                    class_mask = base_y == class_label
                    class_features = (
                        base_feat[class_mask].sum(dim=0) / class_mask.sum()
                    ).detach()
                    if torch.all(self.base_prototypes[class_label - 1] == 0):
                        self.base_prototypes[class_label - 1] = class_features
                    else:
                        self.base_prototypes[class_label - 1] = (
                            self.base_prototypes[class_label - 1] * 0.995
                            + class_features * 0.005
                        )
            mask_list = [
                self.base_class_to_pred_label[base_cls] - 1
                for base_cls in sampled_classes
            ]
            base_mask = self.base_prototypes.new_ones(
                (self.base_prototypes.shape[0]), dtype=torch.bool
            )
            base_mask[mask_list] = False
            base_avail_pts = self.base_prototypes[base_mask]
        else:
            base_avail_pts = self.base_prototypes

        query_pred = []
        for i, q_feat in enumerate(query_feat):
            if epoch < 1:
                base_guidance = None
            else:
                base_similarity = F.cosine_similarity(
                    q_feat[:, None, :],
                    base_avail_pts[None, :, :],
                    dim=2,
                )
                base_guidance = base_similarity.max(dim=1, keepdim=True)[0]

            # ========= 相关性计算 =========
            # 计算基础余弦相似度
            base_corr = F.cosine_similarity(
                q_feat[:, None, :],
                sparse_embeddings[None, :, :],
                dim=2,
            )  # [N_q, (N_way+1)*N_pt]
            
            if self.ablation_config.get('use_multiscale_correlation', False):
                # 温度多尺度：用不同温度缩放相似度
                # 温度低 = 锐利（强调高相似度）
                # 温度高 = 平滑（考虑更多弱匹配）
                weights = F.softmax(self.scale_weights, dim=0)
                
                correlations = torch.zeros_like(base_corr)
                for temp, w in zip(self.temperatures, weights):
                    # 温度缩放后的softmax加权
                    scaled_corr = base_corr / temp
                    correlations = correlations + w * scaled_corr
                
                correlations = correlations.view(correlations.shape[0], self.n_way + 1, -1)
            else:
                # 原始单尺度相关性
                correlations = base_corr.view(base_corr.shape[0], self.n_way + 1, -1)
            
            # 通过lin1转换
            correlations = (
                self.lin1(correlations)
                .permute(2, 1, 0)
                .unsqueeze(0)
            )

            for layer in self.agglayers:
                correlations = layer(correlations, base_guidance)

            correlations = correlations.squeeze(0).permute(2, 1, 0).contiguous()

            if self.n_way == 1:
                correlations = self.class_reduce(correlations).squeeze(1)
            else:
                correlations = self.class_reduce(
                    correlations.view(correlations.shape[0], -1)
                )

            coord = query_x_low_list[i]
            batch = torch.zeros(
                correlations.shape[0], dtype=torch.int64, device=coord.device
            )
            sigma = 2.0
            radius = 2.5 * self.args.grid_size * sigma
            neighbors = tp.ball_query(
                radius, self.args.max_num_neighbors,
                coord, coord,
                mode="partial_dense",
                batch_x=batch, batch_y=batch,
            )[0]
            correlations = self.kpconv(correlations, coord, batch, neighbors.clone())

            out = self.cls(correlations)
            query_pred.append(out)

        query_pred = torch.cat(query_pred)

        # ========= 计算损失 =========
        # Label Smoothing 替换 CE（而非叠加），避免过拟合
        if self.ablation_config.get('use_label_smoothing', False):
            loss = self.label_smoothing_loss(query_pred, query_y_low)
        else:
            loss = self.criterion_ce(query_pred, query_y_low)
        
        # 可选：Dice损失
        if self.ablation_config['use_dice_loss']:
            dice = self.dice_loss(query_pred, query_y_low)
            loss = loss + self.dice_weight * dice
        
        # 可选：Lovász损失
        if self.ablation_config['use_lovasz_loss']:
            lovasz = self.lovasz_loss(query_pred, query_y_low)
            loss = loss + self.lovasz_weight * lovasz
        
        # 可选：对比学习损失
        if self.ablation_config['use_contrastive_loss'] and self.training:
            contrastive = self.contrastive_loss(
                fg_prototypes, bg_prototype, query_feat_all, query_y_low
            )
            loss = loss + self.contrastive_weight * contrastive
        
        # 可选：一致性损失
        if self.ablation_config['use_consistency_loss'] and self.training:
            consistency = self.consistency_loss(query_feat_all, query_y_low)
            loss = loss + self.consistency_weight * consistency
        
        # 可选：Focal损失（关注难分样本）
        if self.ablation_config.get('use_focal_loss', False):
            focal = self.focal_loss(query_pred, query_y_low)
            loss = loss + self.focal_weight * focal
        
        # 可选：边界感知损失
        if self.ablation_config.get('use_boundary_loss', False):
            boundary = self.boundary_loss(query_pred, query_y_low)
            loss = loss + self.boundary_weight * boundary
        
        # 注意：Label Smoothing 已在上方替换 CE，不再叠加
        
        # 可选：支持集-查询集一致性约束
        if self.ablation_config.get('use_sq_consistency', False) and self.training:
            support_feat_cat = torch.cat(list(support_feat), dim=0)
            sq_loss = self.sq_consistency_loss(support_feat_cat, query_feat_all)
            loss = loss + self.sq_consistency_weight * sq_loss
        
        # Base class损失
        if query_base_y is not None:
            loss = loss + self.criterion_base(q_base_pred, query_base_y.cuda())

        # 上采样预测
        final_pred = (
            pointops.interpolation(
                query_x_low,
                query_x[:, :3].cuda().contiguous(),
                query_pred.contiguous(),
                query_offset_low,
                query_offset.cuda(),
            )
            .transpose(0, 1)
            .unsqueeze(0)
        )

        return final_pred, loss

    def getFeatures(self, ptclouds, offset, gt, query_base_y=None):
        """获取backbone特征"""
        coord, feat = (
            ptclouds[:, :3].contiguous(),
            ptclouds[:, 3:6].contiguous(),
        )

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * int(o)) for ii, o in enumerate(offset_)], 0
        ).long()

        sigma = 1.0
        radius = 2.5 * self.args.grid_size * sigma
        batch = batch.to(coord.device)
        neighbor_idx = tp.ball_query(
            radius,
            self.args.max_num_neighbors,
            coord,
            coord,
            mode="partial_dense",
            batch_x=batch,
            batch_y=batch,
        )[0]

        coord, feat, offset, gt = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
            gt.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        if self.args.concat_xyz:
            feat = torch.cat([feat, coord], 1)

        feat, coord, offset, gt, base_pred, query_base_y = self.encoder(
            feat, coord, offset, batch, neighbor_idx, gt, query_base_y
        )

        feat = self.bk_ffn(feat)
        
        # 可选：位置编码
        if self.ablation_config['use_pos_encoding']:
            feat = self.pos_encoding(coord, feat)
        
        return feat, coord, offset, gt, base_pred, query_base_y

    def getPrototypes(self, coords, feats, masks, k=100):
        """提取原型"""
        prototypes = []
        for i in range(0, self.n_way * self.k_shot):
            coord = coords[i][:, :3]
            feat = feats[i]
            mask = masks[i].bool()

            coord_mask = coord[mask]
            feat_mask = feat[mask]
            protos = self.getMutiplePrototypes(coord_mask, feat_mask, k)
            prototypes.append(protos)

        prototypes = torch.cat(prototypes)
        return prototypes

    def getMutiplePrototypes(self, coord, feat, num_prototypes):
        """使用FPS提取多个原型"""
        if feat.shape[0] <= num_prototypes:
            no_feats = feat.new_zeros(1, self.feat_dim).expand(
                num_prototypes - feat.shape[0], -1
            )
            feat = torch.cat([feat, no_feats])
            return feat

        fps_index = pointops.furthestsampling(
            coord,
            torch.cuda.IntTensor([coord.shape[0]]),
            torch.cuda.IntTensor([num_prototypes]),
        ).long()

        num_prototypes = len(fps_index)
        farthest_seeds = feat[fps_index]
        distances = torch.linalg.norm(
            feat[:, None, :] - farthest_seeds[None, :, :], dim=2
        )

        assignments = torch.argmin(distances, dim=1)

        prototypes = torch.zeros((num_prototypes, self.feat_dim), device="cuda")
        for i in range(num_prototypes):
            selected = torch.nonzero(assignments == i).squeeze(1)
            selected = feat[selected, :]
            if len(selected) == 0:
                prototypes[i] = feat[fps_index[i]]
            else:
                prototypes[i] = selected.mean(0)

        return prototypes
    
    def get_ablation_config_str(self):
        """返回当前消融配置的字符串描述"""
        enabled = [k for k, v in self.ablation_config.items() if v]
        if not enabled:
            return "baseline"
        return "+".join([k.replace('use_', '') for k in enabled])
