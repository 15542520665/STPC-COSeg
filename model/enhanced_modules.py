"""
Enhanced Modules for COSeg Optimization.
包含最先进的方法来提升Few-shot 3D Point Cloud Segmentation性能。

Features:
1. Prototype Contrastive Learning - 原型对比学习
2. Cross-Attention Prototype Refinement - 交叉注意力原型精炼
3. Multi-Scale Feature Aggregation - 多尺度特征聚合
4. Adaptive Prototype Generation - 自适应原型生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class PrototypeContrastiveLoss(nn.Module):
    """
    原型对比学习损失：增强前景/背景原型之间的区分度。
    基于InfoNCE，将相同类别的原型拉近，不同类别的原型推远。
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, fg_prototypes, bg_prototypes, query_features, query_labels):
        """
        Args:
            fg_prototypes: [N_way * n_subprototypes, C] 前景原型
            bg_prototypes: [n_subprototypes, C] 背景原型
            query_features: [N_q, C] 查询特征
            query_labels: [N_q] 查询标签
        """
        device = fg_prototypes.device
        
        # 归一化特征
        fg_prototypes = F.normalize(fg_prototypes, dim=-1)
        bg_prototypes = F.normalize(bg_prototypes, dim=-1)
        query_features = F.normalize(query_features, dim=-1)
        
        # 所有原型
        all_prototypes = torch.cat([bg_prototypes, fg_prototypes], dim=0)  # [N_proto, C]
        
        # 计算相似度
        similarity = torch.mm(query_features, all_prototypes.t()) / self.temperature  # [N_q, N_proto]
        
        n_bg = bg_prototypes.shape[0]
        
        # 前景点应该与前景原型相似
        fg_mask = query_labels > 0
        if fg_mask.sum() > 0:
            fg_sim = similarity[fg_mask][:, n_bg:]  # 前景点与前景原型
            fg_labels = torch.zeros(fg_sim.shape[0], dtype=torch.long, device=device)
            fg_loss = F.cross_entropy(fg_sim, fg_labels)
        else:
            fg_loss = torch.tensor(0.0, device=device)
        
        # 背景点应该与背景原型相似
        bg_mask = query_labels == 0
        if bg_mask.sum() > 0:
            bg_sim = similarity[bg_mask][:, :n_bg]  # 背景点与背景原型
            bg_labels = torch.zeros(bg_sim.shape[0], dtype=torch.long, device=device)
            bg_loss = F.cross_entropy(bg_sim, bg_labels)
        else:
            bg_loss = torch.tensor(0.0, device=device)
        
        return (fg_loss + bg_loss) / 2


class CrossAttentionPrototypeRefinement(nn.Module):
    """
    交叉注意力原型精炼模块。
    使用查询特征来精炼原型，使原型更好地适应当前场景。
    """
    def __init__(self, feat_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(feat_dim, feat_dim)
        self.k_proj = nn.Linear(feat_dim, feat_dim)
        self.v_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 4, feat_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, prototypes, query_features):
        """
        Args:
            prototypes: [N_proto, C] 原型
            query_features: [N_q, C] 查询特征
        Returns:
            refined_prototypes: [N_proto, C] 精炼后的原型
        """
        N_proto, C = prototypes.shape
        N_q = query_features.shape[0]
        
        # 交叉注意力：原型作为Query，查询特征作为Key/Value
        q = self.q_proj(prototypes)  # [N_proto, C]
        k = self.k_proj(query_features)  # [N_q, C]
        v = self.v_proj(query_features)  # [N_q, C]
        
        # 多头注意力
        q = rearrange(q, 'n (h d) -> h n d', h=self.num_heads)
        k = rearrange(k, 'n (h d) -> h n d', h=self.num_heads)
        v = rearrange(v, 'n (h d) -> h n d', h=self.num_heads)
        
        attn = torch.einsum('hid,hjd->hij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('hij,hjd->hid', attn, v)
        out = rearrange(out, 'h n d -> n (h d)')
        out = self.out_proj(out)
        
        # 残差连接
        prototypes = self.norm1(prototypes + out)
        prototypes = self.norm2(prototypes + self.ffn(prototypes))
        
        return prototypes


class MultiScaleFeatureAggregation(nn.Module):
    """
    多尺度特征聚合模块。
    融合不同尺度的特征，增强对不同大小物体的感知。
    """
    def __init__(self, feat_dim, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.ReLU(inplace=True)
            ) for _ in scales
        ])
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * len(scales), feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features, coords, batch_idx):
        """
        Args:
            features: [N, C] 点特征
            coords: [N, 3] 点坐标
            batch_idx: [N] batch索引
        Returns:
            aggregated: [N, C] 聚合后的特征
        """
        multi_scale_feats = []
        
        for scale, conv in zip(self.scales, self.convs):
            # 简单的多尺度：通过特征变换模拟不同尺度
            scaled_feat = conv(features)
            multi_scale_feats.append(scaled_feat)
        
        # 融合多尺度特征
        concat_feat = torch.cat(multi_scale_feats, dim=-1)
        aggregated = self.fusion(concat_feat)
        
        return aggregated


class AdaptivePrototypeGeneration(nn.Module):
    """
    自适应原型生成模块。
    不使用固定数量的原型，而是根据输入特征分布动态生成原型。
    使用可学习的聚类中心。
    """
    def __init__(self, feat_dim, max_prototypes=100, min_prototypes=10):
        super().__init__()
        self.feat_dim = feat_dim
        self.max_prototypes = max_prototypes
        self.min_prototypes = min_prototypes
        
        # 可学习的原型生成网络
        self.prototype_predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim * 2, feat_dim),
        )
        
        # 置信度预测
        self.confidence_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, coords, mask, num_prototypes):
        """
        Args:
            features: [N, C] 点特征
            coords: [N, 3] 点坐标
            mask: [N] 目标类别mask
            num_prototypes: int 目标原型数量
        Returns:
            prototypes: [num_prototypes, C] 生成的原型
        """
        # 过滤目标点
        masked_features = features[mask]
        masked_coords = coords[mask]
        
        if masked_features.shape[0] < num_prototypes:
            # 如果点数不足，填充
            padding = torch.zeros(
                num_prototypes - masked_features.shape[0],
                self.feat_dim,
                device=features.device
            )
            return torch.cat([masked_features, padding], dim=0)
        
        # 使用FPS选择代表点
        from lib.pointops2.functions import pointops
        fps_idx = pointops.furthestsampling(
            masked_coords,
            torch.cuda.IntTensor([masked_coords.shape[0]]),
            torch.cuda.IntTensor([num_prototypes])
        ).long()
        
        # 生成原型
        seed_features = masked_features[fps_idx]
        
        # 计算每个点到种子的距离
        distances = torch.cdist(masked_features, seed_features)  # [N, K]
        assignments = torch.argmin(distances, dim=1)  # [N]
        
        # 聚合每个簇
        prototypes = []
        confidences = []
        for i in range(num_prototypes):
            cluster_mask = assignments == i
            if cluster_mask.sum() > 0:
                cluster_features = masked_features[cluster_mask]
                # 使用注意力加权聚合
                conf = self.confidence_head(cluster_features)
                conf = F.softmax(conf, dim=0)
                proto = (cluster_features * conf).sum(dim=0)
                proto = self.prototype_predictor(proto)
            else:
                proto = seed_features[i]
                proto = self.prototype_predictor(proto)
            prototypes.append(proto)
        
        prototypes = torch.stack(prototypes)
        return prototypes


class PositionalEncoding3D(nn.Module):
    """
    3D位置编码，增强模型对空间位置的感知。
    使用正弦余弦位置编码。
    """
    def __init__(self, feat_dim, max_len=10000):
        super().__init__()
        self.feat_dim = feat_dim
        
        # 为x, y, z三个维度分别生成位置编码
        self.linear = nn.Linear(feat_dim, feat_dim)
        
    def forward(self, coords, features):
        """
        Args:
            coords: [N, 3] 点坐标
            features: [N, C] 点特征
        Returns:
            encoded: [N, C] 加入位置编码后的特征
        """
        # 归一化坐标
        coords_norm = coords - coords.min(dim=0, keepdim=True)[0]
        coords_norm = coords_norm / (coords_norm.max(dim=0, keepdim=True)[0] + 1e-6)
        
        # 生成正弦余弦编码
        dim = self.feat_dim // 6
        pe = torch.zeros(coords.shape[0], self.feat_dim, device=coords.device)
        
        for i, coord in enumerate(['x', 'y', 'z']):
            div_term = torch.exp(torch.arange(0, dim, 2, device=coords.device).float() * 
                                -(math.log(10000.0) / dim))
            idx = i * (dim * 2)
            pos = coords_norm[:, i:i+1]
            pe[:, idx:idx+dim:2] = torch.sin(pos * div_term)
            pe[:, idx+1:idx+dim:2] = torch.cos(pos * div_term)
        
        # 融合位置编码
        return features + self.linear(pe[:, :self.feat_dim])


class EnhancedAggregatorLayer(nn.Module):
    """
    增强版聚合层，包含：
    1. 空间注意力
    2. 类别注意力
    3. 通道注意力
    4. 残差连接
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 空间注意力
        self.spatial_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 类别注意力
        self.class_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 通道注意力 (SE-like)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, base_guidance=None):
        """
        Args:
            x: [B, C, T, N] 输入特征 (T=num_classes, N=num_points)
            base_guidance: [N, 1] 可选的基类指导
        Returns:
            x: [B, C, T, N] 输出特征
        """
        B, C, T, N = x.shape
        
        # 空间注意力 (沿N维度)
        x_spatial = rearrange(x, 'B C T N -> (B T) N C')
        x_spatial = self.norm1(x_spatial)
        x_attn, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = x_spatial + x_attn
        x_spatial = rearrange(x_spatial, '(B T) N C -> B C T N', B=B, T=T)
        x = x + x_spatial
        
        # 类别注意力 (沿T维度)
        x_class = rearrange(x, 'B C T N -> (B N) T C')
        x_class = self.norm2(x_class)
        x_attn, _ = self.class_attn(x_class, x_class, x_class)
        x_class = x_class + x_attn
        x_class = rearrange(x_class, '(B N) T C -> B C T N', B=B, N=N)
        x = x + x_class
        
        # 通道注意力
        x_avg = x.mean(dim=(2, 3))  # [B, C]
        channel_weight = self.channel_attn(x_avg.unsqueeze(-1))  # [B, C]
        x = x * channel_weight.unsqueeze(-1).unsqueeze(-1)
        
        # FFN
        x_ffn = rearrange(x, 'B C T N -> B T N C')
        x_ffn = self.norm4(x_ffn)
        x_ffn = self.ffn(x_ffn)
        x_ffn = rearrange(x_ffn, 'B T N C -> B C T N')
        x = x + x_ffn
        
        return x


class PrototypeFusionModule(nn.Module):
    """
    原型融合模块：将多个shot的原型进行自适应融合。
    适用于k-shot场景。
    """
    def __init__(self, feat_dim, k_shot):
        super().__init__()
        self.k_shot = k_shot
        self.feat_dim = feat_dim
        
        # 自注意力融合
        self.self_attn = nn.MultiheadAttention(
            feat_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(torch.ones(k_shot) / k_shot)
        
        self.norm = nn.LayerNorm(feat_dim)
        
    def forward(self, prototypes):
        """
        Args:
            prototypes: [k_shot * n_subprototypes, C] 多个shot的原型
        Returns:
            fused: [n_subprototypes, C] 融合后的原型
        """
        n_total = prototypes.shape[0]
        n_sub = n_total // self.k_shot
        
        # 重塑为 [n_subprototypes, k_shot, C]
        proto_reshaped = prototypes.view(self.k_shot, n_sub, -1).permute(1, 0, 2)
        
        # 自注意力
        proto_normed = self.norm(proto_reshaped)
        proto_attn, _ = self.self_attn(proto_normed, proto_normed, proto_normed)
        proto_reshaped = proto_reshaped + proto_attn
        
        # 加权融合
        weights = F.softmax(self.fusion_weight, dim=0)
        fused = (proto_reshaped * weights.view(1, -1, 1)).sum(dim=1)
        
        return fused


class BoundaryAwareLoss(nn.Module):
    """
    边界感知损失：增强对物体边界的分割精度。
    """
    def __init__(self, boundary_weight=1.0):
        super().__init__()
        self.boundary_weight = boundary_weight
        
    def forward(self, pred, target, coords):
        """
        Args:
            pred: [N, C] 预测logits
            target: [N] 目标标签
            coords: [N, 3] 点坐标
        Returns:
            loss: 边界感知损失
        """
        # 基础交叉熵
        ce_loss = F.cross_entropy(pred, target, ignore_index=255)
        
        # 检测边界点（通过KNN找到标签不一致的点）
        with torch.no_grad():
            # 计算K近邻
            dist = torch.cdist(coords, coords)
            _, knn_idx = dist.topk(k=10, dim=-1, largest=False)
            
            # 获取邻居标签
            neighbor_labels = target[knn_idx]  # [N, K]
            center_labels = target.unsqueeze(-1).expand_as(neighbor_labels)
            
            # 边界点：有邻居标签与自身不同
            valid_mask = (target != 255) & (neighbor_labels != 255).all(dim=-1)
            is_boundary = ((neighbor_labels != center_labels) & (neighbor_labels != 255)).any(dim=-1)
            boundary_mask = valid_mask & is_boundary
        
        if boundary_mask.sum() > 0:
            boundary_loss = F.cross_entropy(
                pred[boundary_mask], 
                target[boundary_mask],
                ignore_index=255
            )
            return ce_loss + self.boundary_weight * boundary_loss
        
        return ce_loss


class FeatureConsistencyLoss(nn.Module):
    """
    特征一致性损失：确保相同类别点的特征相似。
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        Args:
            features: [N, C] 点特征
            labels: [N] 点标签
        Returns:
            loss: 特征一致性损失
        """
        # 归一化
        features = F.normalize(features, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.mm(features, features.t()) / self.temperature
        
        # 创建标签mask
        valid_mask = labels != 255
        labels_valid = labels[valid_mask]
        features_valid = features[valid_mask]
        
        if labels_valid.shape[0] < 2:
            return torch.tensor(0.0, device=features.device)
        
        # 相同标签为正样本
        label_mask = labels_valid.unsqueeze(0) == labels_valid.unsqueeze(1)
        
        # 计算对比损失
        sim_valid = torch.mm(features_valid, features_valid.t()) / self.temperature
        
        # 排除自身
        eye_mask = ~torch.eye(label_mask.shape[0], dtype=torch.bool, device=labels.device)
        label_mask = label_mask & eye_mask
        
        if label_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
        
        # InfoNCE
        exp_sim = torch.exp(sim_valid)
        log_prob = sim_valid - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # 只计算正样本对的损失
        loss = -(log_prob * label_mask.float()).sum() / label_mask.sum()
        
        return loss


# ========= 原型细化模块 (Prototype Refinement) =========

class PrototypeRefinementModule(nn.Module):
    """
    原型细化模块：通过查询特征反馈迭代优化原型。
    
    创新点：
    1. 利用查询集的初步预测结果反馈细化原型
    2. 通过注意力机制选择性聚合高置信度查询特征
    3. 迭代细化提升原型质量
    
    流程：
    初始原型 → 初步预测 → 高置信度特征选择 → 原型更新 → 最终预测
    """
    def __init__(self, feat_dim=256, num_iterations=2, confidence_threshold=0.7):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_iterations = num_iterations
        self.confidence_threshold = confidence_threshold
        
        # 原型更新网络
        self.proto_updater = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        
        # 置信度预测
        self.confidence_net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # 可学习的融合权重
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, prototypes, query_features, return_confidence=False):
        """
        Args:
            prototypes: [N_proto, C] 初始原型
            query_features: [N_q, C] 查询特征
            return_confidence: 是否返回置信度
            
        Returns:
            refined_prototypes: [N_proto, C] 细化后的原型
        """
        refined_protos = prototypes
        
        for iter_idx in range(self.num_iterations):
            # 1. 计算查询特征与原型的相似度
            similarity = F.cosine_similarity(
                query_features.unsqueeze(1),  # [N_q, 1, C]
                refined_protos.unsqueeze(0),   # [1, N_proto, C]
                dim=-1
            )  # [N_q, N_proto]
            
            # 2. 获取每个查询点的预测类别和置信度
            pred_probs = F.softmax(similarity * 10, dim=-1)  # 温度缩放
            pred_confidence, pred_class = pred_probs.max(dim=-1)  # [N_q]
            
            # 3. 对每个原型，聚合高置信度的查询特征
            updated_protos = []
            for proto_idx in range(refined_protos.shape[0]):
                # 找到预测为该类且置信度高的查询点
                mask = (pred_class == proto_idx) & (pred_confidence > self.confidence_threshold)
                
                if mask.sum() > 0:
                    # 用置信度加权聚合查询特征
                    confident_features = query_features[mask]  # [N_conf, C]
                    confident_weights = pred_confidence[mask]  # [N_conf]
                    confident_weights = confident_weights / confident_weights.sum()
                    
                    # 加权平均
                    aggregated_feat = (confident_features * confident_weights.unsqueeze(-1)).sum(dim=0)
                    
                    # 融合原型和聚合特征
                    combined = torch.cat([refined_protos[proto_idx], aggregated_feat], dim=-1)
                    delta = self.proto_updater(combined)
                    
                    # 残差更新
                    new_proto = refined_protos[proto_idx] + self.fusion_weight * delta
                else:
                    # 没有高置信度样本，保持原型不变
                    new_proto = refined_protos[proto_idx]
                
                updated_protos.append(new_proto)
            
            refined_protos = torch.stack(updated_protos, dim=0)
            
            # L2归一化保持原型稳定
            refined_protos = F.normalize(refined_protos, dim=-1) * (prototypes.norm(dim=-1, keepdim=True).mean())
        
        if return_confidence:
            return refined_protos, pred_confidence
        return refined_protos


class QueryGuidedPrototypeRefinement(nn.Module):
    """
    查询引导的原型细化：更轻量级的实现。
    利用查询特征的全局统计信息来细化原型。
    """
    def __init__(self, feat_dim=256):
        super().__init__()
        self.feat_dim = feat_dim
        
        # 查询特征编码
        self.query_encoder = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
        )
        
        # 原型调制网络
        self.modulation = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Sigmoid(),  # 输出调制系数
        )
        
        # 原型偏移网络
        self.offset = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Tanh(),  # 输出偏移量，范围[-1, 1]
        )
        
        self.scale = nn.Parameter(torch.tensor(0.1))  # 控制偏移幅度
    
    def forward(self, prototypes, query_features):
        """
        Args:
            prototypes: [N_proto, C] 原型
            query_features: [N_q, C] 查询特征
            
        Returns:
            refined_prototypes: [N_proto, C] 细化后的原型
        """
        # 1. 编码查询特征的全局信息
        query_encoded = self.query_encoder(query_features)  # [N_q, C]
        query_global = query_encoded.mean(dim=0, keepdim=True)  # [1, C]
        
        # 2. 对每个原型进行细化
        refined_protos = []
        for proto_idx in range(prototypes.shape[0]):
            proto = prototypes[proto_idx:proto_idx+1]  # [1, C]
            
            # 拼接原型和查询全局特征
            combined = torch.cat([proto, query_global], dim=-1)  # [1, 2C]
            
            # 计算调制系数和偏移
            modulation = self.modulation(combined)  # [1, C]
            offset = self.offset(combined) * self.scale  # [1, C]
            
            # 应用调制和偏移
            refined = proto * modulation + offset
            refined_protos.append(refined)
        
        refined_protos = torch.cat(refined_protos, dim=0)  # [N_proto, C]
        
        return refined_protos
