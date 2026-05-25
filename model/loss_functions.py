"""
Advanced Loss Functions for COSeg.
包含最先进的损失函数来提升分割性能。

Features:
1. Dice Loss - 处理类别不平衡
2. Focal Loss - 关注难分样本
3. Lovász Loss - 直接优化IoU
4. OHEM Loss - 在线困难样本挖掘
5. Multi-Task Loss - 多任务学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice Loss用于处理类别不平衡问题。
    特别适合分割任务中前景/背景比例悬殊的情况。
    """
    def __init__(self, smooth=1.0, ignore_index=255, weight=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight
        
    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] 预测logits
            target: [N] 目标标签
        Returns:
            loss: Dice损失
        """
        # 过滤忽略标签
        valid_mask = target != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        num_classes = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)
        
        # One-hot编码
        target_one_hot = F.one_hot(target, num_classes).float()
        
        # 计算Dice
        intersection = (pred_soft * target_one_hot).sum(dim=0)
        union = pred_soft.sum(dim=0) + target_one_hot.sum(dim=0)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        if self.weight is not None:
            dice = dice * self.weight.to(dice.device)
            
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss用于关注难分样本。
    通过调整gamma参数控制对困难样本的关注程度。
    """
    def __init__(self, gamma=2.0, alpha=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] 预测logits
            target: [N] 目标标签
        Returns:
            loss: Focal损失
        """
        valid_mask = target != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[target]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax Loss直接优化IoU。
    是IoU的凸松弛形式，可微分。
    """
    def __init__(self, ignore_index=255, per_class=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_class = per_class
        
    def lovasz_grad(self, gt_sorted):
        """计算Lovász梯度"""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
    
    def lovasz_softmax_flat(self, probas, labels):
        """
        Flatten版本的Lovász-Softmax
        """
        C = probas.shape[1]
        losses = []
        
        for c in range(C):
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            errors = (1 - probas[:, c]) * fg + probas[:, c] * (1 - fg)
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
            fg_sorted = fg[perm]
            grad = self.lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))
            
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=probas.device)
    
    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] 预测logits
            target: [N] 目标标签
        Returns:
            loss: Lovász损失
        """
        valid_mask = target != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        probas = F.softmax(pred, dim=1)
        return self.lovasz_softmax_flat(probas, target)


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining (OHEM) Loss.
    只对最困难的样本计算损失，提高模型对困难样本的处理能力。
    """
    def __init__(self, ratio=0.7, min_kept=10000, ignore_index=255):
        super().__init__()
        self.ratio = ratio
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        
    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] 预测logits
            target: [N] 目标标签
        Returns:
            loss: OHEM损失
        """
        valid_mask = target != self.ignore_index
        num_valid = valid_mask.sum().item()
        
        if num_valid == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 计算所有损失
        loss = self.criterion(pred, target)
        
        # 只保留有效样本的损失
        loss_valid = loss[valid_mask]
        
        # 计算保留的样本数量
        num_kept = max(int(num_valid * self.ratio), min(self.min_kept, num_valid))
        
        # 选择损失最大的样本
        loss_sorted, _ = torch.sort(loss_valid, descending=True)
        loss_kept = loss_sorted[:num_kept]
        
        return loss_kept.mean()


class CombinedSegLoss(nn.Module):
    """
    组合多种损失函数的复合损失。
    可以灵活配置不同损失的权重。
    """
    def __init__(
        self,
        ce_weight=1.0,
        dice_weight=0.5,
        focal_weight=0.0,
        lovasz_weight=0.5,
        ohem_weight=0.0,
        ignore_index=255,
        class_weights=None
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight
        self.ohem_weight = ohem_weight
        
        # 初始化各个损失
        if ce_weight > 0:
            self.ce_loss = nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=ignore_index
            )
        if dice_weight > 0:
            self.dice_loss = DiceLoss(ignore_index=ignore_index)
        if focal_weight > 0:
            self.focal_loss = FocalLoss(ignore_index=ignore_index)
        if lovasz_weight > 0:
            self.lovasz_loss = LovaszSoftmaxLoss(ignore_index=ignore_index)
        if ohem_weight > 0:
            self.ohem_loss = OHEMLoss(ignore_index=ignore_index)
            
    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] 预测logits
            target: [N] 目标标签
        Returns:
            loss: 组合损失
            loss_dict: 各损失的详细值
        """
        loss = 0.0
        loss_dict = {}
        
        if self.ce_weight > 0:
            ce = self.ce_loss(pred, target)
            loss = loss + self.ce_weight * ce
            loss_dict['ce_loss'] = ce.item()
            
        if self.dice_weight > 0:
            dice = self.dice_loss(pred, target)
            loss = loss + self.dice_weight * dice
            loss_dict['dice_loss'] = dice.item()
            
        if self.focal_weight > 0:
            focal = self.focal_loss(pred, target)
            loss = loss + self.focal_weight * focal
            loss_dict['focal_loss'] = focal.item()
            
        if self.lovasz_weight > 0:
            lovasz = self.lovasz_loss(pred, target)
            loss = loss + self.lovasz_weight * lovasz
            loss_dict['lovasz_loss'] = lovasz.item()
            
        if self.ohem_weight > 0:
            ohem = self.ohem_loss(pred, target)
            loss = loss + self.ohem_weight * ohem
            loss_dict['ohem_loss'] = ohem.item()
            
        loss_dict['total_loss'] = loss.item()
        
        return loss, loss_dict


class AdaptiveClassBalancedLoss(nn.Module):
    """
    自适应类别平衡损失。
    根据每个batch中的类别分布动态调整权重。
    """
    def __init__(self, num_classes, beta=0.9999, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.ignore_index = ignore_index
        
        # 类别计数（用于在线更新）
        self.register_buffer('class_counts', torch.ones(num_classes))
        
    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] 预测logits
            target: [N] 目标标签
        Returns:
            loss: 自适应平衡损失
        """
        valid_mask = target != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]
        
        # 更新类别计数
        if self.training:
            for c in range(self.num_classes):
                count = (target_valid == c).sum().float()
                self.class_counts[c] = self.beta * self.class_counts[c] + (1 - self.beta) * count
        
        # 计算有效类别权重
        effective_num = 1.0 - torch.pow(self.beta, self.class_counts)
        weights = (1.0 - self.beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * self.num_classes
        
        # 加权交叉熵
        loss = F.cross_entropy(pred_valid, target_valid, weight=weights)
        
        return loss


class UncertaintyWeightedLoss(nn.Module):
    """
    不确定性加权多任务损失。
    自动学习每个损失项的权重。
    """
    def __init__(self, num_losses=2):
        super().__init__()
        # 可学习的log方差参数
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        
    def forward(self, losses):
        """
        Args:
            losses: list of tensor, 各个损失值
        Returns:
            total_loss: 加权后的总损失
        """
        total_loss = 0.0
        
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss = total_loss + precision * loss + self.log_vars[i]
            
        return total_loss


class ContrastivePrototypeLoss(nn.Module):
    """
    对比原型损失：
    1. 拉近点与对应类别原型的距离
    2. 推远点与其他类别原型的距离
    """
    def __init__(self, temperature=0.1, margin=0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, query_features, prototypes, labels, n_way):
        """
        Args:
            query_features: [N, C] 查询点特征
            prototypes: [(n_way+1)*n_sub, C] 所有原型
            labels: [N] 点标签
            n_way: 类别数
        Returns:
            loss: 对比原型损失
        """
        valid_mask = labels != 255
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=query_features.device)
        
        query_features = query_features[valid_mask]
        labels = labels[valid_mask]
        
        # 归一化
        query_features = F.normalize(query_features, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)
        
        # 计算相似度
        similarity = torch.mm(query_features, prototypes.t()) / self.temperature
        
        # 每个类别的原型数量
        n_proto_per_class = prototypes.shape[0] // (n_way + 1)
        
        # 创建正样本mask
        pos_mask = torch.zeros_like(similarity, dtype=torch.bool)
        for i in range(n_way + 1):
            class_mask = labels == i
            start_idx = i * n_proto_per_class
            end_idx = (i + 1) * n_proto_per_class
            pos_mask[class_mask, start_idx:end_idx] = True
        
        # InfoNCE损失
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # 正样本损失
        pos_loss = -(log_prob * pos_mask.float()).sum() / pos_mask.sum().clamp(min=1)
        
        return pos_loss


class HierarchicalContrastiveLoss(nn.Module):
    """
    层次对比损失：
    在点级、区域级、场景级同时进行对比学习。
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.point_loss = ContrastivePrototypeLoss(temperature)
        
    def forward(self, features_list, prototypes, labels, n_way):
        """
        Args:
            features_list: list of [N_i, C] 多尺度特征
            prototypes: [(n_way+1)*n_sub, C] 原型
            labels: [N] 标签
            n_way: 类别数
        Returns:
            loss: 层次对比损失
        """
        total_loss = 0.0
        
        for i, features in enumerate(features_list):
            weight = 1.0 / (2 ** i)  # 越高层权重越小
            loss = self.point_loss(features, prototypes, labels, n_way)
            total_loss = total_loss + weight * loss
            
        return total_loss


# ============== 新增优化模块 ==============

class BoundaryLoss(nn.Module):
    """
    边界感知损失：关注分割边界区域，提升边界质量。
    通过检测预测和真值边界的差异来计算损失。
    """
    def __init__(self, ignore_index=255, theta=0.5):
        super().__init__()
        self.ignore_index = ignore_index
        self.theta = theta  # 边界权重系数
        
    def get_boundary_mask(self, labels, pred_labels):
        """
        获取边界mask - 预测和真值不一致的区域被认为是边界区域
        """
        # 简化版：预测错误的点更可能在边界
        boundary_mask = (pred_labels != labels).float()
        return boundary_mask
    
    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] 预测logits
            target: [N] 目标标签
        Returns:
            loss: 边界损失
        """
        valid_mask = target != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # 获取预测标签
        pred_labels = pred.argmax(dim=1)
        
        # 计算边界权重
        boundary_weight = self.get_boundary_mask(target, pred_labels)
        boundary_weight = 1.0 + self.theta * boundary_weight
        
        # 加权交叉熵
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = (ce_loss * boundary_weight).mean()
        
        return weighted_loss


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑损失：通过软化标签防止过拟合。
    将hard label转换为soft label，增强模型泛化能力。
    """
    def __init__(self, smoothing=0.1, ignore_index=255):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] 预测logits
            target: [N] 目标标签
        Returns:
            loss: 标签平滑损失
        """
        valid_mask = target != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        num_classes = pred.shape[1]
        
        # 创建软标签
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (num_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # KL散度损失
        log_probs = F.log_softmax(pred, dim=1)
        loss = (-smooth_target * log_probs).sum(dim=1).mean()
        
        return loss


class MultiScalePrototypeLoss(nn.Module):
    """
    多尺度原型损失：在多个特征尺度上计算原型匹配损失。
    通过不同尺度的特征增强原型表示的鲁棒性。
    """
    def __init__(self, scales=[1.0, 0.5, 0.25], temperature=0.1):
        super().__init__()
        self.scales = scales
        self.temperature = temperature
        
    def forward(self, query_features, prototypes, labels, n_way):
        """
        Args:
            query_features: [N, C] 查询点特征
            prototypes: [K, C] 原型特征
            labels: [N] 标签
            n_way: 类别数
        Returns:
            loss: 多尺度原型损失
        """
        valid_mask = labels != 255
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=query_features.device)
        
        query_features = query_features[valid_mask]
        labels = labels[valid_mask]
        
        total_loss = 0.0
        
        for scale in self.scales:
            # 缩放特征维度（通过随机采样模拟多尺度）
            if scale < 1.0:
                dim = int(query_features.shape[1] * scale)
                indices = torch.randperm(query_features.shape[1])[:dim]
                q_feat = query_features[:, indices]
                p_feat = prototypes[:, indices]
            else:
                q_feat = query_features
                p_feat = prototypes
            
            # 归一化
            q_feat = F.normalize(q_feat, dim=-1)
            p_feat = F.normalize(p_feat, dim=-1)
            
            # 计算相似度
            similarity = torch.mm(q_feat, p_feat.t()) / self.temperature
            
            # 计算损失 - 正确类别的原型应该有最高相似度
            # 假设原型按类别排列
            n_proto_per_class = p_feat.shape[0] // (n_way + 1)
            
            # 简化：使用最大相似度原型的类别
            proto_class_labels = torch.arange(n_way + 1).repeat_interleave(n_proto_per_class)
            proto_class_labels = proto_class_labels.to(similarity.device)
            
            # 对每个类别取最大相似度
            max_sim_per_class = []
            for c in range(n_way + 1):
                class_protos = (proto_class_labels == c)
                max_sim = similarity[:, class_protos].max(dim=1)[0]
                max_sim_per_class.append(max_sim)
            
            class_sim = torch.stack(max_sim_per_class, dim=1)  # [N, n_way+1]
            
            # 交叉熵损失
            loss = F.cross_entropy(class_sim, labels)
            total_loss = total_loss + loss / len(self.scales)
        
        return total_loss


class AuxiliaryBranchLoss(nn.Module):
    """
    辅助分支损失：添加辅助预测分支进行深度监督。
    在中间特征层添加额外的预测头，加速收敛并提升性能。
    """
    def __init__(self, ignore_index=255, aux_weight=0.4):
        super().__init__()
        self.ignore_index = ignore_index
        self.aux_weight = aux_weight
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
    def forward(self, main_pred, aux_pred, target):
        """
        Args:
            main_pred: [N, C] 主分支预测
            aux_pred: [N, C] 辅助分支预测
            target: [N] 目标标签
        Returns:
            loss: 组合损失
        """
        main_loss = self.ce_loss(main_pred, target)
        aux_loss = self.ce_loss(aux_pred, target)
        
        return main_loss + self.aux_weight * aux_loss


class PrototypeCalibrationLoss(nn.Module):
    """
    原型对比校准损失 (Prototype Contrastive Calibration Loss)
    
    核心思想：
    1. 类间分离：最大化前景原型与背景原型之间的距离
    2. 类内紧凑：最小化同类原型之间的距离
    
    这种损失通过在原型特征空间施加结构约束，
    使得模型学习到更具判别性的原型表示。
    
    Args:
        margin: 类间距离的最小margin
        intra_weight: 类内紧凑损失的权重
        inter_weight: 类间分离损失的权重
        temperature: 对比学习的温度参数
    """
    def __init__(self, margin=0.5, intra_weight=1.0, inter_weight=1.0, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight
        self.temperature = temperature
        
    def forward(self, fg_prototypes, bg_prototypes):
        """
        计算原型对比校准损失
        
        Args:
            fg_prototypes: [N_fg, D] 前景原型
            bg_prototypes: [N_bg, D] 背景原型
        Returns:
            loss: 对比校准损失
        """
        # 归一化原型（在单位球面上计算距离）
        fg_norm = F.normalize(fg_prototypes, p=2, dim=1)  # [N_fg, D]
        bg_norm = F.normalize(bg_prototypes, p=2, dim=1)  # [N_bg, D]
        
        loss = torch.tensor(0.0, device=fg_prototypes.device)
        
        # ========= 1. 类间分离损失 (Inter-class Separation) =========
        # 计算前景与背景之间的相似度矩阵
        fg_bg_sim = torch.mm(fg_norm, bg_norm.t())  # [N_fg, N_bg]
        
        # Margin-based hinge loss: 相似度应该小于 -margin
        # 即距离应该大于 (1 + margin)
        inter_loss = F.relu(fg_bg_sim + self.margin).mean()
        
        # ========= 2. 类内紧凑损失 (Intra-class Compactness) =========
        # 前景原型内部应该紧凑
        if fg_prototypes.shape[0] > 1:
            fg_fg_sim = torch.mm(fg_norm, fg_norm.t())  # [N_fg, N_fg]
            # 排除对角线（自身相似度为1）
            mask = ~torch.eye(fg_fg_sim.shape[0], dtype=torch.bool, device=fg_fg_sim.device)
            fg_intra_loss = 1.0 - fg_fg_sim[mask].mean()  # 希望相似度接近1
        else:
            fg_intra_loss = torch.tensor(0.0, device=fg_prototypes.device)
        
        # 背景原型内部应该紧凑
        if bg_prototypes.shape[0] > 1:
            bg_bg_sim = torch.mm(bg_norm, bg_norm.t())  # [N_bg, N_bg]
            mask = ~torch.eye(bg_bg_sim.shape[0], dtype=torch.bool, device=bg_bg_sim.device)
            bg_intra_loss = 1.0 - bg_bg_sim[mask].mean()
        else:
            bg_intra_loss = torch.tensor(0.0, device=bg_prototypes.device)
        
        intra_loss = (fg_intra_loss + bg_intra_loss) / 2.0
        
        # ========= 3. 组合损失 =========
        loss = self.inter_weight * inter_loss + self.intra_weight * intra_loss
        
        return loss


# ========= 支持集-查询集一致性约束 =========

class SupportQueryConsistencyLoss(nn.Module):
    """
    支持集-查询集特征分布一致性约束。
    通过减少support和query特征分布的差异来提升泛化能力。
    
    创新点：
    1. 使用MMD (Maximum Mean Discrepancy) 衡量分布差异
    2. 针对few-shot场景优化，减少domain gap
    """
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
    
    def gaussian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma=None):
        """计算高斯核矩阵"""
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / (bw + 1e-8)) for bw in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, support_feat, query_feat):
        """
        Args:
            support_feat: [N_s, C] 支持集特征
            query_feat: [N_q, C] 查询集特征
        Returns:
            loss: MMD损失
        """
        if support_feat.size(0) == 0 or query_feat.size(0) == 0:
            return torch.tensor(0.0, device=support_feat.device)
        
        # 采样以减少计算量
        max_samples = 1024
        if support_feat.size(0) > max_samples:
            idx = torch.randperm(support_feat.size(0))[:max_samples]
            support_feat = support_feat[idx]
        if query_feat.size(0) > max_samples:
            idx = torch.randperm(query_feat.size(0))[:max_samples]
            query_feat = query_feat[idx]
        
        batch_size = support_feat.size(0)
        
        kernels = self.gaussian_kernel(
            support_feat, query_feat,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num
        )
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return loss


class AdaptivePrototypeLoss(nn.Module):
    """
    自适应原型损失：根据样本难度动态调整损失权重。
    
    创新点：
    1. 对难分类样本给予更高权重
    2. 基于预测置信度自适应调整
    """
    def __init__(self, gamma=2.0, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, pred, target, prototypes):
        """
        Args:
            pred: [N, C] 预测logits
            target: [N] 目标标签
            prototypes: [C, D] 类别原型
        Returns:
            loss: 自适应原型损失
        """
        valid_mask = target != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # 计算预测概率
        prob = F.softmax(pred, dim=1)
        
        # 获取正确类别的概率
        target_prob = prob.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # 计算难度权重（概率越低，权重越高）
        weight = (1 - target_prob) ** self.gamma
        
        # 加权交叉熵
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = (weight * ce_loss).mean()
        
        return weighted_loss
