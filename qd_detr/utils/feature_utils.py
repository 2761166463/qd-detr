import torch
import numpy as np
import math
def apply_corruption(features, corruption_type, corruption_ratio):
    original_shape = features.shape
    # 所有操作均在传入特征的副本上进行
    features = features.clone()
    """
    特征破坏函数，支持多种破坏类型
    Args:
        features: torch.Tensor 输入特征
        corruption_type: str 破坏类型 ['none', 'mask', 'noise',  'token_mask', 'text_noise', 'time_mask', 'band_drop']
        ratio: float 破坏比例 (0.0-1.0)
    """
    # 输入校验
    assert corruption_type in ['none', 'mask', 'noise', 'token_mask', 'text_noise', 'time_mask', 'band_drop'], \
        f"Invalid corruption type: {corruption_type}"
    assert 0 <= corruption_ratio <= 1, f"Ratio {corruption_ratio} must be in [0,1]"
    
    if corruption_type == "none" or corruption_ratio <= 0:
        return features
    
    # 保持设备一致性
    device = features.device
    with torch.no_grad():  # 梯度保护
        if corruption_type == "mask":
            mask = torch.rand(features.size(), device=device) < corruption_ratio
            features = features.masked_fill(mask, 0.0)  # 非原位操作
            
        elif corruption_type == "noise":
            noise = torch.randn_like(features, device=device) * corruption_ratio
            features = features + noise  # 加法操作
            
        elif corruption_type == "token_mask":  # 随机遮蔽文本token
            seq_len = features.size(1)
            mask_num = max(1, math.ceil(seq_len * corruption_ratio))
            for i in range(features.size(0)):
                mask_indices = torch.randperm(seq_len)[:mask_num]
                features[i, mask_indices] = 0.0
                
        elif corruption_type == "text_noise":  # 使用真实不相关文本替换
            # 随机选择要替换的特征维度
            mask = torch.rand_like(features, dtype=torch.float32) < corruption_ratio
    
            # 生成随机索引用于获取真实替代文本特征
            batch_size = features.size(0)
            perm_indices = torch.randperm(batch_size)
    
            # 从其他样本获取真实特征进行替换
            features = torch.where(mask, 
                         features[perm_indices],  # 使用其他样本的真实特征
                         features)
        # 时间轴遮挡
        elif corruption_type == "time_mask":
            seq_len = features.size(1)
            mask_len = max(1, math.ceil(seq_len * corruption_ratio))  
            if mask_len >= seq_len:
                features.fill_(0.0)
            else:
                # 为每个样本生成不同的遮蔽位置
                starts = torch.randint(0, seq_len - mask_len + 1, (features.size(0),)) 
                for i in range(features.size(0)):
                    features[i, starts[i]:starts[i]+mask_len] = 0.0

        elif corruption_type == "band_drop":
            feat_dim = features.size(-1)
            drop_dim = max(1, math.ceil(feat_dim * corruption_ratio))  
            # 随机起始位置
            start = torch.randint(0, max(1, feat_dim - drop_dim), (1,)).item()
            features[..., start:start+drop_dim] = 0.0

    assert features.shape == original_shape, \
        f"维度变化 {original_shape}->{features.shape}"

    return features