import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """残差块增强特征重建能力"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        return x + self.block(x)

class MultimodalAutoencoder(nn.Module):
    def __init__(self, 
                 video_feat_dim=2048,
                 text_feat_dim=512,
                 audio_feat_dim=2050,
                 latent_dim=256,
                 apply_to_modality="video"):
        super().__init__()
        self.apply_to_modality = apply_to_modality.lower()
        self.noise_layer = nn.Dropout(p=0.3)  # 对抗噪声层
        
        # 视频编解码器初始化
        if 'video' in self.apply_to_modality:
            self.video_encoder = self._build_video_encoder(video_feat_dim, latent_dim)
            self.video_decoder = self._build_video_decoder(latent_dim, video_feat_dim)
            
        # 文本编解码器初始化
        if 'text' in self.apply_to_modality:
            self.text_encoder = self._build_text_encoder(text_feat_dim, latent_dim)
            self.text_decoder = self._build_text_decoder(latent_dim, text_feat_dim)
        # 新增音频编解码器初始化 ▼▼▼
        if 'audio' in self.apply_to_modality:
            self.audio_encoder = nn.Sequential(
                nn.Linear(audio_feat_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                ResidualBlock(1024),
                nn.Linear(1024, latent_dim)
            )
            self.audio_decoder = nn.Sequential(
                ResidualBlock(latent_dim),
                nn.Linear(latent_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Linear(1024, audio_feat_dim)
            )

    def _build_video_encoder(self, input_dim, latent_dim):
        return nn.Sequential(
            nn.Conv1d(input_dim, 1024, kernel_size=3, padding=1),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Conv1d(1024, latent_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )

    def _build_text_encoder(self, input_dim, latent_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            ResidualBlock(1024),  # 新增残差块
            nn.Linear(1024, latent_dim)
        )

    def _build_video_decoder(self, latent_dim, output_dim):
        return nn.Sequential(
            nn.Linear(latent_dim, 1024),
            ResidualBlock(1024),  # 视频重建残差块
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, output_dim)
        )

    def _build_text_decoder(self, latent_dim, output_dim):
        return nn.Sequential(
            ResidualBlock(latent_dim),  # 文本重建残差块
            nn.Linear(latent_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            ResidualBlock(2048),
            nn.Linear(2048, output_dim)
        )

    def forward(self, inputs):
        device = next(self.parameters()).device
        outputs = {}
        
        # 视频处理流程（添加噪声增强）
        if 'video' in self.apply_to_modality and 'video' in inputs:
            video_feat = self.noise_layer(inputs['video']).permute(0, 2, 1).to(device)
            video_encoded = self.video_encoder(video_feat).squeeze(2)
            outputs['video_encoded'] = video_encoded
            outputs['video_recon'] = self.video_decoder(video_encoded)
            
        # 文本处理流程（添加噪声增强）
        if 'text' in self.apply_to_modality and 'text' in inputs:
            text_feat = self.noise_layer(inputs['text'].to(device))
            text_encoded = self.text_encoder(text_feat)
            outputs['text_encoded'] = text_encoded
            outputs['text_recon'] = self.text_decoder(text_encoded)
            
        # 新增音频处理流程 ▼▼▼
        if 'audio' in self.apply_to_modality and 'audio' in inputs:
            audio_feat = self.noise_layer(inputs['audio'].to(device))
            audio_encoded = self.audio_encoder(audio_feat)
            outputs['audio_encoded'] = audio_encoded
            outputs['audio_recon'] = self.audio_decoder(audio_encoded)
            
        return outputs