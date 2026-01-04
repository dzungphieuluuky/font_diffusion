import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


class MultiScaleTransformerBlock(nn.Module):
    """
    Transformer block with self-attention, cross-attention, and FFN.
    Used for extracting style features at each scale.
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8, ffn_dim: int = 2048):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Self-attention for style features
        self.self_attn = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True
        )
        
        # Cross-attention (for fusing source and target)
        self.cross_attn = nn.MultiheadAttention(
            feature_dim, num_heads, batch_first=True
        )
        
        # FFN layers
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, feature_dim),
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(feature_dim)
        self.ln2 = nn.LayerNorm(feature_dim)
        self.ln3 = nn.LayerNorm(feature_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len, feature_dim)
            key: (batch_size, seq_len, feature_dim)
            value: (batch_size, seq_len, feature_dim)
        
        Returns:
            output: (batch_size, seq_len, feature_dim)
        """
        # Self-attention
        self_attn_out, _ = self.self_attn(query, query, query)
        query = self.ln1(query + self_attn_out)
        
        # Cross-attention
        cross_attn_out, _ = self.cross_attn(query, key, value)
        query = self.ln2(query + cross_attn_out)
        
        # FFN
        ffn_out = self.ffn(query)
        query = self.ln3(query + ffn_out)
        
        return query


class StyleTransformationModule(nn.Module):
    """
    Font Style Transformation (FST) Module.
    
    Computes style transformation features between source and target fonts
    using multi-scale style features with transformer-based attention fusion.
    
    Following the paper equations (3-9):
    - Extract style features at multiple scales
    - Compute style differences L_xy = L_y - L_x
    - Apply self-attention fusion across scales
    - Use residual connection with learnable weight matrix
    """
    
    def __init__(
        self,
        num_scales: int = 4,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 2048,
    ):
        super().__init__()
        self.num_scales = num_scales
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Learnable queries for extracting style features at each scale (Equation 3-4)
        self.query_weights = nn.ParameterList([
            nn.Parameter(torch.randn(1, feature_dim) * 0.02)
            for _ in range(num_scales)
        ])
        
        # Key and Value projection weights for each scale (Equation 4-6)
        self.key_weights = nn.ParameterList([
            nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.02)
            for _ in range(num_scales)
        ])
        
        self.value_weights = nn.ParameterList([
            nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.02)
            for _ in range(num_scales)
        ])
        
        # Positional encoding for each scale
        self.positional_encodings = nn.ParameterList([
            nn.Parameter(torch.randn(1, feature_dim) * 0.02)
            for _ in range(num_scales)
        ])
        
        # Transformer blocks for multi-scale fusion (Equation 8)
        self.transformer_blocks = nn.ModuleList([
            MultiScaleTransformerBlock(
                feature_dim, num_heads, ffn_dim
            )
            for _ in range(num_scales)
        ])
        
        # MLP for channel adjustment (Equation 8)
        self.mlp_channel_adjust = nn.Sequential(
            nn.Linear(feature_dim * num_scales, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        
        # Learnable weight matrix for residual connection (Equation 9)
        self.residual_weight = nn.Parameter(
            torch.randn(feature_dim, feature_dim) * 0.02
        )
        
        # Style difference encoder
        self.style_difference_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Scaling factor for attention
        self.scale = math.sqrt(feature_dim)
    
    def extract_style_features(
        self,
        style_feature: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Extract style features at multiple scales using transformer blocks.
        
        Args:
            style_feature: (batch_size, seq_len, feature_dim) or 
                          (batch_size, feature_dim, height, width)
        
        Returns:
            List of extracted features at each scale
        """
        # Flatten spatial dimensions if needed
        if style_feature.dim() == 4:
            batch_size, channels, height, width = style_feature.shape
            style_feature = style_feature.permute(0, 2, 3, 1).reshape(
                batch_size, height * width, channels
            )
        
        batch_size, seq_len, feature_dim = style_feature.shape
        extracted_features = []
        
        # Extract features at each scale (Equations 3-6)
        for scale_idx in range(self.num_scales):
            # Learnable query (Equation 3)
            query = self.query_weights[scale_idx].expand(batch_size, -1, -1)
            
            # Project style features to key and value (Equation 4-6)
            key = style_feature @ self.key_weights[scale_idx]
            value = style_feature @ self.value_weights[scale_idx]
            
            # Add positional encoding
            query = query + self.positional_encodings[scale_idx]
            
            # Multi-head attention
            attn_output = self._scaled_dot_product_attention(query, key, value)
            
            # Apply transformer block (Equation 8)
            transformed = self.transformer_blocks[scale_idx](
                query, key, value
            )
            
            extracted_features.append(transformed)
        
        return extracted_features
    
    def _scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention (Equation 5).
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output
    
    def compute_style_difference(
        self,
        source_features: List[torch.Tensor],
        target_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Compute style differences at each scale (Equation 7).
        
        L_xy = L_y - L_x (difference between target and source)
        """
        style_differences = []
        for src_feat, tgt_feat in zip(source_features, target_features):
            # Element-wise difference
            diff = tgt_feat - src_feat
            style_differences.append(diff)
        
        return style_differences
    
    def fuse_style_differences(
        self,
        style_differences: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse style differences across scales using self-attention (Equation 8).
        """
        batch_size = style_differences[0].shape[0]
        
        # Concatenate differences along channel dimension
        concatenated = torch.cat(style_differences, dim=-1)  # (batch_size, seq_len, feature_dim * num_scales)
        
        # MLP to adjust channel size
        if concatenated.dim() == 3:
            # (batch_size, seq_len, feature_dim * num_scales) -> (batch_size, feature_dim)
            fused = self.mlp_channel_adjust(concatenated.mean(dim=1))
        else:
            fused = self.mlp_channel_adjust(concatenated)
        
        return fused
    
    def forward(
        self,
        source_style_feature: torch.Tensor,
        target_style_feature: torch.Tensor,
        source_content_feature: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute style transformation from source to target.
        
        Args:
            source_style_feature: (batch_size, feature_dim) or 
                                (batch_size, feature_dim, height, width)
            target_style_feature: Same shape as source_style_feature
            source_content_feature: Optional (batch_size, feature_dim) 
                                   for residual connection
        
        Returns:
            style_transform_feature: Transformation features (batch_size, feature_dim)
            style_diff: Fused style difference (batch_size, feature_dim)
        """
        # Extract multi-scale style features
        source_features = self.extract_style_features(source_style_feature)
        target_features = self.extract_style_features(target_style_feature)
        
        # Compute style differences at each scale (Equation 7)
        style_differences = self.compute_style_difference(
            source_features, target_features
        )
        
        # Fuse style differences (Equation 8)
        fused_style_diff = self.fuse_style_differences(style_differences)
        
        # Apply residual connection with learnable weight (Equation 9)
        # L_xy = [L_xy; f_y^s * W]
        if source_content_feature is not None:
            # Reshape if needed
            if source_content_feature.dim() > 2:
                source_content_flat = source_content_feature.reshape(
                    source_content_feature.shape[0], -1
                )
            else:
                source_content_flat = source_content_feature
            
            residual_term = source_content_flat @ self.residual_weight
            style_transform_feature = torch.cat(
                [fused_style_diff, residual_term], dim=-1
            )
        else:
            style_transform_feature = fused_style_diff
        
        # Encode style difference for later use
        concatenated = torch.cat(
            [
                source_style_feature.reshape(source_style_feature.shape[0], -1),
                target_style_feature.reshape(target_style_feature.shape[0], -1)
            ],
            dim=-1
        )
        style_diff_encoded = self.style_difference_encoder(concatenated)
        
        return style_transform_feature, style_diff_encoded


class FontDiffuserModel(ModelMixin, ConfigMixin):
    """Forward function for FontDiffuser with content encoder,
    style encoder, style transformation module, and unet.
    """

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
        style_transform_module: Optional[StyleTransformationModule] = None,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.style_transform_module = style_transform_module

    def forward(
        self,
        x_t,
        timesteps,
        style_images,
        content_images,
        content_encoder_downsample_size,
        source_style_images: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x_t: Noisy latent
            timesteps: Diffusion timesteps
            style_images: Target style images
            content_images: Content images
            content_encoder_downsample_size: Downsampling size
            source_style_images: Source style images (optional)
        
        Returns:
            noise_pred, offset_out_sum, style_transform_feature
        """
        # Extract target style features
        style_img_feature, _, style_residual_features = self.style_encoder(
            style_images
        )

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        # Get content features
        content_img_feature, content_residual_features = self.content_encoder(
            content_images
        )
        content_residual_features.append(content_img_feature)
        
        # Get reference content features from style image
        style_content_feature, style_content_res_features = self.content_encoder(
            style_images
        )
        style_content_res_features.append(style_content_feature)

        # ✅ Compute style transformation if source style provided
        style_transform_feature = None
        style_diff = None
        
        if source_style_images is not None and self.style_transform_module is not None:
            # Extract source style features
            source_style_img_feature, _, _ = self.style_encoder(
                source_style_images
            )
            
            # Compute style transformation (Equations 3-9)
            style_transform_feature, style_diff = self.style_transform_module(
                source_style_img_feature,
                style_img_feature,
                content_img_feature,
            )
        else:
            # Default style difference encoding
            style_diff = torch.zeros(batch_size, 256, device=style_img_feature.device)

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
            style_diff,
        ]

        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]

        return noise_pred, offset_out_sum, style_transform_feature


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuser with style transformation module."""

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
        style_transform_module: Optional[StyleTransformationModule] = None,
    ):
        super().__init__()
        self.unet = unet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.style_transform_module = style_transform_module

    def forward(
        self,
        x_t,
        timesteps,
        cond,
        content_encoder_downsample_size,
        version,
        source_cond: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Args:
            x_t: Noisy latent
            timesteps: Diffusion timesteps
            cond: Tuple of (content_images, target_style_images)
            content_encoder_downsample_size: Downsampling size
            version: Model version
            source_cond: Optional tuple (source_content, source_style)
        
        Returns:
            noise_pred, style_transform_feature
        """
        content_images = cond[0]
        style_images = cond[1]

        # Extract target style features
        style_img_feature, _, style_residual_features = self.style_encoder(
            style_images
        )

        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        # Get content features
        content_img_feature, content_residual_features = self.content_encoder(
            content_images
        )
        content_residual_features.append(content_img_feature)
        
        # Get reference content features
        style_content_feature, style_content_res_features = self.content_encoder(
            style_images
        )
        style_content_res_features.append(style_content_feature)

        # ✅ Compute style transformation
        style_transform_feature = None
        style_diff = None
        
        if source_cond is not None and self.style_transform_module is not None:
            source_content_images, source_style_images = source_cond
            
            # Extract source style features
            source_style_img_feature, _, _ = self.style_encoder(
                source_style_images
            )
            
            # Compute style transformation
            style_transform_feature, style_diff = self.style_transform_module(
                source_style_img_feature,
                style_img_feature,
                content_img_feature,
            )
        else:
            style_diff = torch.zeros(batch_size, 256, device=style_img_feature.device)

        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features,
            style_diff,
        ]

        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]

        return noise_pred, style_transform_feature