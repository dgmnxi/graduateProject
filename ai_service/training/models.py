"""
MLP, ResidualMLP, Transformer, GCN 모델 구현
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class BasePoseToBetaModel(nn.Module):
    """모든 모델의 베이스 클래스"""
    
    def __init__(self, input_size: int = 99, output_size: int = 10):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.model_name = self.__class__.__name__
    
    def forward(self, x):
        raise NotImplementedError


class ResidualMLPPoseToBeta(BasePoseToBetaModel):
    """
    Residual MLP 모델
    
    구조:
    - Residual Block들로 구성
    - 각 Block: Linear → BatchNorm → ReLU → Linear → BatchNorm → ReLU + Residual
    - 최종 Output MLP
    """
    
    def __init__(self, input_size: int = 99, output_size: int = 10,
                 hidden_sizes: list = None, dropout_rates: list = None,
                 num_blocks_per_layer: int = 1):
        super().__init__(input_size, output_size)
        
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]
        if dropout_rates is None:
            dropout_rates = [0.2, 0.2, 0.1]
        
        self.hidden_sizes = hidden_sizes
        self.dropout_rates = dropout_rates
        self.num_blocks_per_layer = num_blocks_per_layer
        
        # 첫 번째 레이어: input_size → hidden_sizes[0]
        layers = []
        prev_size = input_size
        
        for layer_idx, hidden_size in enumerate(hidden_sizes):
            # Residual Block
            for block_idx in range(num_blocks_per_layer):
                block = ResidualBlock(
                    in_features=prev_size,
                    hidden_features=hidden_size,
                    out_features=hidden_size if block_idx > 0 else hidden_size,
                    dropout=dropout_rates[layer_idx] if layer_idx < len(dropout_rates) else 0.1
                )
                layers.append(block)
                prev_size = hidden_size
        
        self.residual_blocks = nn.Sequential(*layers)
        
        # Output MLP: hidden_sizes[-1] → output_size
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 99)
        
        Returns:
            (batch_size, 10)
        """
        x = self.residual_blocks(x)
        x = self.output_layer(x)
        return x


class ResidualBlock(nn.Module):
    """Residual Block: Linear → BN → ReLU → Linear → BN + Residual"""
    
    def __init__(self, in_features: int, hidden_features: int, 
                 out_features: int, dropout: float = 0.1):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 첫 번째 경로
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # 두 번째 경로
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout2 = nn.Dropout(dropout)
        
        # 차원 불일치 시 projection
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
            self.projection_bn = nn.BatchNorm1d(out_features)
        else:
            self.projection = None
    
    def forward(self, x):
        # Residual path
        residual = x
        
        # Main path
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Add residual
        if self.projection is not None:
            residual = self.projection(residual)
            residual = self.projection_bn(residual)
        
        out = out + residual
        out = self.relu(out)
        
        return out


class MLPPoseToBeta(BasePoseToBetaModel):
    """
    단순 MLP 모델

    구조:
    - Linear → BatchNorm → ReLU → Dropout 반복
    - 기본 구조: 99 → 256 → 128 → 64 → 10
    """

    def __init__(self, input_size: int = 99, output_size: int = 10,
                 hidden_sizes: list = None, dropout_rates: list = None):
        super().__init__(input_size, output_size)

        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]
        if dropout_rates is None:
            dropout_rates = [0.2, 0.2, 0.1]

        self.hidden_sizes = hidden_sizes
        self.dropout_rates = dropout_rates

        layers = []
        prev_size = input_size

        for idx, hidden_size in enumerate(hidden_sizes):
            dropout = dropout_rates[idx] if idx < len(dropout_rates) else 0.1
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch_size, 99)

        Returns:
            (batch_size, 10)
        """
        return self.mlp(x)


class PositionalEncoding(nn.Module):
    """Transformer용 sinusoidal positional encoding"""

    def __init__(self, embed_dim: int, max_len: int = 99):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerPoseToBeta(BasePoseToBetaModel):
    """
    Transformer 기반 모델

    구조:
    - 입력 (batch, 99)을 99개 토큰으로 취급
    - 토큰 임베딩 + Positional Encoding
    - Transformer Encoder stack
    - 전역 평균 풀링 후 Output MLP
    """

    def __init__(self, input_size: int = 99, output_size: int = 10,
                 embed_dim: int = 128, num_heads: int = 8,
                 num_layers: int = 4, feedforward_dim: int = 256,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__(input_size, output_size)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feedforward_dim = feedforward_dim

        # (batch, 99) -> (batch, 99, 1) -> (batch, 99, embed_dim)
        self.token_embedding = nn.Linear(1, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=input_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output MLP: 128 -> 64 -> 32 -> 10
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, 99)

        Returns:
            (batch_size, 10)
        """
        # 각 좌표 스칼라를 하나의 토큰으로 사용
        x = x.unsqueeze(-1)  # (batch_size, 99, 1)
        x = self.token_embedding(x)  # (batch_size, 99, embed_dim)
        x = self.positional_encoding(x)

        x = self.encoder(x)  # (batch_size, 99, embed_dim)
        x = x.mean(dim=1)  # 전역 평균 풀링: (batch_size, embed_dim)

        return self.output_mlp(x)


class GCNPoseToBeta(BasePoseToBetaModel):
    """
    Graph Convolutional Network 모델
    
    구조:
    - 33개 관절을 노드로 하는 그래프
    - MediaPipe 골격 구조를 엣지로 정의
    - GCN 레이어로 관절 특징 집계
    - 최종 MLP로 beta 예측
    """
    
    def __init__(self, input_size: int = 99, output_size: int = 10,
                 feat_dim: int = 64, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__(input_size, output_size)
        
        assert input_size == 99, "Input size must be 99 (33 joints × 3 coords)"
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_joints = 33
        
        # MediaPipe 골격 그래프 정의
        self.register_buffer('adjacency_matrix', self._build_mediapipe_graph())
        
        # 관절 좌표 → 특징 변환
        self.joint_embedding = nn.Linear(3, feat_dim)
        
        # GCN 레이어
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(feat_dim if i == 0 else hidden_dim, 
                          hidden_dim, dropout)
            for i in range(num_layers)
        ])
        
        # 노드 특징 집계 (평균 풀링)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output MLP: hidden_dim → output_size
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
    
    def _build_mediapipe_graph(self):
        """
        MediaPipe 33개 관절의 연결 구조
        
        연결 구조 (bone connections):
        - 0: nose
        - 1-4: left_eye, right_eye (5개)
        - 5-8: left_ear, right_ear (4개)
        - 9-10: 입 (2개)
        - 11-32: 몸 (22개)
        """
        # 인접 행렬 (Adjacency Matrix)
        adj = np.zeros((33, 33))
        
        # MediaPipe 연결 구조
        connections = [
            # 머리 연결
            (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
            (0, 7), (7, 8), (8, 9), (9, 10),
            # 팔다리 연결
            (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (15, 22), (16, 22),
            (11, 23), (23, 24),
            (16, 17), (17, 18), (18, 19), (19, 20),
            (12, 24), (24, 25),
            (23, 25),
            (11, 13), (12, 14),
            (23, 11), (24, 12),
            # 몸통
            (11, 23), (12, 24),
            (23, 25), (24, 26),
            (25, 27), (26, 28),
            (27, 29), (28, 30),
            (29, 31), (30, 32),
        ]
        
        for src, dst in connections:
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0  # 무방향 그래프
        
        # 자기 루프 추가 (self-loop)
        np.fill_diagonal(adj, 1.0)
        
        # Degree normalization: D^{-1/2} A D^{-1/2}
        degree = np.sum(adj, axis=1)
        degree_inv_sqrt = np.diag(np.power(degree, -0.5))
        adj_normalized = degree_inv_sqrt @ adj @ degree_inv_sqrt
        
        return torch.FloatTensor(adj_normalized)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 99) - 33개 관절의 x, y, z 좌표
        
        Returns:
            (batch_size, 10) - beta 값
        """
        batch_size = x.size(0)
        
        # (batch_size, 99) → (batch_size, 33, 3)
        x = x.view(batch_size, self.num_joints, 3)
        
        # (batch_size, 33, 3) → (batch_size, 33, feat_dim)
        x = self.joint_embedding(x)
        
        # GCN 레이어 적용
        adj = self.adjacency_matrix.to(x.device)
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, adj)
        
        # 노드 특징 집계: (batch_size, 33, hidden_dim) → (batch_size, hidden_dim)
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, 33)
        x = self.global_pool(x)  # (batch_size, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch_size, hidden_dim)
        
        # Output MLP
        x = self.output_mlp(x)
        
        return x


class GraphConvLayer(nn.Module):
    """Graph Convolution Layer"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        """
        Args:
            x: (batch_size, num_nodes, in_channels)
            adj: (num_nodes, num_nodes)
        
        Returns:
            (batch_size, num_nodes, out_channels)
        """
        # Graph convolution: A @ X@ W
        batch_size, num_nodes, in_channels = x.shape
        
        # (batch_size, num_nodes, in_channels) × (in_channels, out_channels)
        x = self.linear(x)  # (batch_size, num_nodes, out_channels)
        
        # (num_nodes, num_nodes) × (batch_size, num_nodes, out_channels)^T
        # = (batch_size, num_nodes, out_channels)
        x = torch.matmul(adj.unsqueeze(0), x)
        
        # BatchNorm (channels 기준)
        x = x.view(-1, x.size(-1))
        x = self.bn(x)
        x = x.view(batch_size, num_nodes, -1)
        
        # Activation
        x = self.relu(x)
        x = self.dropout(x)
        
        return x


if __name__ == "__main__":
    # 모델 테스트
    batch_size = 32
    input_size = 99
    
    # ResidualMLP 테스트
    model_resmlp = ResidualMLPPoseToBeta(
        input_size=input_size,
        output_size=10,
        hidden_sizes=[256, 128, 64],
        dropout_rates=[0.2, 0.2, 0.1]
    )
    
    x = torch.randn(batch_size, input_size)

    # MLP 테스트
    model_mlp = MLPPoseToBeta(
        input_size=input_size,
        output_size=10,
        hidden_sizes=[256, 128, 64],
        dropout_rates=[0.2, 0.2, 0.1]
    )
    y_mlp = model_mlp(x)
    print(f"MLP output shape: {y_mlp.shape}")  # (32, 10)

    y_resmlp = model_resmlp(x)
    print(f"ResidualMLP output shape: {y_resmlp.shape}")  # (32, 10)

    # Transformer 테스트
    model_transformer = TransformerPoseToBeta(
        input_size=input_size,
        output_size=10,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        feedforward_dim=256,
        dropout=0.1,
        activation="gelu"
    )
    y_transformer = model_transformer(x)
    print(f"Transformer output shape: {y_transformer.shape}")  # (32, 10)
    
    # GCN 테스트
    model_gcn = GCNPoseToBeta(
        input_size=input_size,
        output_size=10,
        feat_dim=64,
        hidden_dim=128,
        num_layers=3
    )
    
    y_gcn = model_gcn(x)
    print(f"GCN output shape: {y_gcn.shape}")  # (32, 10)
    
    # 파라미터 수 확인
    print(f"\nMLP parameters: {sum(p.numel() for p in model_mlp.parameters()):,}")
    print(f"ResidualMLP parameters: {sum(p.numel() for p in model_resmlp.parameters()):,}")
    print(f"Transformer parameters: {sum(p.numel() for p in model_transformer.parameters()):,}")
    print(f"GCN parameters: {sum(p.numel() for p in model_gcn.parameters()):,}")