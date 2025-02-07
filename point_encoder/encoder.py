import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VoxelizationLayer(nn.Module):
    def __init__(self, voxel_size, grid_size, max_points_per_pillar, in_channels):
        super().__init__()
        self.voxel_size = voxel_size  # (x, y) resolution
        self.grid_size = grid_size  # (H, W) spatial resolution
        self.max_points_per_pillar = max_points_per_pillar
        self.in_channels = in_channels

    def forward(self, all_points, batch_indices):
        # all_points = torch.cat(point_clouds, dim=0)
        # batch_indices = torch.arange(len(batch_offsets), device=all_points.device).repeat_interleave(
        #     batch_offsets[1:] - batch_offsets[:-1])

        xy_indices = (all_points[:, :2] / torch.tensor(self.voxel_size, device=all_points.device)).long()
        grid_size_tensor = torch.tensor(self.grid_size, device=all_points.device)
        xy_indices = torch.clamp(xy_indices, min=0, max=grid_size_tensor - 1)

        voxel_features = torch.zeros(
            (len(point_clouds), self.grid_size[0], self.grid_size[1], self.max_points_per_pillar, self.in_channels),
            device=all_points.device)
        voxel_counts = torch.zeros((len(point_clouds), self.grid_size[0], self.grid_size[1]), dtype=torch.long,
                                   device=all_points.device)

        flat_indices = batch_indices * self.grid_size[0] * self.grid_size[1] + xy_indices[:, 0] * self.grid_size[
            1] + xy_indices[:, 1]
        unique_indices, inverse_indices = torch.unique(flat_indices, return_inverse=True)

        counts = torch.zeros_like(unique_indices, device=all_points.device, dtype=torch.long)
        counts.scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.long))

        mask = counts[inverse_indices] <= self.max_points_per_pillar
        valid_points = all_points[mask]
        valid_indices = inverse_indices[mask]

        sorted_indices = valid_indices.argsort()
        valid_points = valid_points[sorted_indices]
        valid_indices = valid_indices[sorted_indices]

        voxel_features.view(-1, self.max_points_per_pillar, self.in_channels)[valid_indices] = valid_points

        return voxel_features, voxel_counts


class PointNetPillarFeatureNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=64):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x * mask.unsqueeze(-1)  # Mask out empty points
        return torch.max(x, dim=3)[0]  # Max pooling along height


class TransformerFeatureEnhancer(nn.Module):
    def __init__(self, feature_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads),
            num_layers=num_layers
        )

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H * W, C).permute(1, 0, 2)  # Reshape for transformer
        x = self.transformer(x)
        x = x.permute(1, 0, 2).view(B, H, W, C)  # Reshape back
        return x


class PointPillarsEncoder(nn.Module):
    def __init__(self, voxel_size=(0.2, 0.2), grid_size=(128, 128), max_points_per_pillar=100, feature_dim=64):
        super().__init__()
        self.voxel_layer = VoxelizationLayer(voxel_size, grid_size, max_points_per_pillar, in_channels=4)
        self.pillar_feature_net = PointNetPillarFeatureNet(in_channels=4, out_channels=feature_dim)
        self.transformer = TransformerFeatureEnhancer(feature_dim)

    def forward(self, point_clouds, batch_offsets):
        # batch_offsets = torch.tensor([0] + [pc.shape[0] for pc in point_clouds], device=point_clouds[0].device).cumsum(
        #     0)
        voxel_features, voxel_counts = self.voxel_layer(point_clouds, batch_offsets)
        mask = (voxel_counts > 0).float().unsqueeze(-1)  # Create a mask for non-empty pillars
        pillar_features = self.pillar_feature_net(voxel_features, mask)
        enhanced_features = self.transformer(pillar_features)
        return enhanced_features  # Output: (batch, H, W, C)


# # Example usage
# if __name__ == "__main__":
#     model = PointPillarsEncoder()
#     dummy_point_clouds = [torch.rand((100000, 4)), torch.rand((50000, 4))]  # Two different-sized point clouds
#     output = model(dummy_point_clouds)
#     print(output.shape)  # Expected: (batch, H, W, C)
