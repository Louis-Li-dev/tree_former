import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SpatialRTree:
    """
    Sort-Tile-Recursive (STR) spatial R-Tree builder.
    Constructs a spatial hierarchy bottom-up from 2D coordinates.
    Generates binary aggregation matrices A_list where A_list[l] is of shape (N_parent, N_child)
    mapping children at level l to their parents at level l+1.
    """
    def __init__(self, capacity=4):
        self.capacity = capacity
        self.levels = []  # List of levels. Each level is a list of node dicts.
        self.aggregation_matrices = [] # List of matrices A_list

    def build(self, coords):
        """
        coords: numpy array of shape (N_leaf, 2), coordinates (lon, lat) of leaf nodes.
        """
        self.levels = []
        self.aggregation_matrices = []
        
        N_leaf = len(coords)
        current_nodes = []
        for i, c in enumerate(coords):
            current_nodes.append({
                'bbox': [c[0], c[1], c[0], c[1]], # [min_lon, min_lat, max_lon, max_lat]
                'children': [i],
                'centroid': c
            })
        self.levels.append(current_nodes)
        
        while len(self.levels[-1]) > 1:
            prev_level = self.levels[-1]
            num_nodes = len(prev_level)
            
            P = int(np.ceil(num_nodes / self.capacity))
            if P == 1:
                parent_groups = [list(range(num_nodes))]
            else:
                S = int(np.ceil(np.sqrt(P)))
                centroids = np.array([n['centroid'] for n in prev_level])
                # Sort along X coordinate (longitude)
                x_sorted_idx = np.argsort(centroids[:, 0])
                
                parent_groups = []
                slice_size = int(np.ceil(num_nodes / S))
                for i in range(0, num_nodes, slice_size):
                    x_slice_idx = x_sorted_idx[i:i+slice_size]
                    # Sort along Y coordinate (latitude) inside the slice
                    y_sorted_slice = sorted(x_slice_idx, key=lambda idx: centroids[idx, 1])
                    for j in range(0, len(y_sorted_slice), self.capacity):
                        parent_groups.append(y_sorted_slice[j:j+self.capacity])
            
            parents = []
            # Construct aggregation matrix for this level transition
            A = np.zeros((len(parent_groups), num_nodes), dtype=np.float32)
            
            for p_idx, group in enumerate(parent_groups):
                group_bboxes = np.array([prev_level[idx]['bbox'] for idx in group])
                group_centroids = np.array([prev_level[idx]['centroid'] for idx in group])
                
                min_x = np.min(group_bboxes[:, 0])
                min_y = np.min(group_bboxes[:, 1])
                max_x = np.max(group_bboxes[:, 2])
                max_y = np.max(group_bboxes[:, 3])
                
                centroid = np.mean(group_centroids, axis=0)
                parents.append({
                    'bbox': [min_x, min_y, max_x, max_y],
                    'children': group,
                    'centroid': centroid
                })
                # Set mapping in aggregation matrix
                # Here we use Mean Pooling weight (1.0 / group_size)
                for c_idx in group:
                    A[p_idx, c_idx] = 1.0 / len(group)
            
            self.levels.append(parents)
            self.aggregation_matrices.append(A)
            
            print(f"R-Tree Level {len(self.levels)-1} built: {len(parents)} nodes")
            
        return self.levels, self.aggregation_matrices


class CHTPopulationDataset(Dataset):
    """
    Dataset to load and preprocess the 2019 hourly population data.
    Input window: lookback_window hours
    Target window: horizon hours (default 1)
    """
    def __init__(self, csv_path, lookback_window=24, horizon=1, split='train', train_ratio=0.7, val_ratio=0.1, scaler=None):
        df = pd.read_csv(csv_path)
        self.grids = df['Grid'].values
        # Parse grid coordinates
        self.coords = np.array([[float(x) for x in g.split('_')] for g in self.grids], dtype=np.float32)
        
        # Load hourly time series data
        # Skip the 'Grid' column
        data = df.iloc[:, 1:].values.astype(np.float32) # shape: (N_grid, T)
        
        self.N_grid, self.T = data.shape
        
        # Split time-series data chronologically
        train_end = int(self.T * train_ratio)
        val_end = int(self.T * (train_ratio + val_ratio))
        
        if split == 'train':
            split_data = data[:, :train_end]
        elif split == 'val':
            split_data = data[:, train_end:val_end]
        else: # test
            split_data = data[:, val_end:]
            
        # Standardize using training scaler parameters
        if split == 'train':
            mean = np.mean(split_data, axis=1, keepdims=True)
            std = np.std(split_data, axis=1, keepdims=True) + 1e-5
            self.scaler = {'mean': mean, 'std': std}
        else:
            assert scaler is not None, "Scaler must be provided for validation and test splits!"
            self.scaler = scaler
            
        # Apply scaling
        self.scaled_data = (split_data - self.scaler['mean']) / self.scaler['std']
        
        self.lookback_window = lookback_window
        self.horizon = horizon
        self.num_samples = self.scaled_data.shape[1] - lookback_window - horizon + 1

    def __len__(self):
        return max(0, self.num_samples)

    def __getitem__(self, idx):
        # Extract input window and target window
        # shape of input: (lookback_window, N_grid) -> transposed to (N_grid, lookback_window)
        x = self.scaled_data[:, idx : idx + self.lookback_window]
        y = self.scaled_data[:, idx + self.lookback_window : idx + self.lookback_window + self.horizon]
        
        # We output shapes: (N_grid, lookback_window) and (N_grid, horizon)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class SpatialGraphConv(nn.Module):
    """
    Simple Graph Convolution (GCN) layer operating on the W_global spatial adjacency matrix.
    """
    def __init__(self, in_features, out_features):
        super(SpatialGraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x: shape (Batch, N, in_features)
        adj: normalized adjacency matrix, shape (N, N)
        """
        # Linear transform
        support = torch.matmul(x, self.weight) # (B, N, out_features)
        # Message passing
        output = torch.matmul(adj, support) # (B, N, out_features)
        return output + self.bias


class HierarchicalSTGNN(nn.Module):
    """
    Hierarchical Spatio-Temporal Graph Neural Network.
    Integrates spatial graph convolutions on leaf level grids,
    with bottom-up tree aggregation, multi-level temporal modeling (GRU),
    and top-down feedback context fusion.
    """
    def __init__(self, num_nodes_per_level, aggregation_matrices, adj_matrix, 
                 in_dim=24, hidden_dim=64, out_dim=1):
        super(HierarchicalSTGNN, self).__init__()
        self.levels_nodes = num_nodes_per_level
        self.num_levels = len(num_nodes_per_level)
        
        # Save R-Tree transition aggregation matrices (as buffers)
        # A_list[l] matches level l nodes to level l+1 nodes
        self.A_list = nn.ParameterList([
            nn.Parameter(torch.tensor(A, dtype=torch.float32), requires_grad=False) 
            for A in aggregation_matrices
        ])
        
        # Save Normalized Adjacency matrix for leaf level GNN
        self.adj = nn.Parameter(self._normalize_adj(adj_matrix), requires_grad=False)
        
        # 1. Input Linear encoder (on leaf nodes)
        self.leaf_encoder = nn.Linear(in_dim, hidden_dim)
        
        # 2. Leaf level Spatial GNN
        self.leaf_gnn = SpatialGraphConv(hidden_dim, hidden_dim)
        
        # 3. Temporal encoders (GRU) at each level of the tree hierarchy
        self.temporal_encoders = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, batch_first=True) for _ in range(self.num_levels)
        ])
        
        # 4. Top-Down fusion layers
        # For level l, fuses own level features and broadcasted features from level l+1
        self.top_down_linears = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(self.num_levels - 1)
        ])
        
        # 5. Output forecaster (on leaf nodes)
        self.forecaster = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )
        
        self.relu = nn.ReLU()

    def _normalize_adj(self, W):
        """
        Normalize adjacency matrix: D^-1/2 * (W + I) * D^-1/2
        """
        # Convert csr_matrix to dense numpy array if needed
        if not isinstance(W, np.ndarray):
            W = W.toarray()
        
        # Self loops
        A_tilde = W + np.eye(W.shape[0])
        # Degree matrix
        d = np.sum(A_tilde, axis=1)
        d_inv_sqrt = np.power(d, -0.5, where=d>0)
        d_inv_sqrt[d == 0] = 0.0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        
        # Normalized matrix
        normalized = D_inv_sqrt.dot(A_tilde).dot(D_inv_sqrt)
        return torch.tensor(normalized, dtype=torch.float32)

    def forward(self, x):
        """
        x: Input tensor of shape (Batch, N_leaf, lookback_window)
        """
        batch_size = x.size(0)
        N_leaf = x.size(1)
        lookback_window = x.size(2)
        
        # 1. Encode leaf grid population time series
        # Shape: (Batch * N_leaf, lookback_window) -> (Batch * N_leaf, hidden_dim)
        h_leaf = self.leaf_encoder(x) # (B, N_leaf, hidden_dim)
        
        # 2. Local spatial message passing using Graph Convolutions (W_global)
        h_leaf = self.relu(self.leaf_gnn(h_leaf, self.adj)) # (B, N_leaf, hidden_dim)
        
        # 3. Bottom-Up Aggregation along the Spatial R-Tree
        # features_per_level[l] contains representations of level l nodes
        features_per_level = [h_leaf]
        
        for l in range(self.num_levels - 1):
            A = self.A_list[l] # shape: (N_parent, N_child)
            # Aggregate child node features to parent node features
            # A maps (N_parent, N_child) * (Batch, N_child, hidden_dim) -> (Batch, N_parent, hidden_dim)
            h_parent = torch.matmul(A, features_per_level[-1])
            features_per_level.append(h_parent)
            
        # 4. Temporal Modeling at each level of the hierarchy
        updated_features = []
        for l in range(self.num_levels):
            h_level = features_per_level[l] # (B, N_nodes_l, hidden_dim)
            
            # Reshape for GRU: (B * N_nodes_l, 1, hidden_dim)
            h_seq = h_level.view(-1, 1, h_level.size(-1))
            _, h_state = self.temporal_encoders[l](h_seq) # h_state shape: (1, B * N_nodes_l, hidden_dim)
            
            h_updated = h_state.squeeze(0).view(batch_size, -1, h_level.size(-1))
            updated_features.append(h_updated)
            
        # 5. Top-Down Feedback and Guidance
        # Diffuse macro-level temporal context back down to fine-grained nodes
        h_down = updated_features[-1] # Start with root level features
        
        for l in range(self.num_levels - 2, -1, -1):
            A = self.A_list[l] # shape: (N_parent, N_child)
            # Broadcast parent features back to children
            # A_T shape: (N_child, N_parent)
            h_broadcast = torch.matmul(A.t(), h_down)
            
            # Fuse broadcasted parent features with local features
            h_local = updated_features[l]
            # Concatenate local and parent guidance features
            fused = torch.cat([h_local, h_broadcast], dim=-1) # (B, N_child, hidden_dim * 2)
            h_down = self.relu(self.top_down_linears[l](fused))
            
        # 6. Predict next-step grid populations
        # h_down is the leaf level feature after top-down fusion
        output = self.forecaster(h_down) # (B, N_leaf, 1)
        return output
