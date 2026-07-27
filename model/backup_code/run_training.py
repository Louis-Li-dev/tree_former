import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from r_tree_forecaster import SpatialRTree, CHTPopulationDataset, HierarchicalSTGNN

def evaluate(model, dataloader, device, scaler):
    model.eval()
    losses = []
    
    # Pre-extract scaler parameters to numpy/tensors
    mean = torch.tensor(scaler['mean'], device=device) # (N_grid, 1)
    std = torch.tensor(scaler['std'], device=device)   # (N_grid, 1)
    
    mae_list = []
    rmse_list = []
    mape_list = []
    
    with torch.no_grad():
        for x, y in dataloader:
            # x shape: (B, N, T_in)
            # y shape: (B, N, T_out)
            x, y = x.to(device), y.to(device)
            
            # Predict
            pred = model(x) # (B, N, 1)
            
            # Compute loss in scaled domain
            loss = nn.functional.mse_loss(pred, y)
            losses.append(loss.item())
            
            # Denormalize for physical metrics
            # pred: (B, N, 1), y: (B, N, 1)
            # scaler mean/std shape: (N, 1) -> unsqueeze to match batch
            pred_denorm = pred * std.unsqueeze(0) + mean.unsqueeze(0)
            y_denorm = y * std.unsqueeze(0) + mean.unsqueeze(0)
            
            pred_np = pred_denorm.cpu().numpy()
            y_np = y_denorm.cpu().numpy()
            
            # Absolute Error
            abs_err = np.abs(pred_np - y_np)
            mae_list.append(np.mean(abs_err))
            rmse_list.append(np.mean(abs_err ** 2))
            
            # MAPE (avoid division by zero)
            mask = y_np > 1.0 # only compute MAPE where actual population is significant
            if np.sum(mask) > 0:
                mape = np.mean(abs_err[mask] / y_np[mask])
                mape_list.append(mape)
                
    mean_val_loss = np.mean(losses)
    mean_mae = np.mean(mae_list)
    mean_rmse = np.sqrt(np.mean(rmse_list))
    mean_mape = np.mean(mape_list) if len(mape_list) > 0 else 0.0
    
    return mean_val_loss, mean_mae, mean_rmse, mean_mape

def main():
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load W_global spatial adjacency matrix
    print("Loading W_global.npy...")
    W_sparse = np.load('W_global.npy', allow_pickle=True).item()
    print(f"W_global loaded. Shape: {W_sparse.shape}")
    
    # 2. Build Spatial R-Tree Hierarchy
    print("Building Spatial R-Tree from coordinates...")
    df_grid = pd.read_csv('2019_hourly_pop_byCHT.csv', usecols=['Grid'])
    coords = np.array([[float(val) for val in g.split('_')] for g in df_grid['Grid'].values], dtype=np.float32)
    
    # Build tree with capacity=4 (4 child nodes per parent maximum)
    rtree = SpatialRTree(capacity=4)
    levels, agg_matrices = rtree.build(coords)
    
    # Number of nodes at each level: [N_leaf, N_level_1, ..., 1]
    num_nodes_per_level = [len(level) for level in levels]
    print(f"Hierarchy levels node counts: {num_nodes_per_level}")
    
    # 3. Initialize Datasets
    print("Preparing train, validation, and test datasets...")
    lookback = 24 # Use past 24 hours
    horizon = 1   # Predict next 1 hour
    
    train_dataset = CHTPopulationDataset(
        csv_path='2019_hourly_pop_byCHT.csv',
        lookback_window=lookback,
        horizon=horizon,
        split='train'
    )
    scaler = train_dataset.scaler
    
    val_dataset = CHTPopulationDataset(
        csv_path='2019_hourly_pop_byCHT.csv',
        lookback_window=lookback,
        horizon=horizon,
        split='val',
        scaler=scaler
    )
    
    test_dataset = CHTPopulationDataset(
        csv_path='2019_hourly_pop_byCHT.csv',
        lookback_window=lookback,
        horizon=horizon,
        split='test',
        scaler=scaler
    )
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset split size - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 4. Instantiate Hierarchical STGNN Model
    print("Initializing HierarchicalSTGNN model...")
    model = HierarchicalSTGNN(
        num_nodes_per_level=num_nodes_per_level,
        aggregation_matrices=agg_matrices,
        adj_matrix=W_sparse,
        in_dim=lookback,
        hidden_dim=32, # Use 32 hidden units to balance performance and speed
        out_dim=horizon
    ).to(device)
    
    # 5. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 6. Training Loop (Run 5 epochs for demonstration)
    epochs = 5
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        start_time = time.time()
        train_losses = []
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device) # x: (B, N, T_in), y: (B, N, T_out)
            
            optimizer.zero_grad()
            pred = model(x) # (B, N, T_out)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")
                
        elapsed = time.time() - start_time
        mean_train_loss = np.mean(train_losses)
        
        # Evaluate on validation set
        val_loss, val_mae, val_rmse, val_mape = evaluate(model, val_loader, device, scaler)
        
        print(f"=== Epoch {epoch} Summary ===")
        print(f"Train Loss (Scaled MSE): {mean_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f} | Val MAPE: {val_mape*100:.2f}%")
        print(f"Time taken: {elapsed:.1f} seconds\n")
        
    # 7. Final Test Evaluation
    print("Evaluating model on the test dataset...")
    test_loss, test_mae, test_rmse, test_mape = evaluate(model, test_loader, device, scaler)
    print("=== Final Test Results ===")
    print(f"Test Loss (Scaled MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.2f} | Test RMSE: {test_rmse:.2f} | Test MAPE: {test_mape*100:.2f}%")

if __name__ == '__main__':
    main()
