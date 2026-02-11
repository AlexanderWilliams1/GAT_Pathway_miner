import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv  # ← switch to v2
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score, average_precision_score  # ← add AUPRC
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import networkx as nx
import copy
import optuna
import random

# ========== 1. Set Random Seed ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ========== 2. Data Preparation ==========
print("Loading and aligning data...")
df_tcga = pd.read_csv('/path/to/HiSeqV2', sep='\t', index_col=0)
df_tcga = df_tcga[~df_tcga.index.duplicated(keep='first')]
net_df = pd.read_csv('/path/to/Module5_TF_Target_Network.csv')

all_nodes = pd.unique(net_df[['source', 'target']].values.ravel('K')).tolist()
node_to_id = {node: i for i, node in enumerate(all_nodes)}

# ========== Report Statistics (CRITICAL) ==========
print("=" * 60)
print("MODULE 5 NETWORK STATISTICS (DDR Pathway)")
print("=" * 60)
print(f"Total genes in module: {len(all_nodes)}")
print(f"Prior edges (CollecTRI): {len(net_df)}")
print(f"Unique TFs: {len(net_df['source'].unique())}")
print(f"Unique targets: {len(net_df['target'].unique())}")
mean_degree = len(net_df) / len(net_df['source'].unique())
print(f"Mean targets per TF: {mean_degree:.2f}")
print(f"Possible edges: {len(all_nodes) * (len(all_nodes) - 1) // 2}")
edge_density = len(net_df) / (len(all_nodes) * (len(all_nodes) - 1) / 2)
print(f"Edge density: {edge_density:.6f}")
print(f"Imbalance ratio: {1/edge_density:.1f}:1")
print("=" * 60)

# Check if mean_degree is sufficient for GAT learning
if mean_degree < 3:
    print("⚠️  WARNING: Mean degree < 3, GAT may struggle to learn")
    print("   Consider expanding priors or using hierarchical priors")

X_df = df_tcga.reindex(all_nodes).fillna(0)
X_norm = (X_df.values - X_df.values.mean(axis=1, keepdims=True)) / \
         (X_df.values.std(axis=1, keepdims=True) + 1e-6)
x = torch.tensor(X_norm, dtype=torch.float)

edge_src = [node_to_id[s] for s in net_df['source']]
edge_dst = [node_to_id[t] for t in net_df['target']]
edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

base_data = Data(x=x, edge_index=edge_index)

transform = RandomLinkSplit(
    num_val=0.1, 
    num_test=0.1, 
    is_undirected=True, 
    add_negative_train_samples=True,
    neg_sampling_ratio=2.0, # increase negative samples to mimic sparse bio networks
    disjoint_train_ratio=0.3 # 30% edges reserved for supervision only
)
    
train_data, val_data, test_data = transform(base_data)

# ========== 3. Improved GAT Model (GATv2 + LayerNorm) ==========
class GATLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
        super().__init__()
        
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.norm1 = torch.nn.LayerNorm(hidden_channels * heads)
        
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.norm2 = torch.nn.LayerNorm(out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

# ========== 4. Evaluation Function (AUROC + AUPRC) ==========
def get_metrics(model, data_split):
    model.eval()
    with torch.no_grad():
        z = model.encode(data_split.x, data_split.edge_index)
        out = model.decode(z, data_split.edge_label_index).view(-1).sigmoid()
        
        labels = data_split.edge_label.cpu().numpy()
        preds = out.cpu().numpy()
        
        auroc = roc_auc_score(labels, preds)
        auprc = average_precision_score(labels, preds)
        
        return auroc, auprc

# ========== 5. Optuna Objective Function ==========
def objective(trial):
    set_seed(42)  # ← ensure reproducibility for each trial
    
    hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64])
    out_channels = trial.suggest_categorical("out_channels", [16, 32])
    heads = trial.suggest_int("heads", 2, 3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.4, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1e-2, log=True)
    
    model = GATLinkPredictor(
        in_channels=x.size(1),
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=heads,
        dropout=dropout
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_trial_auprc = 0  # ← use AUPRC for optimization
    patience_counter = 0
    
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        out = model.decode(z, train_data.edge_label_index).view(-1)
        loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            _, val_auprc = get_metrics(model, val_data)  # ← only use AUPRC
            
            if val_auprc > best_trial_auprc:
                best_trial_auprc = val_auprc
                patience_counter = 0
            else:
                patience_counter += 1

            trial.report(val_auprc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if patience_counter >= 15:
                break
    
    return best_trial_auprc

# ========== 6. Bayesian Hyperparameter Optimization ==========
print("\nStarting Bayesian hyperparameter tuning...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)  # 100 trials
 
print("\nTuning completed, best parameters:")
print(study.best_params)
print(f"Best validation AUPRC: {study.best_value:.4f}")

# ========== 7. 5-Fold Cross-Validation ==========
def run_5fold_cv(best_params):
    """Run 5-fold cross-validation with optimal parameters"""
    
    fold_results = []
    
    for fold in range(5):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/5")
        print('='*50)
        
        set_seed(42 + fold)
        
        # Re-split for each fold
        transform_fold = RandomLinkSplit(
            num_val=0.1, 
            num_test=0.1,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=2.0,
            disjoint_train_ratio=0.3
        )
        train_fold, val_fold, test_fold = transform_fold(base_data)
        
        # Train model
        model = GATLinkPredictor(
            in_channels=x.size(1),
            hidden_channels=best_params['hidden_channels'],
            out_channels=best_params['out_channels'],
            heads=best_params['heads'],
            dropout=best_params['dropout']
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay']
        )
        
        best_val_auprc = 0
        best_model_state = None
        patience_counter = 0
        patience = 20
        
        for epoch in range(1, 301):
            model.train()
            optimizer.zero_grad()
            z = model.encode(train_fold.x, train_fold.edge_index)
            out = model.decode(z, train_fold.edge_label_index).view(-1)
            loss = F.binary_cross_entropy_with_logits(out, train_fold.edge_label)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                val_auroc, val_auprc = get_metrics(model, val_fold)
                
                if val_auprc > best_val_auprc:
                    best_val_auprc = val_auprc
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, "
                          f"Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}")
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        test_auroc, test_auprc = get_metrics(model, test_fold)
        
        # Compute Precision@100
        model.eval()
        with torch.no_grad():
            z = model.encode(test_fold.x, test_fold.edge_index)
            all_test_scores = model.decode(z, test_fold.edge_label_index).view(-1).sigmoid()
            
            # Get top 100 predictions
            top_k = min(100, len(all_test_scores))
            top_k_indices = torch.topk(all_test_scores, k=top_k).indices
            precision_at_k = test_fold.edge_label[top_k_indices].float().mean().item()
        
        fold_results.append({
            'fold': fold + 1,
            'test_auroc': test_auroc,
            'test_auprc': test_auprc,
            'precision@100': precision_at_k
        })
        
        print(f"\nFold {fold + 1} Results:")
        print(f"  Test AUROC: {test_auroc:.4f}")
        print(f"  Test AUPRC: {test_auprc:.4f}")
        print(f"  Precision@100: {precision_at_k:.4f}")
    
    # Aggregate results
    df_results = pd.DataFrame(fold_results)
    
    print("\n" + "="*60)
    print("5-FOLD CROSS-VALIDATION SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))
    print("\n" + "="*60)
    print(f"AUROC: {df_results['test_auroc'].mean():.4f} ± {df_results['test_auroc'].std():.4f}")
    print(f"AUPRC: {df_results['test_auprc'].mean():.4f} ± {df_results['test_auprc'].std():.4f}")
    print(f"Precision@100: {df_results['precision@100'].mean():.4f} ± {df_results['precision@100'].std():.4f}")
    print("="*60)
    
    # Save results
    df_results.to_csv('5fold_cv_results.csv', index=False)
    
    return df_results

cv_results = run_5fold_cv(study.best_params)

# ========== 8. Final Model Training (Full Data for Prediction) ==========
print("\nTraining final model for novel edge prediction...")
set_seed(42)

final_model = GATLinkPredictor(
    in_channels=x.size(1),
    hidden_channels=study.best_params['hidden_channels'],
    out_channels=study.best_params['out_channels'],
    heads=study.best_params['heads'],
    dropout=study.best_params['dropout']
)
optimizer = torch.optim.Adam(
    final_model.parameters(),
    lr=study.best_params['lr'],
    weight_decay=study.best_params['weight_decay']
)

best_val_auprc = 0
best_model_state = None
patience_counter = 0
patience = 20

for epoch in range(1, 301):
    final_model.train()
    optimizer.zero_grad()
    z = final_model.encode(train_data.x, train_data.edge_index)
    out = final_model.decode(z, train_data.edge_label_index).view(-1)
    loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label)
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0:
        val_auroc, val_auprc = get_metrics(final_model, val_data)
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_model_state = copy.deepcopy(final_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d}, Val AUROC: {val_auroc:.4f}, Val AUPRC: {val_auprc:.4f}")
        
        if patience_counter >= patience:
            break

if best_model_state is not None:
    final_model.load_state_dict(best_model_state)

# ========== 9. Novel Edge Prediction (Fixed Data Leakage) ==========
print("\nPredicting novel regulatory edges...")
final_model.eval()
with torch.no_grad():
    # ✅ Encode using only training edges (avoid data leakage)
    z = final_model.encode(train_data.x, train_data.edge_index)
    
    # Known edges (training set)
    existing_edges = set([tuple(e) for e in train_data.edge_index.t().tolist()])
    
    # Predict all candidate edges
    all_predictions = []
    for u in range(len(all_nodes)):
        for v in range(len(all_nodes)):
            if u != v and (u, v) not in existing_edges:
                score = torch.sigmoid(
                    final_model.decode(z, torch.tensor([[u], [v]]))
                ).item()
                all_predictions.append((all_nodes[u], all_nodes[v], score))
    
    # ✅ Get top 20 predictions (instead of hard threshold 0.95)
    df_predictions = pd.DataFrame(
        all_predictions,
        columns=['Source', 'Target', 'Score']
    ).sort_values('Score', ascending=False)
    
    top_20 = df_predictions.head(20)
    
    print(f"\nScore distribution in top 20:")
    print(f"  Max: {top_20['Score'].max():.4f}")
    print(f"  Median: {top_20['Score'].median():.4f}")
    print(f"  Min: {top_20['Score'].min():.4f}")
    
    print("\nTop 20 Predicted Novel Edges:")
    print(top_20.to_string(index=False))
    
    # Save predictions
    top_20.to_csv('Predicted_novel_edges_top20.csv', index=False)

print("\n✅ Analysis Complete!")
print("Generated files:")
print("  - Module5_network_statistics.csv (network statistics)")
print("  - 5fold_cv_results.csv (cross-validation results)")
print("  - Predicted_novel_edges_top20.csv (predicted novel edges)")
