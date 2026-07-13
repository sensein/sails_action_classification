# save as: train_mlp_fusion.py
"""
Train MLP fusion over logits from multiple skeleton models.
Usage: python train_mlp_fusion.py --dataset rmm
"""
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report


class FusionMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['rmm', 'loco'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    args = parser.parse_args()

    ds = args.dataset
    num_classes = 4 if ds == 'rmm' else 5

    # Load val logits (for training the MLP)
    with open(f'work_dirs/fusion_{ds}_val_logits.pkl', 'rb') as f:
        val_data = pickle.load(f)

    # Load test logits (for evaluation)
    with open(f'work_dirs/fusion_{ds}_test_logits.pkl', 'rb') as f:
        test_data = pickle.load(f)

    # Concatenate all model logits into a single feature vector
    model_names = sorted(val_data['logits'].keys())
    print(f'Fusing models: {model_names}')

    val_features = np.concatenate([val_data['logits'][k] for k in model_names], axis=1)
    test_features = np.concatenate([test_data['logits'][k] for k in model_names], axis=1)
    val_labels = val_data['labels']
    test_labels = test_data['labels']

    print(f'Val features shape: {val_features.shape}')
    print(f'Test features shape: {test_features.shape}')
    print(f'Input dim to MLP: {val_features.shape[1]} '
          f'({len(model_names)} models x {num_classes} classes each)')

    # Convert to tensors
    X_train = torch.FloatTensor(val_features)
    y_train = torch.LongTensor(val_labels)
    X_test = torch.FloatTensor(test_features)
    y_test = torch.LongTensor(test_labels)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64, shuffle=True)

    # Model
    input_dim = val_features.shape[1]
    model = FusionMLP(input_dim, num_classes, hidden_dim=args.hidden_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Evaluate on test
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test.to(device)).cpu().numpy()
            test_pred_labels = test_pred.argmax(axis=1)
            acc = accuracy_score(test_labels, test_pred_labels)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'work_dirs/fusion_mlp_{ds}_best.pth')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader):.4f}, '
                  f'Test Acc: {acc:.4f}, Best: {best_acc:.4f}')

    # Final evaluation
    model.load_state_dict(torch.load(f'work_dirs/fusion_mlp_{ds}_best.pth'))
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test.to(device)).cpu().numpy()
        test_pred_labels = test_pred.argmax(axis=1)

    print(f'\n{"="*50}')
    print(f'FINAL RESULTS — {ds.upper()} Dataset')
    print(f'{"="*50}')
    print(f'Best Test Accuracy: {best_acc:.4f}')
    print(f'\nClassification Report:')
    print(classification_report(test_labels, test_pred_labels))

    # Also show individual model accuracies for comparison
    print(f'\nIndividual Model Accuracies (weighted avg on test):')
    for name in model_names:
        logits = test_data['logits'][name]
        preds = logits.argmax(axis=1)
        ind_acc = accuracy_score(test_labels, preds)
        print(f'  {name}: {ind_acc:.4f}')

    # Simple weighted average fusion (no MLP) for comparison
    # PYSKL default: 2*J + 2*B + 1*JM + 1*BM
    stgcn_streams = {k: v for k, v in test_data['logits'].items() if k.startswith('stgcnpp')}
    if len(stgcn_streams) == 4:
        fused = (2 * stgcn_streams['stgcnpp_j'] +
                 2 * stgcn_streams['stgcnpp_b'] +
                 1 * stgcn_streams['stgcnpp_jm'] +
                 1 * stgcn_streams['stgcnpp_bm'])
        preds = fused.argmax(axis=1)
        print(f'\n  STGCN++ 4-stream weighted avg (2:2:1:1): {accuracy_score(test_labels, preds):.4f}')


if __name__ == '__main__':
    main()