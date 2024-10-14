import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, layers=[64, 32, 16, 8], dropout=0.2):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc_layers = nn.ModuleList()
        prev_layer = embedding_dim * 2
        for layer in layers:
            self.fc_layers.append(nn.Linear(prev_layer, layer))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.BatchNorm1d(layer))
            self.fc_layers.append(nn.Dropout(dropout))
            prev_layer = layer
        
        self.final_layer = nn.Linear(prev_layer, 1)
        
    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        
        x = torch.cat([user_embedded, item_embedded], dim=1)
        
        for layer in self.fc_layers:
            x = layer(x)
        
        output = self.final_layer(x)
        return output.squeeze()

def load_model(model_file_name, embedding_dim, layers, dropout):
    model_path = os.path.join(SCRIPT_DIR, model_file_name)
    state_dict = torch.load(model_path)
    
    num_users = state_dict['user_embedding.weight'].shape[0]
    num_items = state_dict['item_embedding.weight'].shape[0]
    
    model = NCF(num_users, num_items, embedding_dim, layers, dropout)
    model.load_state_dict(state_dict)
    model.eval()
    return model, num_users, num_items

def load_data(data_file='preprocessed_data.npz'):
    file_path = os.path.join(SCRIPT_DIR, data_file)
    data = np.load(file_path)
    return data['X_train'], data['y_train']

def get_predictions(model, X):
    model.eval()
    with torch.no_grad():
        user_ids = torch.LongTensor(X[:, 0])
        item_ids = torch.LongTensor(X[:, 1])
        predictions = torch.sigmoid(model(user_ids, item_ids)).numpy()
    return predictions

def analyze_binary_classification(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    print("Binary Classification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def analyze_predictions(y_true, y_pred):
    print("\nPrediction Analysis:")
    print(f"Minimum prediction: {y_pred.min():.4f}")
    print(f"Maximum prediction: {y_pred.max():.4f}")
    print(f"Mean prediction: {y_pred.mean():.4f}")
    print(f"Predictions above threshold (0.5): {np.sum(y_pred > 0.5)}")
    print(f"Total predictions: {len(y_pred)}")
    print(f"Actual positives (1s): {np.sum(y_true)}")
    print(f"Actual negatives (0s): {len(y_true) - np.sum(y_true)}")

def main():
    try:
        # Model parameters
        embedding_dim = 64
        layers = [64, 32, 16, 8]
        dropout = 0.2
        
        # Load the trained model
        model_file_name = 'best_ncf_model.pth'
        model, num_users, num_items = load_model(model_file_name, embedding_dim, layers, dropout)
        
        print(f"Model loaded with {num_users} users and {num_items} items.")
        
        # Load data
        X, y = load_data()
        
        # Get predictions
        predictions = get_predictions(model, X)
        
        # Analyze binary classification
        analyze_binary_classification(y, predictions)
        
        # Analyze predictions
        analyze_predictions(y, predictions)
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
