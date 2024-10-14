import os
import logging
import time
from datetime import datetime
import json
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up file handler for logging
try:
    file_handler = logging.FileHandler('training.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Failed to set up file logging: {str(e)}. Continuing with console logging only.")

# Custom NCF implementation
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

# Configuration loading
def load_config():
    config_path = 'config.yaml'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_config_path = os.path.join(script_dir, config_path)

    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script directory: {script_dir}")
    logger.info(f"Attempting to load config from: {full_config_path}")

    if os.path.exists(full_config_path):
        try:
            with open(full_config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {full_config_path}")
            return config
        except Exception as e:
            logger.error(f"Error reading config file: {str(e)}")
            raise
    else:
        logger.error(f"Config file not found at {full_config_path}")
        logger.info("Contents of script directory:")
        for file in os.listdir(script_dir):
            logger.info(f"- {file}")
        raise FileNotFoundError(f"Config file {config_path} not found in {script_dir}")

# Load configuration
try:
    config = load_config()
except Exception as e:
    logger.error(f"Failed to load configuration: {str(e)}")
    logger.info("Using default configuration.")
    config = {}

def get_config_value(key, default_value, value_type):
    try:
        value = config.get(key, default_value)
        return value_type(value)
    except ValueError:
        logger.warning(f"Invalid value for {key} in config. Using default: {default_value}")
        return default_value

# Set random seed for reproducibility
RANDOM_SEED = get_config_value('random_seed', 42, int)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available, using CPU")
    return device

def load_and_preprocess_data(file_path):
    logger.info("Loading and preprocessing data...")
    data = np.load(file_path)
    X, y = data['X_train'], data['y_train']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=get_config_value('validation_split', 0.1, float), random_state=RANDOM_SEED)
    
    scaler = StandardScaler()
    X_train[:, 2:] = scaler.fit_transform(X_train[:, 2:])
    X_val[:, 2:] = scaler.transform(X_val[:, 2:])
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")
    
    return X_train, X_val, y_train, y_val

def create_data_loaders(X_train, y_train, X_val, y_val, device):
    # Adjust batch size to make total batches 4400
    total_samples = len(X_train)
    batch_size = total_samples // 4400
    if total_samples % 4400 != 0:
        batch_size += 1  # Ensure we cover all samples

    train_dataset = TensorDataset(
        torch.LongTensor(X_train[:, 0]).to(device),  # User IDs
        torch.LongTensor(X_train[:, 1]).to(device),  # Item IDs
        torch.FloatTensor(y_train).to(device)        # Ratings
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(
        torch.LongTensor(X_val[:, 0]).to(device),
        torch.LongTensor(X_val[:, 1]).to(device),
        torch.FloatTensor(y_val).to(device)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Adjusted batch size: {batch_size}")
    logger.info(f"Number of training batches: {len(train_loader)}")
    
    return train_loader, val_loader

def evaluate_model(model, val_loader, criterion, device, max_val_samples=100000):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    all_user_ids = []
    all_item_ids = []
    samples_processed = 0

    with torch.no_grad():
        for user_ids, item_ids, ratings in val_loader:
            if samples_processed >= max_val_samples:
                break

            outputs = model(user_ids, item_ids)
            loss = criterion(outputs, ratings)
            total_loss += loss.item() * len(ratings)

            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())
            all_user_ids.extend(user_ids.cpu().numpy())
            all_item_ids.extend(item_ids.cpu().numpy())

            samples_processed += len(ratings)

    avg_loss = total_loss / samples_processed

    # Convert lists to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    user_ids = np.array(all_user_ids)
    item_ids = np.array(all_item_ids)

    # Compute metrics
    k = 10  # top-k for recommendations
    threshold = 3.5  # threshold for positive rating

    unique_users = np.unique(user_ids)
    hit_ratio = 0
    ndcg = 0

    for user in unique_users:
        user_mask = user_ids == user
        user_predictions = predictions[user_mask]
        user_targets = targets[user_mask]
        user_items = item_ids[user_mask]

        # Sort items by predicted ratings
        sorted_indices = np.argsort(user_predictions)[::-1]
        top_k_items = user_items[sorted_indices[:k]]

        # Hit Ratio
        hit = np.isin(top_k_items, user_items[user_targets >= threshold]).any()
        hit_ratio += hit

        # NDCG
        ideal_ranking = np.sort(user_targets)[::-1]
        dcg = np.sum((user_targets[sorted_indices[:min(k, len(user_items))]] >= threshold) / 
                     np.log2(np.arange(2, min(k, len(user_items))+2)))
        idcg = np.sum((ideal_ranking[:min(k, len(user_items))] >= threshold) / 
                      np.log2(np.arange(2, min(k, len(user_items))+2)))
        ndcg += dcg / idcg if idcg > 0 else 0

    hit_ratio /= len(unique_users)
    ndcg /= len(unique_users)

    logger.info(f"Validation samples processed: {samples_processed}")
    return avg_loss, hit_ratio, ndcg

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs):
    best_val_loss = float('inf')
    no_improve = 0
    patience = get_config_value('patience', 5, int)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (user_ids, item_ids, ratings) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(user_ids, item_ids)
            loss = criterion(outputs, ratings)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        # Validation
        val_loss, hit_ratio, ndcg = evaluate_model(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}, "
                    f"Val Loss: {val_loss:.4f}, Hit Ratio: {hit_ratio:.4f}, NDCG: {ndcg:.4f}, "
                    f"Time: {epoch_time:.2f}s")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_ncf_model.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve == patience:
                logger.info("Early stopping!")
                break
        
        # Periodic GPU memory cleanup
        if epoch % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return best_val_loss

def main():
    try:
        device = get_device()
        
        # Load and preprocess data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(current_dir, get_config_value('data_file', 'preprocessed_data.npz', str))
        X_train, X_val, y_train, y_val = load_and_preprocess_data(data_file)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val, device)
        
        # Initialize model
        num_users = int(X_train[:, 0].max()) + 1
        num_items = int(X_train[:, 1].max()) + 1
        embedding_dim = get_config_value('embedding_dim', 64, int)
        layers = config.get('layers', [64, 32, 16, 8])
        dropout = get_config_value('dropout', 0.2, float)
        
        model = NCF(num_users, num_items, embedding_dim, layers, dropout).to(device)
        
        # Optimizer and scheduler
        learning_rate = get_config_value('learning_rate', 0.001, float)
        weight_decay = get_config_value('weight_decay', 1e-5, float)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        steps_per_epoch = len(train_loader)
        max_lr = get_config_value('max_lr', 0.01, float)
        num_epochs = get_config_value('num_epochs', 50, int)
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, epochs=num_epochs, steps_per_epoch=steps_per_epoch)
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Train the model
        start_time = time.time()
        best_val_loss = train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs)
        total_time = time.time() - start_time
        
        logger.info(f"Training completed. Best Validation Loss: {best_val_loss:.4f}")
        logger.info(f"Total Training Time: {total_time:.2f} seconds")
        
        # Save model information
        model_info = {
            "model_type": "Neural Collaborative Filtering (Custom Implementation)",
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": embedding_dim,
            "layers": layers,
            "dropout": dropout,
            "best_val_loss": best_val_loss,
            "batch_size": train_loader.batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": "Adam with OneCycleLR",
            "early_stopping_patience": get_config_value('patience', 5, int),
            "total_training_time": total_time,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        with open('ncf_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=4)
        
        logger.info("Model information saved.")
        
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
    finally:
        # Cleanup code
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()