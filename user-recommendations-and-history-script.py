import os
import numpy as np
import torch
import torch.nn as nn

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

def get_top_n_recommendations(model, user_id, item_pool, n=10):
    model.eval()
    with torch.no_grad():
        user_ids = torch.LongTensor([user_id] * len(item_pool))
        item_ids = torch.LongTensor(item_pool)
        predictions = model(user_ids, item_ids).numpy()
    
    top_n_items = item_pool[np.argsort(predictions)[-n:][::-1]]
    return top_n_items

def load_data(data_file='preprocessed_data.npz'):
    file_path = os.path.join(SCRIPT_DIR, data_file)
    data = np.load(file_path)
    return data['X_train'], data['y_train']

def get_user_history(user_id, X, y):
    user_data = X[X[:, 0] == user_id]
    watched_movies = user_data[user_data[:, 2] > 3, 1]  # Movies rated above 3
    return watched_movies

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
        
        while True:
            user_input = input("Enter a user ID (or 'q' to quit): ")
            
            if user_input.lower() == 'q':
                print("Exiting the program.")
                break
            
            try:
                user_id = int(user_input)
                if user_id < 0 or user_id >= num_users:
                    print(f"Invalid user ID. Please enter a number between 0 and {num_users - 1}.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a valid integer for the user ID.")
                continue
            
            # Get user's watch history
            watched_movies = get_user_history(user_id, X, y)
            
            print(f"\nUser {user_id}'s watch history:")
            for movie_id in watched_movies[:10]:  # Show top 10 watched movies
                print(f"- Movie ID: {movie_id}")
            
            if len(watched_movies) > 10:
                print(f"... and {len(watched_movies) - 10} more.")
            
            # Generate recommendations
            item_pool = np.arange(num_items)
            recommendations = get_top_n_recommendations(model, user_id, item_pool)
            
            print(f"\nTop 10 recommendations for user {user_id}:")
            for i, movie_id in enumerate(recommendations, 1):
                print(f"{i}. Movie ID: {movie_id}")
            
            print("\n" + "-"*50)  # Separator for readability
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
