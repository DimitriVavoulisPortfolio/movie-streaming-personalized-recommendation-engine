# MODEL.md - Movie Recommendation System Model Documentation

## Model Architecture

The movie recommendation system uses a Neural Collaborative Filtering (NCF) model, which combines the power of matrix factorization with neural networks to predict user-item interactions.

### Key Components

1. **Embedding Layers**:
   - User Embedding: Maps user IDs to dense vectors
   - Item Embedding: Maps item (movie) IDs to dense vectors

2. **Neural Network Layers**:
   - Multiple fully connected layers with ReLU activation
   - Batch Normalization for each layer
   - Dropout for regularization

3. **Output Layer**: 
   - Single neuron with linear activation for rating prediction

### Model Configuration

- Base model: Custom Neural Collaborative Filtering (NCF)
- Task: Rating Prediction
- Output: Continuous rating prediction

### Detailed Architecture

```python
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
```

### Training Configuration

- Framework: PyTorch
- Batch size: Dynamically calculated based on dataset size
- Maximum epochs: Configurable (default: 50)
- Learning rate strategy:
  - Scheduler: OneCycleLR
  - Initial learning rate: Configurable (default: 0.001)
  - Max learning rate: Configurable (default: 0.01)
- Weight decay: Configurable (default: 1e-5)
- Loss function: Mean Squared Error (MSE)
- Early stopping: Implemented with configurable patience

## Performance Metrics

- Mean Squared Error (MSE): Primary metric for rating prediction accuracy
- Hit Ratio: For evaluating recommendation ranking quality (currently under investigation)
- Normalized Discounted Cumulative Gain (NDCG): For evaluating recommendation ranking quality (currently under investigation)

## Model Inputs and Outputs

### Inputs:
- User ID: Integer
- Item (Movie) ID: Integer

### Outputs:
- Predicted Rating: Float (typically in the range of the original rating scale)

## Additional Notes

- The model leverages GPU acceleration for faster training when available.
- The architecture is flexible, allowing for easy configuration of embedding size and hidden layers.
- Current performance metrics for Hit Ratio and NDCG are under investigation due to calculation issues.
- Future work may include experimenting with different model architectures, incorporating additional features, and resolving evaluation metric issues.

