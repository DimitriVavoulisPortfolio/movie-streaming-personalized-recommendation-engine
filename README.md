# Movie Recommendation System

## Project Overview

This project implements a Neural Collaborative Filtering (NCF) based movie recommendation system. It uses a custom NCF model to predict user preferences for movies and generate personalized recommendations. The system is designed to process large-scale user-item interaction data and provide accurate movie suggestions.

### Key Features

- Custom Neural Collaborative Filtering (NCF) model implementation
- Efficient data preprocessing and standardization pipeline
- Advanced model training with early stopping and learning rate scheduling
- User-specific movie recommendation generation
- Scalable architecture suitable for large datasets

## Project Structure

1. **model-training-script v1.5.py**: Main script for NCF model training
2. **user-recommendations-and-history-script.py**: Script for generating user-specific recommendations
3. **preprocessed_data.npz**: https://github-1.s3.amazonaws.com/preprocessed_data.npz (not included in the repository)
4. **best_ncf_model.pth**: Saved best model weights
5. **ncf_model_info.json**: Model metadata and training information
6. **config.yaml**: Configuration file for model parameters and training settings
7. **training.log**: Log file containing training progress and results

## Documentation

- **MODEL.md**: https://github.com/DimitriVavoulisPortfolio/movie-streaming-personalized-recommendation-engine/blob/main/MODEL.md
- **PROCESS.md**: https://github.com/DimitriVavoulisPortfolio/movie-streaming-personalized-recommendation-engine/blob/main/PROCESS.md

## Model Performance

Note: Current performance metrics are under investigation due to issues with Hit Ratio and NDCG calculations.

- **Accuracy**: TBD
- **MSE**: Varies by epoch, refer to training logs
- **Hit Ratio**: Currently 0 (under investigation)
- **NDCG**: Currently 0 (under investigation)

## Quick Start Guide

1. Clone the repository:
   ```
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. To train the model:
   ```
   python model-training-script v1.5.py
   ```

4. To generate recommendations:
   ```
   python user-recommendations-and-history-script.py
   ```

## Future Work

- Investigate and resolve issues with Hit Ratio and NDCG metrics
- Implement more advanced recommendation algorithms
- Create a user-friendly web interface for the recommendation system
- Develop an API for real-time recommendation generation
- Optimize model performance for larger datasets

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please open an issue in this repository or contact [Your Name](mailto:your.email@example.com).

