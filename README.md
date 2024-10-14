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

1. **logs and other info**: logs of the entire process as well as JSON and YAML files made during the process including screenshots of model usage
2. **scripts**: scripts of the whole process except for model usage
4. **user-recommendations-and-history-script.py**: Script for generating user-specific recommendations, get the preprocessed_data.npz file in the same folder as this for it to work
5. **preprocessed_data.npz**: https://github-1.s3.amazonaws.com/preprocessed_data.npz (not included in the repository)
6. **best_ncf_model.pth**: Saved best model weights
7. **Dataset**: https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system

## Documentation

- **MODEL.md**: https://github.com/DimitriVavoulisPortfolio/movie-streaming-personalized-recommendation-engine/blob/main/MODEL.md
- **PROCESS.md**: https://github.com/DimitriVavoulisPortfolio/movie-streaming-personalized-recommendation-engine/blob/main/PROCESS.md

## Model Performance

- **Accuracy**:  0.8210
- **Precision**:  0.8210
- **Recall**: 1.0000
- **F1 Score**: 0.9017

## Quick Start Guide

1. Clone the repository:
   ```
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Install dependencies:
   ```
   pip install pandas numpy scikit-learn PyYAML torch
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

- Implement more advanced recommendation algorithms
- Create a user-friendly web interface for the recommendation system
- Develop an API for real-time recommendation generation
- Optimize model performance for larger datasets

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please open an issue in this repository or contact [Dimitri Vavoulis](mailto:dimitrivavoulis3@gmail.com).

