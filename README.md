# LSTM Autoencoder for Signal Anomaly Detection

This project implements an LSTM Autoencoder for anomaly detection in signal processing, specifically designed for Laser Doppler Velocimetry (LDV) data analysis. The implementation includes both pre-trained model usage and transfer learning capabilities.

## Overview

The project uses a deep learning approach to detect anomalies in time series data from LDV measurements. It employs an LSTM Autoencoder architecture that can be used in two modes:
1. Pre-trained model inference
2. Transfer learning for domain adaptation

The system processes LDV signals to detect anomalies in velocity measurements, which can be crucial for monitoring and maintaining system stability.

## Features

- LSTM Autoencoder architecture for time series anomaly detection
- Transfer learning capabilities with layer freezing
- Signal feature extraction and preprocessing
- Adaptive threshold calculation for anomaly detection
- Visualization tools for results analysis
- Support for multiple LDV channels (LDV1 and LDV2)
- Real-time anomaly detection capabilities

## Dependencies

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Spyder Kernels
- Plotly (for interactive visualizations)

## Project Structure

The main script (`Transfer_trainKKK15_testBNSF11_41-57_v1.py`) contains:
1. Data preprocessing and feature extraction
2. Model loading and transfer learning setup
3. Anomaly detection implementation
4. Visualization tools

### Data Processing Pipeline
- Feature extraction from LDV signals
- Signal preprocessing and normalization
- Sequence generation for LSTM input
- Anomaly detection and thresholding
- Results visualization and analysis

## Usage

1. Data Preparation:
   - Place your LDV data files in the appropriate directory
   - Ensure data files follow the naming convention: `LDV_train_classification_signalfeatures_[Dataset]_tstep[tstep].xls`
   - Required input files:
     - LDV signal features file
     - Original velocity data files
     - Pre-trained model file

2. Model Usage:
   - Load pre-trained model using: `model = keras.models.load_model(save_path)`
   - For transfer learning, modify layer trainability as needed
   - Default model path: `./SDS_KKK15.h5`

3. Anomaly Detection:
   - The model calculates reconstruction error
   - Anomalies are detected based on adaptive thresholding
   - Results are visualized with time series plots
   - Supports both single and multi-channel analysis

## Parameters

Key parameters that can be adjusted:
- `TIME_STEPS`: Sequence length for LSTM (default: 20)
- `threshold`: Anomaly detection threshold (calculated based on training data)
- Window size for rolling mean calculation (default: 70-75 samples)
- Learning rate for transfer learning (default: 0.001)
- Early stopping patience (default: 7 epochs)
- Batch size (default: 72)

## Visualization

The script provides several visualization options:
- MAE Loss plots with threshold lines
- Velocity time series with anomaly highlighting
- Original signal comparison between LDV1 and LDV2
- Interactive plots using Plotly
- Time-domain analysis of detected anomalies

## Technical Details

### Data Preprocessing
- Log transformation for skewed features (threshold: skewness > 0.75)
- Rolling mean calculation for noise reduction
- Standardization using StandardScaler
- Feature selection based on PCA (selected features: std, fe, energy, rms, kurt)

### Transfer Learning Implementation
- Freezes outer layers (first 5 and last 5)
- Reinitializes center layers using GlorotUniform initializer
- Uses modified learning rate for frozen layers
- Implements early stopping with patience=7

### Anomaly Detection
- Calculates reconstruction error using MAE
- Implements adaptive thresholding based on training data statistics
- Supports continuous anomaly detection in time series
- Provides anomaly indices for further analysis

## Notes

- The implementation includes data preprocessing steps including:
  - Log transformation for skewed features
  - Rolling mean calculation
  - Standardization
  - Feature selection based on PCA

- Transfer learning is implemented by:
  - Freezing outer layers
  - Reinitializing center layers
  - Using a modified learning rate

- Performance considerations:
  - Window size may need adjustment based on signal characteristics
  - Threshold calculation can be modified for different sensitivity levels
  - Transfer learning parameters can be tuned for specific domains

## License
Contact the authors for more information

## Contact
chiyang.lucas@gmail.com

## Cite
Yang, C., Kaynardag, K., Lee, G. W., & Salamone, S. (2025). Long short-term memory autoencoder for anomaly detection in rails using laser doppler vibrometer measurements. Journal of Nondestructive Evaluation, Diagnostics and Prognostics of Engineering Systems, 8(3), 031003.
