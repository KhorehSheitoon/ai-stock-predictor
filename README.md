
# Stock Price Prediction Using LSTM

An AI-powered stock market prediction system using LSTM neural networks and technical indicators for short-term trading signals.

## Features

- Multi-stock analysis with batch processing capability
- Technical indicators (SMA, RSI, MACD, Bollinger Bands)
- Memory-efficient processing for large datasets
- Customizable prediction timeframe
- Progress tracking with tqdm
- Model save/load functionality

## Requirements

```
pandas
numpy
scikit-learn
tensorflow
ta
tqdm
```

## Installation

```bash
git clone https://github.com/yourusername/stock-prediction-lstm.git
cd stock-prediction-lstm
pip install -r requirements.txt
```

## Usage

1. Data Format:

```csv
Date,Open,High,Low,Close,Volume,Dividends,Stock Splits
```

2. Basic Usage:

```python
from stock_predictor import BatchStockPredictor

# Initialize predictor
predictor = BatchStockPredictor(data_dir='stock_data', batch_size=32)

# Train model
symbols = [f.split('.')[0] for f in os.listdir('stock_data') if f.endswith('.csv')]
history = predictor.train_on_stocks(symbols, epochs=50)

# Predict specific stock
predictions = predictor.predict_stock('AAPL')
```

3. Save/Load Model:

```python
# Save
predictor.save_model('model.h5')

# Load
predictor.load_saved_model('model.h5')
```

## Model Architecture

- Dual LSTM layers (50 units each)
- Dropout layers (0.2)
- Dense layers for final prediction
- Binary classification (price movement up/down)

## Performance Considerations

- Batch processing option for large datasets
- Automatic memory management
- Error handling for problematic files
- Progress monitoring during training

## Disclaimer

This tool is for educational purposes only. Trading stocks carries significant risks, and past performance doesn't guarantee future results.

## Data Source

The stock data used in this project comes from the [Yahoo Finance All Stocks Dataset (Daily Update)](https://www.kaggle.com/datasets/tanavbajaj/yahoo-finance-all-stocks-dataset-daily-update/data) on Kaggle.

## License

Dataset: Open Data Commons Open Database License (ODbL) v1.0

Code: MIT License - See LICENSE file for details

### Data Disclaimer

This dataset is provided 'as is', without any warranties. Any damages resulting from its use are disclaimed. Users should seek appropriate legal/financial advice before using this data for trading decisions.
