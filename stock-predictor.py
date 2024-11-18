import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
Sequential = keras.models.Sequential
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
import ta
import os


class MultiStockPredictor:
    def __init__(self, data_dir, prediction_days=3):
        self.data_dir = data_dir
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler()
        self.model = None
        
    def load_stock_data(self, symbol):
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.sort_values('Date')
    
    def prepare_data(self, df):
        # Calculate technical indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.volatility.bollinger_bands(df['Close'])
        
        # Calculate price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Target'] = (df['Close'].shift(-self.prediction_days) > df['Close']).astype(int)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        # Select features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'SMA_20', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'Price_Change']
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df[features])
        
        # Prepare sequences
        X, y = [], []
        sequence_length = 10
        
        for i in range(sequence_length, len(scaled_data) - self.prediction_days):
            X.append(scaled_data[i-sequence_length:i])
            y.append(df['Target'].iloc[i])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def train_on_stock(self, symbol, validation_split=0.2, epochs=50, batch_size=32):
        df = self.load_stock_data(symbol)
        X, y = self.prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split, shuffle=False)
        
        if not self.model:
            self.build_model(input_shape=(X.shape[1], X.shape[2]))
            
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history, (X_test, y_test)
    
    def train_on_multiple_stocks(self, symbols, validation_split=0.2, epochs=50, batch_size=32):
        combined_X = []
        combined_y = []
        
        for symbol in symbols:
            df = self.load_stock_data(symbol)
            X, y = self.prepare_data(df)
            combined_X.append(X)
            combined_y.append(y)
        
        X = np.concatenate(combined_X)
        y = np.concatenate(combined_y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split, shuffle=True)
        
        if not self.model:
            self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history, (X_test, y_test)
    
    def predict_stock(self, symbol):
        df = self.load_stock_data(symbol)
        X, _ = self.prepare_data(df)
        predictions = self.model.predict(X)
        
        results = pd.DataFrame({
            'Date': df['Date'].iloc[10:].reset_index(drop=True),
            'Close': df['Close'].iloc[10:].reset_index(drop=True),
            'Prediction': predictions.flatten() > 0.5,
            'Confidence': predictions.flatten()
        })
        
        return results
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
    def load_saved_model(self, filepath):
        self.model = load_model(filepath)

# Example usage:
"""
# Initialize predictor
predictor = MultiStockPredictor(data_dir='stock_data')

# Train on multiple stocks
stock_symbols = ['AAPL', 'GOOGL', 'MSFT']
history, test_data = predictor.train_on_multiple_stocks(stock_symbols, epochs=50)

# Make predictions for a specific stock
predictions = predictor.predict_stock('AAPL')
print(predictions.tail())

# Save the model
predictor.save_model('stock_predictor_model.h5')
"""