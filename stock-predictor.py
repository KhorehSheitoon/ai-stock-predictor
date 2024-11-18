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
load_model = keras.models.load_model
import ta
import os
from tqdm import tqdm


class BatchStockPredictor:
    def __init__(self, data_dir, prediction_days=3, batch_size=32):
        self.data_dir = data_dir
        self.prediction_days = prediction_days
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = None
        
    def get_user_preference(self):
        while True:
            response = input("Would you like to use batch processing for large datasets? (Y/N): ").strip().upper()
            if response in ['Y', 'N']:
                return response == 'Y'
    
    def load_stock_data(self, symbol):
        file_path = os.path.join(self.data_dir, f"{symbol}.csv")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        return df.sort_values('Date')
    
    def prepare_data(self, df):
        # Technical indicators calculation
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        
        # Update Bollinger Bands calculation
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_middle'] = bollinger.bollinger_mavg()
        df['BB_lower'] = bollinger.bollinger_lband()
        
        df['Price_Change'] = df['Close'].pct_change()
        df['Target'] = (df['Close'].shift(-self.prediction_days) > df['Close']).astype(int)
        
        df.dropna(inplace=True)
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'SMA_20', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'Price_Change']
        
        scaled_data = self.scaler.fit_transform(df[features])
        
        X, y = [], []
        sequence_length = 10
        
        for i in range(sequence_length, len(scaled_data) - self.prediction_days):
            X.append(scaled_data[i-sequence_length:i])
            y.append(df['Target'].iloc[i])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        model = Sequential([
            keras.layers.Input(shape=input_shape),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def batch_generator(self, symbols, batch_size):
        while True:
            # Shuffle symbols for each epoch
            np.random.shuffle(symbols)
            
            batch_symbols = []
            batch_X = []
            batch_y = []
            
            for symbol in symbols:
                try:
                    df = self.load_stock_data(symbol)
                    X, y = self.prepare_data(df)
                    
                    batch_X.append(X)
                    batch_y.append(y)
                    batch_symbols.append(symbol)
                    
                    if len(batch_symbols) == batch_size:
                        X = np.concatenate(batch_X)
                        y = np.concatenate(batch_y)
                        
                        # Shuffle data within batch
                        indices = np.arange(len(X))
                        np.random.shuffle(indices)
                        X = X[indices]
                        y = y[indices]
                        
                        yield X, y
                        
                        batch_symbols = []
                        batch_X = []
                        batch_y = []
                        
                except Exception as e:
                    print(f"Error processing {symbol}: {str(e)}")
                    continue
            
            # Handle remaining data
            if batch_symbols:
                X = np.concatenate(batch_X)
                y = np.concatenate(batch_y)
                indices = np.arange(len(X))
                np.random.shuffle(indices)
                X = X[indices]
                y = y[indices]
                yield X, y

    def train_on_stocks(self, symbols, epochs=50, validation_split=0.2):
        use_batch = self.get_user_preference()
        
        if use_batch:
            return self._train_with_batches(symbols, epochs, validation_split)
        else:
            return self._train_all_at_once(symbols, epochs, validation_split)
    
    def _train_with_batches(self, symbols, epochs, validation_split):
        print("Training with batch processing...")
        
        # Get input shape from first stock
        df = self.load_stock_data(symbols[0])
        X_sample, _ = self.prepare_data(df)
        
        if not self.model:
            self.build_model(input_shape=(X_sample.shape[1], X_sample.shape[2]))
        
        # Split symbols into train and validation
        train_symbols = symbols[:int(len(symbols) * (1-validation_split))]
        val_symbols = symbols[int(len(symbols) * (1-validation_split)):]
        
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss = []
            train_acc = []
            train_generator = self.batch_generator(train_symbols, self.batch_size)
            
            for _ in tqdm(range(len(train_symbols) // self.batch_size)):
                X_batch, y_batch = next(train_generator)
                metrics = self.model.train_on_batch(X_batch, y_batch)
                train_loss.append(metrics[0])
                train_acc.append(metrics[1])
            
            # Validation
            val_loss = []
            val_acc = []
            val_generator = self.batch_generator(val_symbols, self.batch_size)
            
            for _ in range(len(val_symbols) // self.batch_size):
                X_batch, y_batch = next(val_generator)
                metrics = self.model.test_on_batch(X_batch, y_batch)
                val_loss.append(metrics[0])
                val_acc.append(metrics[1])
            
            # Update history
            history['loss'].append(np.mean(train_loss))
            history['val_loss'].append(np.mean(val_loss))
            history['accuracy'].append(np.mean(train_acc))
            history['val_accuracy'].append(np.mean(val_acc))
            
            print(f"loss: {history['loss'][-1]:.4f} - accuracy: {history['accuracy'][-1]:.4f} - "
                  f"val_loss: {history['val_loss'][-1]:.4f} - val_accuracy: {history['val_accuracy'][-1]:.4f}")
        
        return history
    
    def _train_all_at_once(self, symbols, epochs, validation_split):
        print("Training with all data at once...")
        combined_X = []
        combined_y = []
        
        for symbol in tqdm(symbols, desc="Loading data"):
            try:
                df = self.load_stock_data(symbol)
                X, y = self.prepare_data(df)
                combined_X.append(X)
                combined_y.append(y)
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
        
        X = np.concatenate(combined_X)
        y = np.concatenate(combined_y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split, shuffle=True)
        
        if not self.model:
            self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=self.batch_size,
            verbose=1
        )
        
        return history.history
    
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

# Initialize predictor
predictor = BatchStockPredictor(data_dir='data', batch_size=32)

# Get list of stock symbols from directory
symbols = [f.split('.')[0] for f in os.listdir('data') if f.endswith('.csv')]

# Train model
history = predictor.train_on_stocks(symbols, epochs=50)

# Make predictions for a specific stock
predictions = predictor.predict_stock('AAPL')
print(predictions.tail())

# Save the model
predictor.save_model('batch_stock_predictor_model.h5')
