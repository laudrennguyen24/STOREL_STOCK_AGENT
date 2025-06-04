import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
import yfinance as yf
from vietfin import vf
from collections import deque
import random
import warnings
import ta
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class DataProcessor:
    """Lớp xử lý và thu thập dữ liệu chứng khoán"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.data = None
    
    
    def fetch_stock_data(self, symbol, period='2y'):
      try:
          print(f"Đang tải dữ liệu cho {symbol}...")

          today = datetime.today()
          if period.endswith('y'):
              days = int(period[:-1]) * 365
          elif period.endswith('m'):
              days = int(period[:-1]) * 30
          elif period.endswith('d'):
              days = int(period[:-1])
          else:
              days = 365

          start_date = (today - timedelta(days=days)).strftime('%Y-%m-%d')
          end_date = today.strftime('%Y-%m-%d')

          results = vf.equity.price.historical(
              symbol=symbol,
              start_date=start_date,
              end_date=end_date
          )
          df = results.to_df()

          if df is None or df.empty:
              raise ValueError(f"Không tìm thấy dữ liệu cho {symbol}")

          # Xử lý dataframe tương tự như trước
          if 'date' not in df.columns:
              if isinstance(df.index, pd.DatetimeIndex):
                  df = df.reset_index().rename(columns={df.index.name or 0: 'date'})
              else:
                  df = df.reset_index()
                  if 'date' not in df.columns:
                      idx_col = df.columns[0]
                      df = df.rename(columns={idx_col: 'date'})

          df['date'] = pd.to_datetime(df['date'])
          df['symbol'] = symbol
          df = df.sort_values('date').reset_index(drop=True)

          df = df.rename(columns={
              'open': 'Open',
              'high': 'High',
              'low': 'Low',
              'close': 'Close',
              'volume': 'Volume'
          })

          print(f"✅ Đã tải {len(df)} ngày dữ liệu")
          self.data = df
          print(df)
          return self.data

      except Exception as e:
          print(f"❌ Lỗi khi tải dữ liệu: {e}")
          return None
    
    def add_technical_indicators(self, data):
        """Thêm các chỉ số kỹ thuật"""
        df = data.copy()
        
        # Moving Averages
        df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MACD
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
        
        # Bollinger Bands
        df['BB_High'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_Low'] = ta.volatility.bollinger_lband(df['Close'])
        df['BB_Width'] = df['BB_High'] - df['BB_Low']
        
        # Volume indicators
        df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price change
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Close'].diff()
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Support and Resistance levels
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        
        return df.dropna()
    
    def prepare_lstm_data(self, data, lookback_days=60):
        """Chuẩn bị dữ liệu cho LSTM"""
        # Chọn features quan trọng
        features = ['Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 
                   'BB_Width', 'Volume_Ratio', 'Volatility']
        
        df = data[features].copy()
        
        # Chuẩn hóa dữ liệu
        scaled_data = self.scaler.fit_transform(df)
        
        X, y = [], []
        for i in range(lookback_days, len(scaled_data)):
            X.append(scaled_data[i-lookback_days:i])
            y.append(scaled_data[i, 0])  # Dự đoán giá Close
            
        return np.array(X), np.array(y), df

class LSTMPredictor:
    """Mô hình LSTM để dự đoán giá"""
    
    def __init__(self, input_shape, units=120):
        self.model = None
        self.input_shape = input_shape
        self.units = units
        self.history = None
        self.dropout_rate = 0.3
        
    def build_model(self):
        """Xây dựng mô hình LSTM"""
        self.model = Sequential([
            Input(shape=self.input_shape),
            
            # First LSTM layer
            LSTM(self.units, return_sequences=True, activation='tanh'),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Second LSTM layer
            LSTM(self.units // 2, return_sequences=True, activation='tanh'),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Third LSTM layer
            LSTM(self.units // 4, return_sequences=False, activation='tanh'),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            # Dense layers
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(self.dropout_rate / 2),
            
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print("🧠 Mô hình LSTM đã được xây dựng:")
        self.model.summary()
        
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Huấn luyện mô hình"""
        print("🚀 Bắt đầu huấn luyện LSTM...")
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=7, min_lr=0.0001
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("✅ Huấn luyện LSTM hoàn thành!")
        
    def predict(self, X):
        """Dự đoán giá"""
        return self.model.predict(X)
    
    def plot_training_history(self):
        """Vẽ biểu đồ quá trình huấn luyện"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

class RLTradingAgent:
    """Reinforcement Learning Agent cho trading"""
    
    def __init__(self, state_size, action_size=3, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size  # 0: Hold, 1: Buy, 2: Sell
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
        
    def _build_model(self):
        """Build improved Q-network"""
        model = Sequential([
            Dense(256, input_dim=self.state_size, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            loss='huber',
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        
        return model
    
    def update_target_network(self):
        """Cập nhật target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Lưu experience vào memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Chọn hành động dựa trên epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """Huấn luyện agent từ experience replay"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Tính target Q-values
        # Double DQN
        q_next = self.q_network.predict(next_states, verbose=0)
        best_actions = np.argmax(q_next, axis=1)
        target_q_next = self.target_network.predict(next_states, verbose=0)
        max_target_q_values = target_q_next[np.arange(batch_size), best_actions]

        
        target_q = rewards + (0.95 * max_target_q_values * (1 - dones))
        
        # Tính current Q-values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Cập nhật Q-values cho actions đã thực hiện
        for i in range(batch_size):
            current_q_values[i][actions[i]] = target_q[i]
        
        # Huấn luyện
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class TradingEnvironment:
    """Môi trường trading cho RL agent"""
    
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001, slippage_rate = 0.001):
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage_rate = slippage_rate
        self.reset()
        
    def reset(self):
        """Reset môi trường"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.initial_balance
        self.trades = []
        return self._get_state()
    
    def _get_state(self):
        """Lấy trạng thái hiện tại"""
        if self.current_step >= len(self.data):
            return np.zeros(10)
            
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Tính các features cho state
        state = [
            current_price / self.data['Close'].max(),  # Normalized price
            self.balance / self.initial_balance,       # Normalized balance
            self.shares_held / 100,                    # Normalized shares
            self.data.iloc[self.current_step]['RSI'] / 100,
            self.data.iloc[self.current_step]['MACD'],
            self.data.iloc[self.current_step]['Volume_Ratio'],
            self.data.iloc[self.current_step]['Volatility'] * 100,
            (current_price - self.data.iloc[self.current_step]['SMA_20']) / current_price,
            self.data.iloc[self.current_step]['Returns'],
            len(self.trades) / 100  # Normalized number of trades
        ]
        
        return np.array(state)
    
    def step(self, action):
        """Thực hiện hành động và trả về reward"""
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {}
            
        current_price = self.data.iloc[self.current_step]['Close']
        next_price = self.data.iloc[self.current_step + 1]['Close']
        
        # Thực hiện hành động
        reward = 0
        
        if action == 1:  # Buy
            if self.balance > current_price * (1 + self.transaction_cost):
                shares_to_buy = int(self.balance / (current_price * (1 + self.transaction_cost)))
                cost = shares_to_buy * current_price * (1 + self.transaction_cost + self.slippage_rate)
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.trades.append({
                    'type': 'BUY',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'step': self.current_step
                })
                
        elif action == 2:  # Sell
            if self.shares_held > 0:
                proceeds = self.shares_held * current_price * (1 - self.transaction_cost - self.slippage_rate)
                self.balance += proceeds
                self.trades.append({
                    'type': 'SELL',
                    'price': current_price,
                    'shares': self.shares_held,
                    'step': self.current_step
                })
                self.shares_held = 0
        
        # Tính reward
        old_total_value = self.total_value
        self.total_value = self.balance + self.shares_held * next_price
        
        # Reward dựa trên sự thay đổi portfolio value
        value_change = (self.total_value - old_total_value) / old_total_value
        price_change = (next_price - current_price) / current_price

        # Thưởng nhiều hơn nếu hành động đúng hướng
        if action == 1 and price_change > 0:
            reward = value_change + 0.01
        elif action == 2 and price_change < 0:
            reward = value_change + 0.01
        else:
            reward = value_change - 0.005  # phạt nhẹ nếu hành động sai

        # Penalty cho việc không hoạt động quá lâu
        if action == 0 and len(self.trades) == 0:
            reward -= 0.001
            
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        return self._get_state(), reward, done, {'total_value': self.total_value}

class AITradingSystem:
    """Hệ thống AI Trading tổng hợp"""
    
    def __init__(self, symbol, initial_balance=10000):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.data_processor = DataProcessor()
        self.lstm_model = None
        self.rl_agent = None
        self.trading_env = None
        self.results = {}
        
    def load_and_prepare_data(self, days=365):
        """Tải và chuẩn bị dữ liệu"""
        print(f"🔄 Chuẩn bị dữ liệu cho {self.symbol}...")

        raw_data = self.data_processor.fetch_stock_data(self.symbol, days)
        if raw_data is None:
            return False

        self.processed_data = self.data_processor.add_technical_indicators(raw_data)
        X, y, feature_data = self.data_processor.prepare_lstm_data(self.processed_data)

        train_size = int(len(X) * 0.8)
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]

        print(f"✅ Dữ liệu đã sẵn sàng: {len(X)} samples, {X.shape[1]} timesteps, {X.shape[2]} features")
        return True
    
    def train_lstm_model(self, epochs=100):
        """Huấn luyện mô hình LSTM"""
        print("🧠 Huấn luyện mô hình LSTM...")
        
        # Khởi tạo và xây dựng model
        self.lstm_model = LSTMPredictor(
            input_shape=(self.X_train.shape[1], self.X_train.shape[2])
        )
        self.lstm_model.build_model()
        
        # Huấn luyện
        self.lstm_model.train(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            epochs=epochs
        )
        
        # Đánh giá
        train_pred = self.lstm_model.predict(self.X_train)
        test_pred = self.lstm_model.predict(self.X_test)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        
        print(f"📊 LSTM Performance:")
        print(f"   Training RMSE: {train_rmse:.4f}")
        print(f"   Testing RMSE: {test_rmse:.4f}")
        
        self.results['lstm_train_rmse'] = train_rmse
        self.results['lstm_test_rmse'] = test_rmse
        
    def train_rl_agent(self, episodes=1000):
        """Huấn luyện RL agent"""
        print("🤖 Huấn luyện RL Agent...")
        
        # Khởi tạo environment và agent
        self.trading_env = TradingEnvironment(
            self.processed_data, 
            initial_balance=self.initial_balance
        )
        
        self.rl_agent = RLTradingAgent(
            state_size=10,
            action_size=3
        )
        
        episode_rewards = []
        episode_values = []
        
        for episode in range(episodes):
            state = self.trading_env.reset()
            total_reward = 0
            
            while True:
                action = self.rl_agent.act(state)
                next_state, reward, done, info = self.trading_env.step(action)
                
                self.rl_agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if done:
                    final_value = info['total_value']
                    episode_rewards.append(total_reward)
                    episode_values.append(final_value)
                    break
            
            # Replay experience
            if len(self.rl_agent.memory) > 32:
                self.rl_agent.replay()
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_value = np.mean(episode_values[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, Avg Portfolio Value: ${avg_value:.2f}")
            
            # Update target network
            if episode % 50 == 0:
                self.rl_agent.update_target_network()
        
        self.results['rl_training_rewards'] = episode_rewards
        self.results['rl_training_values'] = episode_values
        
        print("✅ RL Agent huấn luyện hoàn thành!")
    
    def backtest_strategy(self):
        """Backtest chiến lược trading"""
        print("📈 Thực hiện backtest...")
        
        # Reset environment cho backtest
        state = self.trading_env.reset()
        
        # Lưu trữ results
        portfolio_values = [self.initial_balance]
        actions = []
        prices = []
        predictions = []
        
        # Chuẩn bị dữ liệu cho prediction
        lookback = 60
        
        while True:
            # Lấy dự đoán từ LSTM nếu có đủ dữ liệu
            if self.trading_env.current_step >= lookback:
                # Lấy sequence data cho LSTM
                current_idx = self.trading_env.current_step
                sequence_data = self.processed_data.iloc[current_idx-lookback:current_idx]
                
                # Chuẩn bị input cho LSTM
                features = ['Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 
                           'BB_Width', 'Volume_Ratio', 'Volatility']
                feature_data = sequence_data[features].values
                scaled_sequence = self.data_processor.scaler.transform(feature_data)
                
                # Dự đoán
                lstm_input = scaled_sequence.reshape(1, lookback, len(features))
                price_pred = self.lstm_model.predict(lstm_input)[0][0]
                predictions.append(price_pred)
            else:
                predictions.append(0)
            
            # Agent chọn action
            action = self.rl_agent.act(state)
            
            # Thực hiện action
            next_state, reward, done, info = self.trading_env.step(action)
            
            # Lưu kết quả
            actions.append(action)
            portfolio_values.append(info['total_value'])
            prices.append(self.processed_data.iloc[self.trading_env.current_step]['Close'])
            
            state = next_state
            
            if done:
                break
        
        # Tính toán metrics
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance * 100
        
        # Sharpe ratio
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
        
        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) * 100
        
        # Win rate
        winning_trades = sum(1 for i in range(1, len(portfolio_values)) 
                           if portfolio_values[i] > portfolio_values[i-1] and actions[i-1] != 0)
        total_trades = sum(1 for a in actions if a != 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Lưu kết quả
        self.results.update({
            'final_portfolio_value': final_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'portfolio_history': portfolio_values,
            'actions_history': actions,
            'prices_history': prices,
            'predictions_history': predictions,
            'trades_history': self.trading_env.trades
        })
        
        print(f"📊 Kết quả Backtest:")
        print(f"   Portfolio cuối: ${final_value:,.2f}")
        print(f"   Tổng lợi nhuận: {total_return:.2f}%")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")
        print(f"   Tỷ lệ thắng: {win_rate:.1f}%")
        print(f"   Tổng số giao dịch: {total_trades}")
    
    def plot_results(self):
        """Vẽ biểu đồ kết quả"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['Portfolio Performance', 'Price vs Predictions', 
                          'Trading Actions', 'LSTM Training Loss',
                          'RL Training Progress', 'Trade Distribution'],
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Portfolio performance
        dates = self.processed_data.index[-len(self.results['portfolio_history']):]
        fig.add_trace(
            go.Scatter(x=dates, y=self.results['portfolio_history'], 
                      name='Portfolio Value', line=dict(color='green')),
            row=1, col=1
        )
        
        # Price vs Predictions
        fig.add_trace(
            go.Scatter(x=dates[:-1], y=self.results['prices_history'], 
                      name='Actual Price', line=dict(color='blue')),
            row=1, col=2
        )
        
        # Denormalize predictions for plotting
        if len(self.results['predictions_history']) > 0:
            pred_prices = []
            for i, pred in enumerate(self.results['predictions_history']):
                if pred != 0:
                    # Simple denormalization (approximate)
                    actual_price = self.results['prices_history'][i] if i < len(self.results['prices_history']) else 0
                    pred_prices.append(pred * self.processed_data['Close'].max())
                else:
                    pred_prices.append(0)
            
            fig.add_trace(
                go.Scatter(x=dates[:-1], y=pred_prices, 
                          name='LSTM Prediction', line=dict(color='red', dash='dash')),
                row=1, col=2
            )
        
        # Trading actions
        buy_signals = [i for i, a in enumerate(self.results['actions_history']) if a == 1]
        sell_signals = [i for i, a in enumerate(self.results['actions_history']) if a == 2]
        
        if buy_signals:
            fig.add_trace(
                go.Scatter(x=[dates[i] for i in buy_signals], 
                          y=[self.results['prices_history'][i] for i in buy_signals],
                          mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'),
                          name='Buy'),
                row=2, col=1
            )
        
        if sell_signals:
            fig.add_trace(
                go.Scatter(x=[dates[i] for i in sell_signals], 
                          y=[self.results['prices_history'][i] for i in sell_signals],
                          mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'),
                          name='Sell'),
                row=2, col=1
            )
        
        # Price background
        fig.add_trace(
            go.Scatter(x=dates[:-1], y=self.results['prices_history'],
                      line=dict(color='lightblue'), showlegend=False),
            row=2, col=1
        )
        
        # LSTM training history
        if hasattr(self.lstm_model, 'history') and self.lstm_model.history:
            epochs = range(1, len(self.lstm_model.history.history['loss']) + 1)
            fig.add_trace(
                go.Scatter(x=list(epochs), y=self.lstm_model.history.history['loss'],
                          name='Training Loss', line=dict(color='orange')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=list(epochs), y=self.lstm_model.history.history['val_loss'],
                          name='Validation Loss', line=dict(color='red')),
                row=2, col=2
            )
        
        # RL training progress
        if 'rl_training_values' in self.results:
            episodes = range(len(self.results['rl_training_values']))
            fig.add_trace(
                go.Scatter(x=list(episodes), y=self.results['rl_training_values'],
                          name='Portfolio Value', line=dict(color='purple')),
                row=3, col=1
            )
        
        # Trade distribution
        action_counts = [self.results['actions_history'].count(i) for i in range(3)]
        fig.add_trace(
            go.Bar(x=['Hold', 'Buy', 'Sell'], y=action_counts,
                   marker_color=['gray', 'green', 'red']),
            row=3, col=2
        )
        
        fig.update_layout(height=1200, showlegend=True, 
                         title_text=f"AI Trading System Results - {self.symbol}")
        fig.show()
    
    def get_recommendation(self):
        """Đưa ra khuyến nghị cuối cùng"""
        if not self.results:
            return "Chưa có dữ liệu để đưa ra khuyến nghị"
        
        # Phân tích kết quả
        total_return = self.results['total_return']
        sharpe_ratio = self.results['sharpe_ratio']
        max_drawdown = self.results['max_drawdown']
        win_rate = self.results['win_rate']
        
        # Lấy action gần nhất từ agent
        recent_state = self.trading_env._get_state()
        recommended_action = self.rl_agent.act(recent_state)
        
        action_names = ['HOLD 📊', 'BUY 📈', 'SELL 📉']
        recommendation = action_names[recommended_action]
        
        # Đánh giá độ tin cậy
        confidence_score = 0
        reasons = []
        
        if total_return > 0:
            confidence_score += 0.3
            reasons.append(f"Lợi nhuận dương {total_return:.1f}%")
        
        if sharpe_ratio > 1:
            confidence_score += 0.2
            reasons.append(f"Sharpe ratio tốt ({sharpe_ratio:.2f})")
        
        if win_rate > 50:
            confidence_score += 0.2
            reasons.append(f"Tỷ lệ thắng cao ({win_rate:.1f}%)")
        
        if max_drawdown < 20:
            confidence_score += 0.2
            reasons.append(f"Rủi ro thấp (Max DD: {max_drawdown:.1f}%)")
        
        if self.results['lstm_test_rmse'] < 0.05:
            confidence_score += 0.1
            reasons.append("Mô hình dự đoán chính xác")
        
        confidence_level = "Thấp" if confidence_score < 0.4 else "Trung bình" if confidence_score < 0.7 else "Cao"
        
        report = f"""
🤖 KHUYẾN NGHỊ TỪ AI TRADING SYSTEM
{'='*50}

📊 MÃ CỔ PHIẾU: {self.symbol}
🎯 HÀNH ĐỘNG ĐỀ XUẤT: {recommendation}
🎚️ ĐỘ TIN CẬY: {confidence_level} ({confidence_score:.1%})

📈 HIỆU SUẤT BACKTEST:
   • Tổng lợi nhuận: {total_return:.1f}%
   • Sharpe Ratio: {sharpe_ratio:.2f}
   • Max Drawdown: {max_drawdown:.1f}%
   • Tỷ lệ thắng: {win_rate:.1f}%
   • Tổng giao dịch: {self.results['total_trades']}

🧠 PHÂN TÍCH AI:
   • LSTM RMSE: {self.results['lstm_test_rmse']:.4f}
   • Portfolio cuối: ${self.results['final_portfolio_value']:,.2f}

💡 LÝ DO:
"""
        for reason in reasons:
            report += f"   • {reason}\n"
        
        if not reasons:
            report += "   • Cần thêm dữ liệu để đánh giá chính xác\n"
        
        report += f"""
⚠️  LƯU Ý:
   • Đây là khuyến nghị từ AI, không phải lời khuyên đầu tư
   • Luôn cân nhắc rủi ro và tình hình tài chính cá nhân
   • Kết hợp với phân tích cơ bản và tin tức thị trường
   • Chỉ đầu tư số tiền có thể chấp nhận mất

🔄 CẬP NHẬT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
    
    def run_complete_analysis(self, period='2y', lstm_epochs=50, rl_episodes=500):
        """Chạy toàn bộ phân tích"""
        print(f"🚀 BẮT ĐẦU PHÂN TÍCH HOÀN CHỈNH CHO {self.symbol}")
        print("="*60)
        
        # Bước 1: Tải và chuẩn bị dữ liệu
        if not self.load_and_prepare_data(period):
            print("❌ Không thể tải dữ liệu!")
            return False
        
        # Bước 2: Huấn luyện LSTM
        self.train_lstm_model(epochs=lstm_epochs)
        
        # Bước 3: Huấn luyện RL Agent
        self.train_rl_agent(episodes=rl_episodes)
        
        # Bước 4: Backtest
        self.backtest_strategy()
        
        # Bước 5: Hiển thị kết quả
        print("\n" + self.get_recommendation())
        
        # Bước 6: Vẽ biểu đồ
        self.plot_results()
        
        return True

def main():
    """Hàm chính để chạy hệ thống"""
    print("🤖 AI TRADING SYSTEM WITH LSTM + RL")
    print("="*50)
    
    # Nhập thông tin
    symbol = input("📊 Nhập mã cổ phiếu (VD: AAPL): ").upper() or "AAPL"
    
    try:
        initial_balance = float(input("💰 Nhập số tiền ban đầu ($): ") or "10000")
    except:
        initial_balance = 10000
    
    period_options = {"1": "3mo", "2": "6mo", "3": "1y", "4": "2y"}
    print("\nChọn khoảng thời gian:")
    for key, value in period_options.items():
        print(f"  {key}. {value}")
    
    period_choice = input("Lựa chọn (1-4): ") or "3"
    period = period_options.get(period_choice, "2y")
    
    # Khởi tạo hệ thống
    trading_system = AITradingSystem(symbol, initial_balance)
    
    # Chạy phân tích
    success = trading_system.run_complete_analysis(
        period=period,
        lstm_epochs=50,  # Giảm epochs để demo nhanh hơn
        rl_episodes=500  # Giảm episodes để demo nhanh hơn
    )
    
    if success:
        print("\n✅ Phân tích hoàn thành!")
        
        # Lưu kết quả
        save_choice = input("\n💾 Lưu kết quả vào file? (y/n): ").lower()
        if save_choice == 'y':
            filename = f"{symbol}_trading_results_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(trading_system.get_recommendation())
            print(f"📁 Kết quả đã được lưu vào {filename}")
    else:
        print("\n❌ Có lỗi xảy ra trong quá trình phân tích!")

# Thêm class để demo nhanh
class QuickDemo:
    """Demo nhanh với dữ liệu mẫu"""
    
    @staticmethod
    def generate_sample_data(days=252):
        """Tạo dữ liệu mẫu"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Tạo giá cổ phiếu với xu hướng tăng và volatility
        price = 100
        prices = []
        volumes = []
        
        for i in range(days):
            # Random walk với drift
            price += np.random.normal(0.1, 2)  # Slight upward trend
            price = max(price, 10)  # Không cho giá xuống dưới 10
            prices.append(price)
            
            # Volume ngẫu nhiên
            volumes.append(np.random.randint(100000, 1000000))
        
        # Tạo OHLC từ Close
        data = pd.DataFrame({
            'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
            'High': [p * np.random.uniform(1.0, 1.05) for p in prices],
            'Low': [p * np.random.uniform(0.95, 1.0) for p in prices],
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        return data
    
    @staticmethod
    def run_demo():
        """Chạy demo với dữ liệu mẫu"""
        print("🎮 CHẠY DEMO NHANH VỚI DỮ LIỆU MẪU")
        print("="*50)
        
        # Tạo dữ liệu mẫu
        sample_data = QuickDemo.generate_sample_data(200)
        
        # Khởi tạo system
        demo_system = AITradingSystem("DEMO", 10000)
        demo_system.data_processor.data = sample_data
        
        # Thêm technical indicators
        demo_system.processed_data = demo_system.data_processor.add_technical_indicators(sample_data)
        
        # Chuẩn bị dữ liệu LSTM
        X, y, feature_data = demo_system.data_processor.prepare_lstm_data(demo_system.processed_data, lookback_days=30)
        
        # Chia train/test
        train_size = int(len(X) * 0.8)
        demo_system.X_train, demo_system.X_test = X[:train_size], X[train_size:]
        demo_system.y_train, demo_system.y_test = y[:train_size], y[train_size:]
        
        print("✅ Dữ liệu mẫu đã sẵn sàng!")
        
        # Huấn luyện nhanh với ít epochs
        demo_system.train_lstm_model(epochs=20)
        demo_system.train_rl_agent(episodes=100)
        demo_system.backtest_strategy()
        
        # Hiển thị kết quả
        print(demo_system.get_recommendation())
        
        return demo_system

if __name__ == "__main__":
    print("Chọn chế độ:")
    print("1. Phân tích thực tế với dữ liệu Yahoo Finance")
    print("2. Demo nhanh với dữ liệu mẫu")
    
    choice = input("Lựa chọn (1-2): ") or "2"
    
    if choice == "1":
        main()
    else:
        demo = QuickDemo.run_demo()
        
        # Vẽ biểu đồ cho demo
        plot_choice = input("\n📊 Hiển thị biểu đồ? (y/n): ").lower()
        if plot_choice == 'y':
            demo.plot_results()

# Thêm utility functions
def calculate_portfolio_metrics(returns):
    """Tính các chỉ số hiệu suất portfolio"""
    if len(returns) == 0:
        return {}
    
    returns = np.array(returns)
    
    # Annual return
    annual_return = np.mean(returns) * 252
    
    # Volatility
    volatility = np.std(returns) * np.sqrt(252)
    
    # Sharpe ratio
    sharpe = annual_return / volatility if volatility != 0 else 0
    
    # Max drawdown
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (running_max - cumulative) / running_max
    max_drawdown = np.max(drawdown)
    
    # Calmar ratio
    calmar = annual_return / max_drawdown if max_drawdown != 0 else 0
    
    return {
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar
    }





