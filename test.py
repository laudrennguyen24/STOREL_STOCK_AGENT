from STOCK import AITradingSystem

def test_system(symbol="AAPL", initial_balance=10000, period="1y"):
    """Test the trading system performance."""
    trading_system = AITradingSystem(symbol, initial_balance)

    if not trading_system.load_and_prepare_data(period):
        print("❌ Data loading failed.")
        return

    # Train LSTM only (skip RL to speed up)
    trading_system.train_lstm_model(epochs=10)

    # Optional: Load pre-trained RL agent or run fewer episodes
    trading_system.train_rl_agent(episodes=100)

    # Run backtest
    trading_system.backtest_strategy()

    # Print recommendation
    print(trading_system.get_recommendation())

if __name__ == "__main__":
    test_system()
