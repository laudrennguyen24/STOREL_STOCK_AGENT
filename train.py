from STOCK import AITradingSystem, QuickDemo

def train_real(symbol="AAPL", initial_balance=10000, period="1y"):
    """Train using real stock data."""
    trading_system = AITradingSystem(symbol, initial_balance)
    if trading_system.run_complete_analysis(
        period=period,
        lstm_epochs=50,
        rl_episodes=500
    ):
        print("✅ Real training complete!")
    else:
        print("❌ Failed to train on real data.")

def train_demo():
    """Train using demo synthetic data."""
    demo = QuickDemo.run_demo()
    return demo

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM and RL models")
    parser.add_argument("--symbol", type=str, help="Stock symbol", default="AAPL")
    parser.add_argument("--balance", type=float, help="Initial balance", default=10000)
    parser.add_argument("--period", type=str, help="Period for fetching data", default="1y")
    parser.add_argument("--demo", action="store_true", help="Run quick demo with synthetic data")

    args = parser.parse_args()

    if args.demo:
        train_demo()
    else:
        train_real(args.symbol, args.balance, args.period)
