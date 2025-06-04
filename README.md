# AI Trading System 📈🤖

This project implements an **AI-driven trading system** using:
- LSTM neural networks for price prediction
- Reinforcement Learning (Deep Q-Network) for trading decisions
- Technical indicators for feature enhancement
- `plotly` visualization for results

---

## 📂 Project Structure

```
.
├── STOCK.py              # Main script containing data, model, and RL logic
├── train.py              # Script to train LSTM and RL agent
├── test.py               # Script to test/backtest the model
├── requirements.txt      # Required Python packages
└── README.md             # Project overview and usage
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train with Demo Data

```bash
python train.py --demo
```

### 3. Train with Real Stock Data

```bash
python train.py --symbol AAPL --balance 10000 --period 1y
```

### 4. Run Test/Backtest

```bash
python test.py
```

---

## 🧠 Components

- **LSTM Model:** Forecasts stock prices using historical and technical features.
- **RL Agent:** Learns optimal trading strategies using Q-learning.
- **Trading Environment:** Simulates portfolio management with slippage and transaction cost.

---

## 📊 Visualizations

Interactive charts include:
- Portfolio value over time
- Buy/Sell action markers
- LSTM prediction vs. actual price
- Training loss and performance metrics

---

## 📌 Notes

- This system is for educational purposes and not financial advice.
- Ensure the `vietfin` API and `yfinance` sources are accessible.

---
