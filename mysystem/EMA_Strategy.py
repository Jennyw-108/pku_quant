import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class EMA_IntradayReversalStrategy:
    def __init__(self, data):
        self.data = data
        self.signals = pd.Series(index=data.index)
        self.ema_short = data['close'].ewm(span=5).mean()
        self.ema_long = data['close'].ewm(span=10).mean()
        self.net_value = None

    def generate_signals(self):
        # 生成交易信号
        cross_above = (self.ema_short > self.ema_long) & (self.ema_short.shift(1) <= self.ema_long.shift(1))
        cross_below = (self.ema_short < self.ema_long) & (self.ema_short.shift(1) >= self.ema_long.shift(1))
        self.signals = pd.Series(0, index=self.data.index)  # 初始化信号为0
        self.signals[cross_above] = 1.0
        self.signals[cross_below] = -1.0
        return self.signals

    def calculate_performance_metrics(self):
        # 计算净值曲线、超额收益、年化收益、年化波动、夏普比率和最大回撤
        self.net_value = (1 + self.signals.shift(1) * self.data['close'].pct_change()).cumprod()
        excess_returns = self.net_value - self.data['close'].pct_change().cumsum()
        annualized_returns = self.net_value[-1] ** (252 / len(self.net_value.index)) - 1
        annualized_volatility = self.net_value.pct_change().std() * np.sqrt(252)
        sharpe_ratio = annualized_returns / annualized_volatility
        cum_returns = self.net_value.pct_change().add(1).cumprod()
        max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()

        return {
            'excess_returns': excess_returns[-1],
            'annualized_returns': annualized_returns,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def plot_signals_and_performance(self):

        # 绘制净值曲线
        plt.subplot(2, 1, 1)
        plt.plot(self.data['close'], label='Close Price', linewidth=1.5)
        plt.plot(self.net_value, label='Net Value', linewidth=1.5)
        plt.title('Intraday Reversal Strategy with Net Value')
        plt.xlabel('Date')
        plt.ylabel('Price/Net Value')
        plt.legend()
        plt.grid(True)

        # 绘制交易信号
        plt.subplot(2, 1, 2)
        plt.plot(self.data['close'], label='Close Price', linewidth=1.5)
        plt.plot(self.ema_short, label='5-day EMA', linestyle='--', linewidth=1.5)
        plt.plot(self.ema_long, label='10-day EMA', linestyle='--', linewidth=1.5)
        plt.plot(self.signals[self.signals == 1.0].index,
                 self.data['close'][self.signals == 1.0],
                 '^', markersize=10, color='g', label='Buy Signal')
        plt.plot(self.signals[self.signals == -1.0].index,
                 self.data['close'][self.signals == -1.0],
                 'v', markersize=10, color='r', label='Sell Signal')
        plt.title('Intraday Reversal Strategy with EMA Signals')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)

        plt.show()