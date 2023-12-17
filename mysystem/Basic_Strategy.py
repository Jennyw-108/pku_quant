import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 基本日内反转策略
class Basic_intraday_reversal_strategy:
    
    def __init__(self, open_close_threshold, high_low_threshold):
        self.open_close_threshold = open_close_threshold
        self.high_low_threshold = high_low_threshold
        self.signals = pd.DataFrame()
        
    
    # 生成交易信号 
    def generate_signals(self, data):

        # 计算价格变化
        data['open_close_change'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        data['high_low_change'] = (data['high'] - data['low']) / data['close'].shift(1)

        # 初始化为0
        self.signals = pd.DataFrame(index=data.index)
        self.signals['signal'] = 0.0
       
        # 买入信号
        self.signals['signal'] = np.where(data['open_close_change'] < -self.open_close_threshold, 1.0, self.signals['signal']) 
        self.signals['signal'] = np.where(data['high_low_change'] < -self.high_low_threshold, 1.0, self.signals['signal'])

        # 卖出信号
        self.signals['signal'] = np.where(data['open_close_change'] > self.open_close_threshold, -1.0, 0.0)  
        self.signals['signal'] = np.where(data['high_low_change'] > self.high_low_threshold, -1.0, self.signals['signal']) 
       
        return self.signals
    
    # 计算策略表现指标
    def calculate_performance_metrics(self, data, signals):
        data['strategy'] = signals['signal'].shift(1)

        # 计算净值曲线
        data['returns'] = data['strategy'] * data['returns']
        data['cumulative_returns'] = (1 + data['returns']).cumprod()

        # 计算超额收益
        excess_returns = data['cumulative_returns'] - data['cumulative_returns'].cummax()
        data['excess_returns'] = excess_returns

        # 计算年化收益率
        annual_returns = (data['cumulative_returns'].iloc[-1])**(252/len(data.index)) - 1
        data['annual_returns'] = annual_returns

        # 计算年化波动
        annual_volatility = data['returns'].std() * np.sqrt(252)
        data['annual_volatility'] = annual_volatility

        # 计算夏普比率
        sharpe_ratio = annual_returns / annual_volatility
        data['sharpe_ratio'] = sharpe_ratio

        # 计算最大回撤
        max_drawdown = (data['cumulative_returns'] / data['cumulative_returns'].cummax() - 1).min()
        data['max_drawdown'] = max_drawdown

        return data
    
    def basic_plot(self, data):
    
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['close'], label='Close Price')
        plt.plot(data.loc[data['signal'] == 1.0].index, data['close'][data['signal'] == 1.0], '^', markersize=3, color='g', label='Buy Signal')
        plt.plot(data.loc[data['signal'] == -1.0].index, data['close'][data['signal'] == -1.0], 'v', markersize=3, color='r', label='Sell Signal')
        plt.title('Basic Intraday Reversal Strategy')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['cumulative_returns'], label='Strategy Returns')
        plt.plot(data.index, data[['excess_returns','annual_returns', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']])
        plt.title('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.show()

   
    







