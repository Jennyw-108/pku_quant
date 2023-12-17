# 1. 导入系统库函数
from jqdatasdk import *

# 2. 导入其它库
import datetime
import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# 1. 初始化函数, 交易相关全局变量
def trade_vars():
    # （g.为全局变量）
    # 持仓数量
    g.stocknum = 6
    # 交易日计时器
    g.days = 0
    # 调仓频率
    g.refresh_rate = 10
    # 初始收益率
    g.ret = -0.05
    # 交易状态，初始不进行交易
    g.if_trade = False
    # 大盘走势参考指数，上证指数
    g.ref_index_stock = '000001.XSHG'
    # 使用的股票池，使用沪深300
    g.stock_pool = '000300.XSHG'
    g.list_to_buy = []
    g.list_to_sell = []


# 初始化函数，设定基准等等
def initialize(context):
    trade_vars()
    set_test_conditions(context)
    run_daily(before_market_open, time='before_open', reference_security=g.ref_index_stock)
    run_daily(market_open, time='open', reference_security=g.ref_index_stock)
    run_daily(after_market_close, time='after_close', reference_security=g.ref_index_stock)


# 设定回测相关参数
def set_test_conditions(context):
    set_benchmark(g.ref_index_stock)
    set_option('use_real_price', True)
    log.info('初始函数开始运行且全局只运行一次')
    set_option("avoid_future_data", True)
    set_slippage(PriceRelatedSlippage(0.00246))
    # 股票类每笔交易时的手续费,佣金，印花税等设定
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5),
                   type='stock')


# 2. 开盘前运行函数 
def before_market_open(context):
    log.info('函数运行时间(before_market_open)：' + str(context.current_dt.time()))
    # 调仓日判断
    if g.days % g.refresh_rate == 0:
        g.if_trade = True  
        stock_lst = filter_paused_stocks(context)
        g.list_to_buy = stocks_to_buy(context, stock_lst)
        log.info('get buy list')
        if len(g.list_to_buy):
            g.list_to_sell = stocks_to_sell(context, g.list_to_buy)
            log.info('get sell list')
        else:
            g.if_trade = False
    g.days += 1


# 过滤停牌的股票
def filter_paused_stocks(context):
    stocks = get_index_stocks(g.stock_pool)
    paused_stocks = []
    current_data = get_current_data()
    for i in stocks:
        paused_stocks.append(current_data[i].paused)
    log.info('停牌股票数量（%s / %s）' % (len(paused_stocks), len(stocks)))
    df_paused_stocks = pd.DataFrame({'paused_stocks': paused_stocks}, index=stocks)
    stock_lst = list(df_paused_stocks.index[df_paused_stocks.paused_stocks == False])
    log.info('过滤停牌后可选股票数量' + str(len(stock_lst)))
    return stock_lst


# 生成买入股票列表
def stocks_to_buy(context, stock_lst):
    list_to_buy = []
    today = context.current_dt
    yesterday = today - datetime.timedelta(days=1)
    day2 = today - datetime.timedelta(days=5)
    log.info('yesterday' + str(yesterday))

    # 获取大盘收盘价
    hs300_clos = get_price(g.ref_index_stock, day2, yesterday, fq='pre')['close']
    log.info('hs300_clos len:' + str(len(hs300_clos)))
    if len(hs300_clos):
        hs300_ret = hs300_clos[-1] / hs300_clos[0] - 1
        if hs300_ret > g.ret:
            factor = get_mdl(context, stock_lst)
            list_to_buy = list(factor.index[:g.stocknum])
    return list_to_buy


# 生成卖出股票列表
def stocks_to_sell(context, list_to_buy):
    list_to_sell = []
    today = context.current_dt
    yesterday = today - datetime.timedelta(days=1)
    day2 = today - datetime.timedelta(days=5)
    hs300_clos = get_price(g.ref_index_stock, day2, yesterday, fq='pre')['close']
    if len(hs300_clos):
        hs300_ret = hs300_clos[-1] / hs300_clos[0] - 1
        for stock_sell in context.portfolio.positions:
            if hs300_ret <= g.ret:
                list_to_sell.append(stock_sell)
            else:
                if context.portfolio.positions[stock_sell].price / \
                        context.portfolio.positions[stock_sell].avg_cost < 0.95 or \
                        stock_sell not in list_to_buy:
                    list_to_sell.append(stock_sell)
    return list_to_sell


# 模型训练和预测
def get_mdl(context, stock_lst):
    df, x, y = data_prepare(context, stock_lst)
    x1, y1 = feature_selection(df, x, y)
    mdl = train_mdl(x1, y1)
    predict = pd.DataFrame(mdl.predict(x1),index=y1.index,
                           columns=['市值'])
    diff = y1 - predict  
    diff = pd.DataFrame(diff, index = y.index, columns = ['diff'])
    diff = diff.sort_values(by='市值', ascending=True)
    return diff


# 机器学习模型
def data_prepare(context, stock_lst):
    q = query(valuation.code, valuation.market_cap,
              balance.total_current_assets - balance.total_current_liability,
              balance.total_liability - balance.total_assets,
              balance.total_liability / balance.equities_parent_company_owners,
              (balance.total_assets - balance.total_current_assets) / balance.total_assets,
              balance.equities_parent_company_owners / balance.total_assets,
              indicator.inc_total_revenue_year_on_year, valuation.turnover_ratio, valuation.pe_ratio,
              valuation.pb_ratio, valuation.ps_ratio, indicator.roa).filter(valuation.code.in_(stock_lst))
    df = get_fundamentals(q, date=None)
    df.columns = ['code', '市值', '净营运资本','净债务', '产权比率', '非流动资产比率',
                  '股东权益比率', '营收增长率', '换手率', 'PE', 'PB', 'PS', '总资产收益率']
    df.index = df.code.values
    del df['code']

    # 主要时间点设定
    today = context.current_dt
    delta50 = datetime.timedelta(days=50)
    delta1 = datetime.timedelta(days=1)
    # 50日前作为一个历史节点
    history = today - delta50
    yesterday = today - delta1

    # 获取最新的技术因子，如动量线、成交量、累计能量线、平均差、指数移动平均、移动平均、乖离率等因子，时间范围都设为10天
    df['动量线'] = list(MTM(df.index, yesterday,
                            timeperiod=10, unit='1d',
                            include_now=True,
                            fq_ref_date=None).values())
    df['成交量'] = list(VOL(df.index, yesterday, M1=10,
                            unit='1d', include_now=True,
                            fq_ref_date=None)[0].values())
    df['累计能量线'] = list(OBV(df.index, check_date=yesterday,
                                timeperiod=10).values())
    df['平均差'] = list(DMA(df.index, yesterday, N1=10,
                            unit='1d', include_now=True,
                            fq_ref_date=None)[0].values())
    df['指数移动平均'] = list(EMA(df.index, yesterday, timeperiod=10,
                                  unit='1d', include_now=True,
                                  fq_ref_date=None).values())
    df['移动平均'] = list(MA(df.index, yesterday, timeperiod=10,
                             unit='1d', include_now=True,
                             fq_ref_date=None).values())
    df['乖离率'] = list(BIAS(df.index, yesterday, N1=10,
                             unit='1d', include_now=True,
                             fq_ref_date=None)[0].values())
    # 把数据表中的空值用0来代替
    df.fillna(0,inplace=True)
    # 获取股票前一日的收盘价
    df['close1'] = get_price(stock_lst,
                             start_date=yesterday,
                             end_date=yesterday,
                             fq='pre', panel=False)['close'].T
    # 获取股票50日前的收盘价
    df['close2'] = get_price(stock_lst,
                             start_date=history,
                             end_date=history,
                             fq='pre', panel=False)['close'].T

    # 计算出收益
    df['return'] = df['close1'] / df['close2'] - 1
    # 如果收益大于平均水平，则标记为1，否则标记为0
    df['signal'] = np.where(df['return'] < df['return'].mean(), 0, 1)

    # dropna 会丢掉所有数据
    # log.info('df len pre' + str(len(df)))
    # df = df.dropna()
    # log.info('df len after' + str(len(df)))

    # 把因子值作为样本的特征，所以要去掉刚刚添加的几个字段
    x = df.drop(['close1', 'close2', 'return', 'signal'], axis=1)
    # 把signal作为分类标签
    y = df['signal']
    x = x.fillna(0)
    y = y.fillna(0)
    return df, x, y


# 特征选择
def feature_selection(df, x, y):
    clf = RandomForestClassifier()
    clf.fit(x, y)
    factor_weight = pd.DataFrame({'features': list(x.columns),
                                  'importance': clf.feature_importances_}).sort_values(
        by='importance', ascending=False)

    # 选出最重要的5个特征
    features = factor_weight['features'][:5]
    x_new = df[features]
    y_new = df['市值']
    x_new = x_new.fillna(0)
    y_new = y_new.fillna(0)
    df = df.dropna()
    return x_new, y_new


# 训练机器学习模型
def train_mdl(x, y):
    # 训练支持向量机
    # svr = SVR()
    # model = svr.fit(x, y)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    reg = RandomForestRegressor()  # random_state=20
    model = reg.fit(x, y)
    return model


# 3. 开盘时运行函数 
def market_open(context):
    log.info('函数运行时间(market_open):' + str(context.current_dt.time()))
    if g.if_trade:
        sell_operation(g.list_to_sell)
        buy_operation(context, g.list_to_buy)
    g.if_trade = False


# 执行买入操作
def buy_operation(context, list_to_buy):
    if len(context.portfolio.positions) < g.stocknum:
        num = g.stocknum - len(context.portfolio.positions)
        if num == 0:
            cash = 0
        else:
            cash = context.portfolio.cash / num
    else:
        cash = 0
        num = 0

    for stock_sell in list_to_buy[:num + 1]:
        order_target_value(stock_sell, cash)
        num = num - 1
        if num == 0:
            break

# 执行卖出操作
def sell_operation(list_to_sell):
    for stock_sell in list_to_sell:
        order_target_value(stock_sell, 0)


# 4. 收盘后运行函数 
def after_market_close(context):
    log.info(str('函数运行时间(after_market_close):' + str(context.current_dt.time())))
    # 得到当天所有成交记录
    trades = get_trades()
    for _trade in trades.values():
        log.info('成交记录：' + str(_trade))
    log.info('一天结束')


