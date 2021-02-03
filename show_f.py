import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):  # 计算单次模拟的回报率和协方差
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets * [1. / num_assets, ], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets * [1. / num_assets, ], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]


def efficient_return(mean_returns, cov_matrix, target):  # 用于计算给定回报的最有效投资组合
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets * [1. / num_assets, ], args=args, method='SLSQP',
                          bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):  # 用于计算一系列目标回报并计算每个回报对应的有效投资组合
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('fivethirtyeight')
    np.random.seed(777)

    stocks = ['600519.SH', '000858.SZ', '000568.SZ', '002304.SZ', '002415.SZ', '300015.SZ']
    table = pd.read_csv('CSI_selected_closes.csv', index_col=0)
    table = table.dropna()
    returns = table.pct_change()  # 回报率
    mean_returns = returns.mean()  # 回报方差
    cov_matrix = returns.cov()  # 回报协方差
    risk_free_rate = float(input('短期无风险利率:'))  # 无风险短期利率

    shares = pd.read_json('shares.json')
    df = pd.read_json('random_result.json')
    results = df.loc[:, ['cov', 'ret', 'shapes']]
    results = results.T
    weights = df.loc[:, [shares.编码[i] for i in range(len(shares.编码))]]
    table = pd.read_csv('CSI_selected_closes.csv', index_col=0)
    table = table.dropna()

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x, index=shares.名称.values, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x, index=shares.名称.values, columns=['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print("-" * 80)
    print("最大夏普指数\n")
    print("预期收益率:", round(rp, 2))
    print("预期波动性:", round(sdp, 2))
    print("\n占比：\n")
    print('\t'.join([i for i in shares.名称.values]))
    print('\t'.join([str(j) for j in max_sharpe_allocation.values[0]]))

    print("\n" + "-" * 80)
    print("最小波动性\n")
    print("预期收益率:", round(rp_min, 2))
    print("预期波动性:", round(sdp_min, 2))
    print("\n占比：\n")
    print('\t'.join([i for i in shares.名称.values]))
    print('\t'.join([str(j) for j in min_vol_allocation.values[0]]))

    plt.figure(figsize=(16, 9))
    plt.scatter(results.loc['cov', :].values, results.loc['ret', :].values, c=results.loc['shapes', :].values,
                cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp, rp, marker='*', color='r', s=500, label='最大夏普指数')
    plt.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='最小波动率')

    target = np.linspace(rp_min, round(results.loc['ret', :].values.max(), 2), 100)  # 参数：起始点，终止点，采样的点个数
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black',
             label='有效前沿')
    plt.title('MPT')
    plt.xlabel('波动性')
    plt.ylabel('收益率')
    plt.legend(labelspacing=0.8)
    plt.show()
