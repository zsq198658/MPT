import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):  # 计算单次模拟的回报率和协方差
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):  # 模拟各stocks占比，
    results = np.zeros((3, num_portfolios))  # 第一行波动性，第二行收益率，第三行夏普值
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    df = np.concatenate((results.T, weights_record), axis=1)
    df = pd.DataFrame(df)
    df.columns = ['cov', 'ret', 'shapes'] + [stocks[i] for i in range(len(stocks))]
    df.to_json('random_result.json')
    return results, weights_record


if __name__ == '__main__':
    plt.style.use('fivethirtyeight')
    np.random.seed(777)

    stocks = ['600519.SH', '000858.SZ', '000568.SZ', '002304.SZ', '002415.SZ', '300015.SZ']
    table = pd.read_csv('CSI_selected_closes.csv', index_col=0)
    table = table.dropna()
    returns = table.pct_change()  # 回报率
    mean_returns = returns.mean()  # 回报方差
    cov_matrix = returns.cov()  # 回报协方差
    num_portfolios = int(input('组合随机模拟次数:'))  # 模拟次数
    risk_free_rate = float(input('短期无风险利率:'))  # 无风险短期利率
    random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)
