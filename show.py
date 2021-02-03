import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False

    shares = pd.read_json('shares.json')
    df = pd.read_json('random_result.json')
    results = df.loc[:, ['cov', 'ret', 'shapes']]
    results = results.T
    weights = df.loc[:, [shares.编码[i] for i in range(len(shares.编码))]]
    table = pd.read_csv('CSI_selected_closes.csv', index_col=0)
    table = table.dropna()

    max_sharpe_idx = np.argmax(results.loc['shapes'])
    sdp = results.loc['cov', max_sharpe_idx]
    rp = results.loc['ret', max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights.loc[max_sharpe_idx].values, index=shares.名称, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results.loc['cov'])
    sdp_min, rp_min = results.loc['cov', min_vol_idx], results.loc['ret', min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights.loc[min_vol_idx].values, index=shares.名称, columns=['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp, 2))
    print("Annualised Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("-" * 80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min, 2))
    print("Annualised Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)

    plt.figure(figsize=(16, 9))
    plt.scatter(results.loc['cov', :], results.loc['ret', :], c=results.loc['shapes', :], cmap='YlGnBu', marker='o',
                s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp, rp, marker='*', color='r', s=500, label='最大夏普指数')
    plt.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='最小波动率')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    plt.show()
