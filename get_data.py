import tushare as ts
import pandas as pd


def put_some_stock_price_into_one_df(shares_list):
    some_stock_price_df = pd.DataFrame
    for count, ticker in enumerate(shares_list):
        df = pd.read_csv('stock_day/{}.csv'.format(ticker))
        df.set_index('trade_date', inplace=True)

        df.rename(columns={'close': ticker}, inplace=True)
        df.drop(['Unnamed: 0', 'ts_code', 'open', 'high', 'low', 'pre_close',
                 'change', 'pct_chg', 'vol', 'amount'], 1, inplace=True)
        # drop中的‘1’代表删除列，‘0’代表删除行
        if some_stock_price_df.empty:
            some_stock_price_df = df
        else:
            some_stock_price_df = some_stock_price_df.join(df, how='outer')
    #        print(count)
    #    print(all_stock_price_df.head())
    some_stock_price_df.to_csv('CSI_selected_closes.csv')
    return some_stock_price_df


if __name__ == '__main__':
    # 获取日线数据------------------------------------------------------------------------------
    shares_list = ['']
    # shares_list_hk = ['']
    # df_hk = pro.hk_daily(ts_code=shares_list_hk, start_date='20190101', end_date='20190904')
    key = ''
    pro = ts.pro_api(key)
    for i in shares_list:
        df = pro.daily(ts_code=i, start_date='', end_date='')
        df.to_csv('stock_day/' + i + '.csv')
    put_some_stock_price_into_one_df(shares_list)
