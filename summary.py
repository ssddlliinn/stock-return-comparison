import sys
import pandas as pd
# sys.path.append(r'C:\Users\YourUsername\Desktop\data_program')  # Commented out for deployment
from data_general import get_finmind_data # type: ignore


# 合併單一股票的日股價資料與除權息紀錄
def summary_monthly_data(stock_id, market='tw', start_date = '1996-01-01', end_date = None):
    # 日股價資料
    dataset_name = 'taiwan_stock_daily' if market=='tw' else 'us_stock_price'
    price_col = 'close' if market=='tw' else 'Adj_Close'
    
    price_data = get_finmind_data(
        dataset_name=dataset_name,
        stock_id=stock_id, 
        start_date=start_date,
        end_date=end_date
        )
    
    if price_data.empty:
        return pd.DataFrame()
    
    # 確定日期資料是時間格式
    if 'date' in price_data.columns:
        price_data['date'] = pd.to_datetime(price_data['date'])
        price_data = price_data.set_index('date')
    # 以月分為單位做groupby，取最後交易日股價並做百分比增減，得到月份報酬資料(日期以最後交易日表示)
    monthly_last_close = price_data.groupby([pd.Grouper(freq='ME')])[price_col].last() # 一般groupby只能以日為單位，要用Grouper才能以年、月、劑等頻率整併資料
    #TODO: 處理0050分割問題
    if stock_id == '0050':
        # print(monthly_last_close.tail(10))
        condition = monthly_last_close.index > '2025-06-18'
        monthly_last_close.loc[condition] = monthly_last_close.loc[condition] * 4
    
    monthly_summary = pd.DataFrame({'monthly_return':monthly_last_close.pct_change()})
    
    # monthly_summary = monthly_summary.reset_index()
    monthly_summary.index.rename('month', inplace=True)
    # 把日期資料換成以月分資料表示
    monthly_summary.index = monthly_summary.index.to_period('M')
    
    # 除權息資料
    if market=='tw':
        div_data = get_finmind_data(
            dataset_name='taiwan_stock_dividend_result',
            stock_id=stock_id, 
            start_date=start_date,
            end_date=end_date
            )
        
        if not div_data.empty:
            # 確定日期資料是時間格式
            if 'date' in div_data.columns:
                div_data['date'] = pd.to_datetime(div_data['date'])
                div_data = div_data.set_index('date')
            # 把日期資料換成以月分資料表示
            div_data['month'] = div_data.index.to_period('M')
            div_data['dividend_yield'] = div_data['stock_and_cache_dividend'] / div_data['before_price'] #計算股價殖利率
            monthly_div = div_data.groupby('month')['dividend_yield'].sum()
            

            # 以股票代碼及年月為合併基準，把月報酬跟除權息殖利率合併
            # index:month、columns=['stock_id', 'monthly_return', 'dividend_yield']
            #TODO: expect columns for US stock, index:'month', columns:['stock_id', monthly_return_adj']
            final_df = pd.merge(monthly_summary, monthly_div, on=['month'], how='left')
            final_df['stock_id'] = stock_id
            final_df = final_df.loc[:, ['stock_id', 'monthly_return', 'dividend_yield']]
            final_df = final_df.fillna(0)
            return final_df
    
    final_df = monthly_summary.copy()
    final_df['stock_id'] = stock_id
    final_df['dividend_yield'] = 0
    final_df = final_df.loc[:, ['stock_id', 'monthly_return', 'dividend_yield']]
    
    final_df = final_df.fillna(0)
    return final_df