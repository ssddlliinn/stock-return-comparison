from data import get_finmind_data
import pandas as pd

def summary_data(stock_id, start_date, end_date):
    price_data = get_finmind_data(
        dataset_name='taiwan_stock_daily',
        stock_id=stock_id
        )
    
    div_data = get_finmind_data(
        dataset_name='taiwan_stock_dividend_result',
        stock_id=stock_id
        )
    
    price_data['date'] = pd.to_datetime(price_data['date'])
    price_data = price_data.set_index('date')
    monthly_last_close = price_data.groupby([pd.Grouper(freq='ME')])['close'].last()
    monthly_summary = pd.DataFrame({'monthly_return':monthly_last_close.pct_change()})
    
    monthly_summary = monthly_summary.reset_index()
    monthly_summary.rename(columns={'date': 'month'}, inplace=True)
    monthly_summary['month'] = monthly_summary['month'].dt.to_period('M')


    div_data['date'] = pd.to_datetime(div_data['date'])
    div_data['month'] = div_data['date'].dt.to_period('M')
    div_data['yield'] = div_data['stock_and_cache_dividend'] / div_data['before_price']
    monthly_div = pd.DataFrame({'month':div_data['month'], 'dividend_yield':div_data['yield']})

    final_df = pd.merge(monthly_summary, monthly_div, on=['stock_id', 'month'], how='left')
    final_df['stock_id'] = stock_id
    return final_df