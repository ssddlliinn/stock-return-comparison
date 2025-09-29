## 不管是什麼資料，日期都需要當作column且格視為datetime格式
import pandas as pd

#####處理finmind期貨資料，合併日夜盤，最後資料會有以下欄位['date', 'open_night', 'high_night', 'low_night', 'close_night',
#####'volume_night', 'contract_date', 'open', 'high', 'low', 'close', 'volume', 'all_oi', 'trade_price', 'diff']
#####其中diff表示(近月-遠月)的價差，若是做多可以用加回的方式調整
def new_contract_diff(df_origin, df_finmind):
    df_diff = df_origin[['date', 'contract_date']].copy()
    df_diff['next'] = pd.to_datetime(df_diff['contract_date'], format='%Y%m').dt.to_period('M') + 1
    df_diff['next'] = df_diff['next'].dt.strftime('%Y%m')
    # df_diff['contract_date'] = df_diff['contract_date'] + '/' + df_diff['next']
    
    # df_diff = pd.merge(df_finmind[df_finmind['trading_session'] == 'position'], df_diff, how='inner', on=['date', 'contract_date'])
    # df_origin['diff'] = df_diff['open'] * -1
    df_diff = pd.merge(df_finmind[df_finmind['trading_session'] == 'position'][['date', 'contract_date', 'open']], 
                       df_diff, how='inner', on=['date', 'contract_date'])
    
    df_diff = pd.merge(df_finmind[df_finmind['trading_session'] == 'position'][['date', 'contract_date', 'open']], 
                       df_diff, how='inner', left_on=['date', 'contract_date'], right_on=['date', 'next'], suffixes=('_next', ''))
    df_origin['diff'] = df_diff['open'] - df_diff['open_next']
    return df_origin

def process_taiwan_futures_daily(df):
    """
    處理原始的 FinMind 期貨資料，合併日夜盤並計算價差。
    此函數將獨立於下載邏輯，專注於資料處理。
    """
    df = df.drop(['futures_id', 'spread', 'spread_per', 'settlement_price'], errors='ignore', axis=1)
    df.rename(columns={'max': 'high', 'min': 'low', 'open_interest': 'all_oi'}, inplace=True)
    df_finmind = df.copy()
    df = df[~(df['contract_date'].str.contains("/"))]
    
    night = df[df['trading_session'] == 'after_market'].copy()
    daily = df[df['trading_session'] == 'position'].copy()
    
    night.drop(['trading_session'], axis=1, inplace=True)
    daily.drop(['trading_session'], axis=1, inplace=True)
    
    night['contract_date'] = night['contract_date'].astype(str)
    daily['contract_date'] = daily['contract_date'].astype(str)
    
    #取得每個日期的近月合約，再用merge的方式把原始資料篩選，留下近月合約資料
    maturity = daily.groupby('date')['contract_date'].first().reset_index()
    
    daily_near = pd.merge(daily, maturity, how='inner', on=['date', 'contract_date'])
    night_near = pd.merge(night, maturity, how='inner', on=['date', 'contract_date'])
    
    #TODO: 未平倉資料視情況看要不要做，目前保留原始近月未平倉量
    
    all_day = pd.merge(night_near, daily_near, how='outer', on='date', suffixes=('_night', ''))
    
    cols_to_drop = ['contract_date_night', 'all_oi_night']
    all_day.drop(columns=cols_to_drop, errors='ignore', inplace=True)
    
    # all_day.rename(columns={'open_interest_day': 'near_oi'}, inplace=True)
    
    # all_day['contract_date_night'] = all_day['contract_date_night'].ffill()
    # all_day.rename(columns={'contract_date_night': 'contract_date'}, inplace=True)
    
    # 處理夜盤資料缺失
    nulls = all_day['volume_night'].isnull()
    all_day.loc[nulls, 'volume_night'] = 0
    all_day.loc[nulls, ['open_night', 'high_night', 'low_night', 'close_night']] = all_day['close'].shift().loc[nulls]

    # 加上交易點價格（次日夜盤開盤價）
    all_day['trade_price'] = all_day['open_night'].shift(-1)
    all_day['trade_price'] = all_day['trade_price'].fillna(all_day['close'])
    
    # 計算轉約價差
    # all_day = get_contract_diff(all_day, original=night)
    all_day = new_contract_diff(all_day, df_finmind)
    all_day['date'] = pd.to_datetime(all_day['date'])
    
    return all_day

##三大法人期貨日交易，分成內外資合計
##['date', 'long_deal_volume_foreign', 'long_deal_amount_foreign',
###'short_deal_volume_foreign', 'short_deal_amount_foreign',
###'long_open_interest_balance_volume_foreign',
###'long_open_interest_balance_amount_foreign',
###'short_open_interest_balance_volume_foreign',
###'short_open_interest_balance_amount_foreign', 'long_deal_volume_local',
###'long_deal_amount_local', 'short_deal_volume_local',
###'short_deal_amount_local', 'long_open_interest_balance_volume_local',
###'long_open_interest_balance_amount_local',
###'short_open_interest_balance_volume_local',
###'short_open_interest_balance_amount_local']
def process_taiwan_futures_institutional_investors(df):
    df = df.drop(['futures_id', 'name'], errors='ignore', axis=1)
    
    local = df[df['institutional_investors'] != '外資']
    foreign = df[df['institutional_investors'] == '外資']

    local_sum = local.groupby(by=['date']).sum()
    foreign_sum = foreign.set_index(['date'])
    
    local_sum = local_sum.drop(['institutional_investors'], axis=1)
    foreign_sum = foreign_sum.drop(['institutional_investors'], axis=1)

    all_df = pd.merge(foreign_sum, local_sum, how='inner', on='date', suffixes=('_foreign', '_local'))
    all_df = all_df.reset_index()
    all_df['date'] = pd.to_datetime(all_df['date'])
    
    return all_df

def process_taiwan_stock_daily(df):
    df.rename(columns={'max': 'high', 'min': 'low'}, inplace=True)
    df.drop(['spread'], axis=1, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    return df

def process_us_stock_price(df):
    df['date'] = pd.to_datetime(df['date'])
    return df

def process_taiwan_stock_dividend_result(df):
    df['date'] = pd.to_datetime(df['date'])
    return df

# def get_open(row, original):
#     date = row['date']
#     miss_cont = row['contract_next']
#     next_open = original[(original['date'] == date) & (original['contract_date'] == miss_cont)]['open']
#     row['open_far'] = next_open.values[0]
#     return row

# def get_contract_diff(whole_day_data, original):
#     contract = whole_day_data[['date', 'contract_date_day', 'open_night']].copy()
#     contract.columns = ['date', 'contract', 'open_near']
#     contract['contract_next'] = contract['contract'].shift(-1)
#     contract = contract[contract['contract'] != contract['contract_next']].dropna()
#     contract = contract.apply(get_open, axis=1, original=original)
#     contract['diff'] = contract['open_near'] - contract['open_far']
    
#     whole_day_data = pd.merge(whole_day_data, contract[['date', 'diff']], how='outer', on='date')
#     whole_day_data['diff'] = whole_day_data['diff'].fillna(0)
#     return whole_day_data

