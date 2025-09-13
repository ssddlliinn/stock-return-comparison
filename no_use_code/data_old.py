from pathlib import Path
import os
import pandas as pd
from datetime import datetime, timedelta
from FinMind.data import DataLoader


DATASET_BASE_PATH = Path(r'.\dataset')
API_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNS0wNy0yNiAxNToxOToxNyIsInVzZXJfaWQiOiJBbmR5MzA2IiwiaXAiOiIxNzUuMTgxLjE3OC4xNSJ9.By2XXOdZXgc_ng21sG9Nf9rjjBuu19WxJEHsIc4tNL4'
DATASETS = dir(DataLoader())
TW_TRADING_DATES = pd.to_datetime(pd.read_parquet(r'.\dataset\tw_trading_date.parquet')['date'])
US_TRADING_DATES = pd.to_datetime(pd.read_parquet(r'.\dataset\us_trading_date.parquet')['date'])

def get_finmind_data(dataset_name, start_date = '1996-01-01', end_date = None, stock_id = '2330', **kwargs):
    """
    檢查本地是否存在指定資料集，沒有則從 FinMind 下載並儲存。

    Args:
        dataset_name (str): FinMind 資料集名稱，例如 'TaiwanStockDailyPrice'。
        stock_id: 只有給為all時才下載所有資料(付費版)，以年份儲存。
        start_date (str, optional): 資料起始日期 (YYYY-MM-DD)。
                                    如果本地有資料，會從本地資料的最新日期+1天開始下載。
                                    如果本地無資料，則從此日期開始下載。
                                    預設為 None，會根據本地資料或 FinMind 的預設最早日期來決定。
        end_date (str, optional): 資料結束日期 (YYYY-MM-DD)。預設為今天。
        目前儲存到交易日期為1996-01-01~2024-12-31
        ******該函式以獲取完整每日資料為目標設計，所以非每日資料的資料集會一直重複撈取資料(Ex.除權息)

    Returns:
        pd.DataFrame: 合併後的資料集 DataFrame。
    """
    #資料集名稱確認
    if dataset_name not in DATASETS:
        raise ValueError("Dataset name not found. Please check again.")
    
    #取當日為最後日
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 1. 定義本地資料夾路徑，建立一個新的
    if stock_id != 'all':
        dataset_path = DATASET_BASE_PATH / dataset_name / 'stock'
    else:
        dataset_path = DATASET_BASE_PATH / dataset_name
        
    if not dataset_path.exists():
        os.makedirs(dataset_path, exist_ok=True)
        
    # 如果stock_id = 'all'，才讀所有資料夾
    if stock_id != 'all':
        local_files = [f for f in os.listdir(dataset_path) if f.startswith(stock_id) and f.endswith('.parquet')]
    else:
        local_files = [f for f in os.listdir(dataset_path) if f.endswith('.parquet')]
        
    # 2. 取出所有要下載日期的list[目前trading date 以台積電1996-2024的交易日為主]
    if dataset_name == 'us_stock_price':
        download_date_set = set(US_TRADING_DATES[US_TRADING_DATES.between(start_date, end_date)])
    else:
        download_date_set = set(TW_TRADING_DATES[TW_TRADING_DATES.between(start_date, end_date)])
        
    
    # 3. 檢查本地資料
    local_data_exists = False
    download_required = False
    local_dfs = []

    if local_files:
        #讀取資料庫，所有資料版本都從1996開始
        #****當股票代碼為all時，跑遍所有年份資料庫，挑取需要的年份資料做讀取並使用
        print(f"Checking existing local data for {dataset_name} {stock_id}...")
        for f in sorted(local_files): # 按名稱排序，通常按年份排
            if (f[:4] not in {str(i.year) for i in download_date_set}) and (stock_id == 'all'):
                continue
            else:
                file_path = dataset_path / f
            
            try:
                # 這裡可以只讀取部分欄位來檢查日期，減少記憶體消耗
                df_temp = pd.read_parquet(file_path, columns=['date'])
                if not df_temp.empty:
                    current_date = set(pd.to_datetime(df_temp['date']))
                    download_date_set = download_date_set - current_date
                      
                    # 載入完整的本地資料，以便後續合併（如果數據量太大，可能需要分塊處理）
                    local_dfs.append(pd.read_parquet(file_path))
                
            except Exception as e:
                print(f"Error reading local file {f}: {e}. Skipping.")
                continue # 如果檔案損壞，跳過這個檔案
            
        if local_dfs:
            local_data_exists = True
            if len(local_dfs) > 1: #all條件，讀取多個資料庫
                all_local_data = pd.concat(local_dfs, ignore_index=True)
            else:
                all_local_data = local_dfs[0] #指定股票，讀取單一資料庫
                # assert all_local_data['stock_id'].unique()
            
        else: # 沒有成功讀取任何本地檔案
            local_data_exists = False
            
    else: # 本地完全沒有檔案
        local_data_exists = False
        
    #date_set 裡面有日期表示有需要下載資料
    if download_date_set:
        download_required = True
        download_date_set = sorted(download_date_set) #set變成list
        print(f"Download required for {len(download_date_set)} days for {dataset_name}.")
    else:
        print(f"Local data for {dataset_name} is up to date (or user-specified range is fully covered). No download needed.")
    
    
    finmind_df = pd.DataFrame()
    download_dfs = []
    if download_required:
        # 3. 從 FinMind 下載資料
        fm = DataLoader()
        if API_TOKEN:
            fm.login_by_token(api_token=API_TOKEN)
            print('成功登入FinMind')
            
        try:
            # 根據資料集類型設定參數，例如日股價不需要股票代碼
            # 如果是台灣期貨逐筆資料，dataset 可能不同，且沒有 stock_id 參數
            # 這裡假設是股票日價，有 stock_id 參數，但我們下載所有股票
            # FinMind 下載所有股票時 stock_id 留空
            dataset_method = getattr(fm, dataset_name)
            if callable(dataset_method):
                if stock_id != 'all':
                    #以download_date_set的起始日跟結束日作為FinMind資料的輸入值
                    download_start_date_str = download_date_set[0].strftime('%Y-%m-%d') #已經sort過了，所以可以直接取第一跟最後
                    download_end_date_str = download_date_set[-1].strftime('%Y-%m-%d')
                    if stock_id is None:
                        finmind_df = dataset_method(
                            start_date = download_start_date_str,
                            end_date = download_end_date_str
                        )
                    else:
                        finmind_df = dataset_method(
                            stock_id = stock_id, 
                            start_date = download_start_date_str,
                            end_date = download_end_date_str
                        )
                else: #讀取特定日期所有資料
                    for date in download_date_set:
                        date_str = date.strftime('%Y-%m-%d')

                        finmind_df = dataset_method(
                            start_date=date_str,
                        )
                        
                        download_dfs.append(finmind_df)
                    finmind_df = pd.concat(download_dfs, ignore_index=True)
                    
            else:
                raise ModuleNotFoundError("該資料集不存在finmind方法中，請再查詢。")
            
            
            print(f"{fm.api_usage} / {fm.api_usage_limit}")
            if finmind_df.empty:
                print(f"No new data downloaded from FinMind for {dataset_name} {stock_id} in the specified range.")
                print(f"Maybe this dataset does not have data between {download_start_date_str} and {download_end_date_str}. Or maybe you entered a wrong stock ID.")
            else:
                print(f"Downloaded {len(finmind_df)} rows from FinMind for {dataset_name} {stock_id}.")
                
                ### 可能需要更新任意區間日期資料，所以這部分先不使用，直接用final_df去除重複值
                # 確保日期是 datetime 類型
                # finmind_df['date'] = pd.to_datetime(finmind_df['date'])
                # # FinMind 下載的資料可能包含重複或與本地數據重複的部分
                # # 通常下載的是最新數據，所以我們要處理日期範圍的重疊
                # if local_data_exists and not finmind_df.empty:
                #     # 篩選掉 FinMind 下載數據中，日期早於或等於本地最新日期的數據
                #     finmind_df = finmind_df[finmind_df['date'] > latest_local_date]
                #     if finmind_df.empty:
                #         print("Downloaded data is older than or same as local data. No new data to add.")
                #     else:
                #         print(f"Filtered downloaded data: {len(finmind_df)} new rows after removing overlaps.")
                
        except Exception as e:
            print(f"Error downloading from FinMind: {e}")
            finmind_df = pd.DataFrame() # 下載失敗，返回空 DataFrame

    # 4. 合併本地和新下載的資料，並儲存到本地
    final_df = None
    if local_data_exists and not all_local_data.empty:
        #有本地資料、本地資料df不為空、下載資料不為空
        if not finmind_df.empty:
            all_local_data['date'] = pd.to_datetime(all_local_data['date'])
            finmind_df['date'] = pd.to_datetime(finmind_df['date'])
            
            final_df = pd.concat([all_local_data, finmind_df], ignore_index=True)
            # 去除重複項，以 'date' 和 'stock_id' 作為判斷重複的依據
            final_df = final_df.drop_duplicates(subset=['date', 'stock_id']).sort_values(by=['date', 'stock_id']).reset_index(drop=True)
            print(f"Combined local and downloaded data. Total rows: {len(final_df)}")
        #有本地資料、本地資料df不為空、下載資料空的
        else:
            all_local_data['date'] = pd.to_datetime(all_local_data['date'])
            final_df = all_local_data

    #本地資料df為空、下載資料不為空
    elif not finmind_df.empty:
        finmind_df['date'] = pd.to_datetime(finmind_df['date'])
        final_df = finmind_df
        final_df = final_df.drop_duplicates(subset=['date', 'stock_id']).sort_values(by=['date', 'stock_id']).reset_index(drop=True)
    # 本地資料df為空、下載資料空的
    else:
        print("No data available locally or from download.")
        return pd.DataFrame()
    
    # 5. 將合併後的資料按年份儲存到本地 Parquet 檔案    
    if final_df is not None and not final_df.empty:
        #下載資料不為空、所有股票
        if not finmind_df.empty and (stock_id == 'all'):
            final_df['year'] = final_df['date'].dt.year
            
            for year, group_df in final_df.groupby('year'):
                output_file = dataset_path / f"{str(year)}.parquet"
                
                # 使用 Parquet 儲存
                group_df.drop(columns=['year']).to_parquet(output_file, index=False, compression='snappy')
                print(f"Saved data for year {year} to {output_file}")
                
            # 返回完整的合併資料
            print(f"現階段日期區間為{final_df['date'].min()}到{final_df['date'].max()}，共有", final_df.shape, "筆資料")
            return final_df.drop(columns=['year'])[final_df['date'].between(start_date, end_date)] # 返回時移除輔助的'year'欄位
        
        elif not finmind_df.empty and (stock_id != 'all'):
            output_file = dataset_path / f"{stock_id}.parquet"
            final_df.to_parquet(output_file, index=False, compression='snappy')
            print(f"現階段日期區間{final_df['date'].min()}到{final_df['date'].max()}，共有", final_df.shape, "筆資料")

            return final_df[final_df['date'].between(start_date, end_date)]
        else:
            print("不需下載資料，直接返回本地資料並不做儲存。")
            print(f"現階段日期區間{final_df['date'].min()}到{final_df['date'].max()}，共有", final_df.shape, "筆資料")
            if stock_id != 'all':
                return final_df[final_df['date'].between(start_date, end_date)]
            else:
                return final_df[final_df['stock_id'] == stock_id]
    else:
        return pd.DataFrame()
    

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
    
    monthly_summary = monthly_summary.reset_index()
    monthly_summary.rename(columns={'date': 'month'}, inplace=True)
    # 把日期資料換成以月分資料表示
    monthly_summary['month'] = monthly_summary['month'].dt.to_period('M')
    
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
            div_data['date'] = pd.to_datetime(div_data['date'])
            # 把日期資料換成以月分資料表示
            div_data['month'] = div_data['date'].dt.to_period('M')
            div_data['dividend_yield'] = div_data['stock_and_cache_dividend'] / div_data['before_price'] #計算股價殖利率
            monthly_div = div_data.groupby('month')['dividend_yield'].sum().reset_index()
            

            # 以股票代碼及年月為合併基準，把月報酬跟除權息殖利率合併
            # no index、columns=['stock_id', 'month', 'monthly_return', 'dividend_yield']
            #TODO: expect columns for US stock ['stock_id', 'month', monthly_return_adj']
            final_df = pd.merge(monthly_summary, monthly_div, on=['month'], how='left')
            final_df['stock_id'] = stock_id
            final_df = final_df.loc[:, ['stock_id', 'month', 'monthly_return', 'dividend_yield']]
            final_df = final_df.fillna(0)
            return final_df
    
    final_df = monthly_summary.copy()
    final_df['stock_id'] = stock_id
    final_df['dividend_yield'] = 0
    final_df = final_df.loc[:, ['stock_id', 'month', 'monthly_return', 'dividend_yield']]
    
    final_df = final_df.fillna(0)
    return final_df

    
    
if __name__ == "__main__":
    # 範例 1: 下載台灣股價資料 (假設本地沒有或不完整)
    # print("--- 測試下載台灣股價資料 ---")
    # tw_stock_data = get_finmind_data(
    #     # dataset_name='taiwan_stock_dividend_result',
    #     dataset_name='taiwan_stock_daily',
    #     stock_id='2882',
    #     start_date='2005-01-01',
    #     end_date='2025-06-30'
    # )
    
    # if not tw_stock_data.empty:
    #     print(f"Successfully retrieved Taiwan Stock Data. Last 5 rows:\n{tw_stock_data.head()}\n{tw_stock_data.tail()}")
    #     print(f"Data range: {tw_stock_data['date'].min().strftime('%Y-%m-%d')} to {tw_stock_data['date'].max().strftime('%Y-%m-%d')}")
    #     print(f"Total rows: {len(tw_stock_data)}")
    # else:
    #     print("Failed to retrieve Taiwan Stock Data.")

    # print("\n" + "="*50 + "\n")
    # input("Checked:")

    # 範例 2: 再次執行，應該會檢查本地數據並判斷是否需要更新
    # print("--- 再次測試下載台灣股價資料 (應該會檢查並可能只下載最新部分) ---")
    # tw_stock_data_updated = get_finmind_data(
    #     dataset_name='taiwan_stock_daily',
    #     end_date= '1996-01-31'
    # )
    # if not tw_stock_data_updated.empty:
    #     print(f"Successfully retrieved (and potentially updated) Taiwan Stock Data. Total rows: {len(tw_stock_data_updated)}")
    #     print(f"Data range: {tw_stock_data_updated['date'].min().strftime('%Y-%m-%d')} to {tw_stock_data_updated['date'].max().strftime('%Y-%m-%d')}")
    # else:
    #     print("Failed to retrieve Taiwan Stock Data.")

    # 你也可以測試其他資料集，例如美國股票 (如果 FinMind 提供)
    # us_stock_data = get_finmind_data(
    #     dataset_name='USStockDailyPrice',
    #     start_date='2024-01-01'
    # )
    
    # 範例3：使用summary合併股價與除權息資料
    print("--- 測試合併股價與除權息資料 ---")
    df = summary_monthly_data(
        stock_id='2882',
        start_date='2005-01-01',
        end_date='2025-06-30'
        )
    print(df[df['month'] == '2008-07'])
    # print(df.tail(20))
    
    # if not df.empty:
    #     print(f"Successfully retrieved Summary Date. Last 5 rows:\n{df.head()}\n{df.tail()}")
    #     print(df.dtypes)
    # else:
    #     print("Failed to retrieve Summary Data.")
    
    
    #個別使用
    pass    