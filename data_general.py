
import os
from FinMind.data import DataLoader
import pandas as pd
import os
from pathlib import Path
from datetime import datetime as dt
import preprocess_data

##資料儲存時，日期為column、資料處理(preprocess)時也是，但最後return時一律當作index
##parquet內日期格式為datetime，但是放在column中
##取出時先set_index方便提取資料，當要合併新資料時才放回column，照這樣儲存，返回資料時才變index
##所有的日期變數都以字串形式為預設，需要比較時再轉換(df.loc可以接受字串)

DATASETS = dir(DataLoader())
# 使用環境變數或預設路徑
DATASET_BASE_PATH = Path(os.getenv('DATASET_PATH', './dataset'))
API_TOKEN = os.getenv('FINMIND_API_TOKEN', '')

def get_finmind_data(dataset_name, start_date, end_date=None, **kwargs):
    """
    根據日期範圍取得台灣期貨日夜盤資料，優先讀取本地檔案，若無則從 FinMind 下載並處理。

    Args:
        start_date (str): 資料起始日期 (YYYY-MM-DD)。
        end_date (str, optional): 資料結束日期 (YYYY-MM-DD)。預設為今天。
        store_file (str, optional): 本地儲存檔名。預設為 'tw_futures_data.parquet'。

    Returns:
        pd.DataFrame: 合併本地與新下載、處理後的期貨資料。
    """
    if dataset_name not in DATASETS:
        raise ValueError("Dataset name not found. Please check again.")
    
    if end_date is None:
        end_date = dt.now().strftime('%Y-%m-%d')
    
    dataset_path = DATASET_BASE_PATH / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)
    #股票的要分別以股票代碼為檔名儲存[以年為單位儲存的程式單獨寫(資料量太大，無法一次儲存)]
    if 'stock_id' in kwargs.keys():
        file_path = dataset_path / f'{kwargs['stock_id']}.parquet'
    else:
        file_path = dataset_path / f'{dataset_name}.parquet'

    local_data = pd.DataFrame()
    local_data_exists = False
    download_start_date = None
    download_end_date = None
    
    # 1. 檢查本地資料是否存在並讀取
    if file_path.exists():
        try:
            local_data = pd.read_parquet(file_path)
            local_data['date'] = pd.to_datetime(local_data['date'])
            local_data.set_index('date', inplace=True)
            local_data_exists = True
            print(f'讀取到本地資料，範圍從 {local_data.index.min()} 到 {local_data.index.max()}，共有{len(local_data)}筆資料。')
            
            # 檢查本地資料範圍是否足夠
            if (local_data.index.min() <= pd.to_datetime(start_date)) and \
               (local_data.index.max() >= pd.to_datetime(end_date)):
                print(f"本地期貨資料已存在。返回 {start_date} 到 {end_date} 的資料。")
                return local_data.loc[start_date:end_date]

            # 找到需要下載的日期區間
            earliest_local_date = local_data.index.min()
            latest_local_date = local_data.index.max()
            if earliest_local_date > pd.to_datetime(start_date):
                download_start_date = start_date
            else:
                download_start_date = latest_local_date.strftime('%Y-%m-%d')
            
            if latest_local_date < pd.to_datetime(end_date):
                download_end_date = end_date
            else:
                download_end_date = earliest_local_date.strftime('%Y-%m-%d')
            
            print(f"本地資料已存在，但需要更新。正在下載從 {download_start_date} 到 {download_end_date} 的新資料。")
            
        except Exception as e:
            print(f"讀取本地檔案 {file_path} 時出錯: {e}。將下載所有資料。")
            local_data_exists = False
            download_start_date = start_date
            download_end_date = end_date
    else:
        print("未找到本地檔案，將從頭下載資料。")
        download_start_date = start_date
        download_end_date = end_date
        
    # 2. 從 FinMind 下載資料
    if download_start_date < end_date:
        print(f"正在從 FinMind 下載 {download_start_date} 到 {download_end_date} 的期貨資料...")
    else:
        print("日期區間錯誤，請重新檢視。")
        return pd.DataFrame()
    
    fm = DataLoader()
    if API_TOKEN:
        fm.login_by_token(api_token=API_TOKEN)
        print('成功登入 FinMind')
    
    try:
        dataset_method = getattr(fm, dataset_name)
        if callable(dataset_method):
            #TODO:參數可能會出問題
            df = dataset_method(
                start_date=download_start_date,
                end_date=download_end_date,
                **kwargs
            )
            print(f"{fm.api_usage} / {fm.api_usage_limit}")
            if df.empty:
                print("沒有從 FinMind 下載到新資料，返回本地端資料。")
                return local_data.loc[start_date:end_date]
            else:
                print(f"下載了 {df['date'].min()} 到 {df['date'].max()} 的資料，共有 {len(df)} 筆資料。")

            
    except Exception as e:
        print(f"從 FinMind 下載時出錯: {e}")
        return pd.DataFrame()

    #### 3. 處理新下載的資料
    #TODO:
    try:
        process_method = getattr(preprocess_data, f'process_{dataset_name}')
        if callable(process_method):
            processed_df = process_method(df)
        else:
            raise ModuleNotFoundError("前處理函數尚未建立，無法使用。")
    except Exception as e:
        print(f'處理新下載的期貨資料時出錯: {e}')
        return pd.DataFrame()

    # 4. 合併本地與新下載的資料，並儲存
    if local_data_exists and not local_data.empty:
        local_data.reset_index(inplace=True)
        
        # print(local_data.columns.to_list(), processed_df.columns.to_list())
        assert local_data.columns.to_list() == processed_df.columns.to_list(), "新舊資料欄位不一樣，無法合併。"
        
        final_df = pd.concat([local_data, processed_df], ignore_index=True)
        # 移除重複的日期，保留最新的資料
        final_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
        final_df.sort_values(by='date', inplace=True)
        print(f"已合併本地與下載的資料。總共筆數: {len(final_df)}")
    else:
        final_df = processed_df
    
    final_df.to_parquet(file_path, index=False, compression='snappy')
    
    # 5. 返回指定日期範圍的資料
    final_df['date'] = pd.to_datetime(final_df['date'])
    final_df.set_index('date', inplace=True)
    
    print(f"最終資料範圍: {final_df.index.min().strftime('%Y-%m-%d')} 到 {final_df.index.max().strftime('%Y-%m-%d')}")
    return final_df.loc[start_date:end_date]

if __name__ == '__main__':
    # 範例用法
    data = get_finmind_data('taiwan_stock_dividend_result', '2003-07-07', '2024-12-31', stock_id='2330')#futures_id='TX')
    if not data.empty:
        print("\n--- 成功取得並處理期貨資料 ---")
        print(data.info())
        # print(data.head())
        print(data.tail())

    #範例2
    # data = get_finmind_data('taiwan_futures_institutional_investors', '2018-06-05', '2024-12-31', data_id='TX')
    # if not data.empty:
    #     print("\n--- 成功取得並處理期貨資料 ---")
    #     print(data.info())
    #     # print(data.head())
    #     print(data.tail())

