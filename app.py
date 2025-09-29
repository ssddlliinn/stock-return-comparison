from flask import Flask, render_template, request, jsonify
from summary import summary_monthly_data
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import json
import numpy as np
import re
import os

app = Flask(__name__)

def calculate_portfolio_returns(portfolio, portfolio_dfs, start_month):
    """
    根據使用者輸入計算投資組合的月報酬率。
    """
    total_df = None
    
    # 變更點：現在 portfolio_data 是一個字典，我們用 .items() 來遍歷
    for stock_id, weight in portfolio.items():
        print(stock_id, weight)
        weight = int(weight) / 100
        
        df = portfolio_dfs[stock_id]
        if not df.empty:
            assert df['stock_id'].iloc[0] == stock_id, f"股票代碼{stock_id}與月報酬資料代碼{df['stock_id'].iloc[0]}對不上"
            # df period 可以跟string格式的年月直接比較
            df = df[df.index >= start_month].copy()
            if not df.empty:
                df = df.drop('stock_id', axis=1)
                if total_df is None:
                    total_df = df * weight
                else:
                    total_df += df * weight
    
    return total_df #'month':['monthly_return', 'dividend_yield']

def calculate_metrics(df, reinvest_dividends, initial_capital, annuity_mode=False):
    """
    計算並回傳量化指標的函數。
    """
    df = df.copy()
    return_dict = {}
    
    if not annuity_mode:
        # 計算年化報酬率 (Annualized Return)
        # 取累積報酬率最後一個值就是整個區間整體報酬率
        total_periods = len(df)
        total_cumulative_return = df['cumulative_return'].iloc[-1]
        annualized_return = ((1 + total_cumulative_return)**(12/total_periods) - 1).round(6)
        
        # 計算波動率 (Annualized Volatility)
        annualized_volatility = (df['effective_return'].std() * np.sqrt(12)).round(6)
        
        # 計算最大跌幅(MDD)
        MDD = (df['cumulative_value'] / df['cumulative_value'].cummax() - 1).min()
        
        return_dict['annualized_return'] = annualized_return
        return_dict['annualized_volatility'] = annualized_volatility
        return_dict['MDD'] = MDD

    # 取得期末金額
    final_return = str(df['cumulative_value'].iloc[-1])

    # 計算總領股息
    # specific:表示把殖利率比照第一個投資組合做提領
    # true:表示全部再投資，不會有現金被領出
    # false:按照其原本的配發條件做配發
    df['cumulative_value'] = df['cumulative_value'].shift(1).fillna(initial_capital).copy()
    if 'dividend_value' in df.columns:
        total_dividends = int(df['dividend_value'].sum())
    elif 'specific_dividend_yield' in df.columns:
        total_dividends = (df['specific_dividend_yield'] * df['cumulative_value']).round(0).sum()
    elif reinvest_dividends == 'true':
        total_dividends = 0
    else:
        total_dividends = (df['dividend_yield'] * df['cumulative_value']).round(0).sum()
    
    # 這裡可以加入更多指標，例如夏普比率、最大回撤等
    
    
    return_dict['final_return'] = final_return
    return_dict['total_dividends'] = total_dividends
        
    return return_dict


@app.route('/')
def index():
    """
    首頁路由，渲染 HTML 模板。
    """
    return render_template('index.html')


@app.route('/calculator')
def new_page():
    return render_template('calculator.html')

@app.route('/pmt-investment')
def pmt_investment():
    return render_template('pmt-investment.html')


@app.route('/get_chart_data', methods=['POST'])
def get_chart_data():
    """
    API 路由，處理來自前端的 POST 請求，並回傳圖表資料。
    """
    #取得所有從網頁回傳的資料
    start_date_str = request.form.get('start_date')
    end_date_str = request.form.get('end_date')
    rebalance = request.form.get('rebalance')
    initial_capital = float(request.form.get('initial_capital'))
    reinvest_dividends = request.form.get('reinvest_dividends')
    display_mode = request.form.get('display_mode')
    portfolios_data = [
        request.form.get('portfolio1_data'),
        request.form.get('portfolio2_data'),
        request.form.get('portfolio3_data')
    ]

    valid_portfolios = []
    dataframes = []
    metrics = []
    earliest_month = []
    specify_div = None
    start_month = None

    for p_data in portfolios_data: #list of dict-like strings, content is portfolios like stock_a: weight
        if p_data:
            portfolio = json.loads(p_data)
            stocks_in_portfolio = list(portfolio.keys())
            
            portfolio_dfs = {}
            for stock_id in stocks_in_portfolio:
                if stock_id == 'PENSION_FUND':
                    pension_path = os.getenv('PENSION_DATA_PATH', './dataset/pension.parquet')
                    try:
                        df = pd.read_parquet(pension_path)
                        df = df[df.index.between(start_date_str, end_date_str)]
                    except FileNotFoundError:
                        print(f"Warning: Pension data file not found at {pension_path}")
                        df = pd.DataFrame()
                else:
                    df = summary_monthly_data(
                        stock_id=stock_id,
                        market='us' if re.match(r'^[A-Z\^\.]+$', stock_id) else 'tw',
                        start_date=start_date_str,
                        end_date=end_date_str
                    )
                
                # 如果有空的資料，就直接回傳找不到資料
                if not df.empty:
                    earliest_month.append(df.index.min())
                    portfolio_dfs[stock_id] = df
                    print(f'股票:{stock_id}的最早資料年月為{df.index.min().strftime("%Y-%m")}')
                    assert df.index.max().strftime("%Y-%m") != end_date_str[:8], f"{stock_id}資料最後與結束日期對不上"
                else:
                    return jsonify({'error': f'無法找到股票代碼{stock_id}或資料。請重新輸入。'}), 400
            if portfolio_dfs:
                valid_portfolios.append(portfolio_dfs)  
                ### list of portfolios, where every portfolio is a dict having keys as stock_id and 
                ### value of return data df
    
    if not valid_portfolios:
        return jsonify({'error': f'無法找到股票代碼{stock_id}或資料。請重新輸入。'}), 400            
    start_month = max(earliest_month)
    
    for i, portfolio in enumerate(portfolios_data): #list of dict-like strings, content is portfolios like stock_a: weight
        portfolio = json.loads(portfolio)
        
        # 產出一個'month' :['monthly_return', 'dividend_yield', ('specific_dividend_yield'), 
        # 'effective_return', 'cumulative_return', 'cumulative_value']
        if rebalance == 'true': ###### 先用權重乘以月報酬及股利，再依條件得到有效月報酬，最後算出總金額
            df = calculate_portfolio_returns(portfolio, valid_portfolios[i], start_month)
            # 根據下拉式選單選項計算累計報酬率
            if reinvest_dividends == 'specific':
                if i == 0:
                    specify_div = df['dividend_yield']
                    df['effective_return'] = df['monthly_return']
                else:
                    df['specific_dividend_yield'] = specify_div
                    df['effective_return'] = df['monthly_return'] + df['dividend_yield'] - df['specific_dividend_yield']
            elif reinvest_dividends == 'true':
                df['effective_return'] = df['monthly_return'] + df['dividend_yield']
            else:
                df['effective_return'] = df['monthly_return']
                
            df['cumulative_return'] = (1 + df['effective_return']).cumprod() - 1

            # 計算累積價值
            df['cumulative_value'] = initial_capital * (1 + df['cumulative_return'])
        
        # 產出一個['month', 'monthly_return', 'dividend_yield', ('dividend_value'), 
        # 'effective_return', 'cumulative_return', 'cumulative_value']
        else: ###### 先個別算出總金額，再用權重得到最後總金額，再回推有效報酬
            
            # 從有效投資組合列表中取得對應的資料
            stock_data_dict = valid_portfolios[i]
            
            # 初始化累積報酬和現金股利
            combined_cumulative_value = None
            
            for stock_id, weight in portfolio.items():
                print(stock_id, weight)
                weight = weight / 100
                df_stock = stock_data_dict[stock_id]
                df_stock = df_stock[df_stock.index >= start_month].copy()
                
                # 計算每檔股票的累積價值
                if reinvest_dividends == 'true':
                    # 股息再投入
                    df_stock['cumulative_return'] = (1 + df_stock['monthly_return'] + df_stock['dividend_yield']).cumprod() - 1
                    df_stock['cumulative_value'] = initial_capital * weight * (1 + df_stock['cumulative_return'])
                    df_stock['dividend_value'] = 0
                else:
                    # 股息不再投入
                    df_stock['cumulative_return'] = (1 + df_stock['monthly_return']).cumprod() - 1
                    df_stock['cumulative_value'] = initial_capital * weight * (1 + df_stock['cumulative_return'])
                    df_stock['dividend_value'] = df_stock['cumulative_value'].shift(1).fillna(initial_capital * weight) * df_stock['dividend_yield']
                
                # df_stock['cumulative_value'] = initial_capital * weight * (1 + df_stock['cumulative_return'])
                
                df_stock = df_stock.drop('stock_id', axis=1)
                # print(df_stock)
                
                if combined_cumulative_value is None:
                    combined_cumulative_value = df_stock.copy()
                else:
                    print(df_stock.head(5))
                    print(df_stock.tail(5))
                    # print(len(df_stock.index))
                    combined_cumulative_value += df_stock
            
            # 回推投資組合的報酬率
            df = combined_cumulative_value.copy()
            df['cumulative_return'] = (df['cumulative_value'] / initial_capital) - 1
            
            # 為了計算指標，我們需要 `effective_return`，可以從 `cumulative_return` 回推
            # 但這是一個簡化方式，可能與實際的 monthly_return 有微小誤差
            df['effective_return'] = df['cumulative_value'].pct_change().fillna(0)
            df['dividend_value'] = df['dividend_value'].round(0)
        
        
        df.index = df.index.strftime('%Y-%m')
        df['cumulative_return'] = df['cumulative_return'].round(6)
        df['effective_return'] = df['effective_return'].round(6)
        df['cumulative_value'] = df['cumulative_value'].round(0)
        # print(df)
        
        dataframes.append((f"投資組合{i+1}", df))
        # 用df資料計算顯示在表格的資訊
        metrics.append(calculate_metrics(df, reinvest_dividends, initial_capital))

    if not all(dataframes):
        return jsonify({'error': '無法找到股票代碼或資料。請重新輸入。'}), 400
    
    # 使用 Plotly.express 創建圖表，但這裡使用更靈活的 go.Figure
    fig = go.Figure()
    
    # 定義三種線條顏色
    line_colors = ["#2c4cff", '#d62728', "#2ed728"]
    
    for i, (stock_id, df) in enumerate(dataframes):
        y_data = df['cumulative_value'].to_list() if display_mode == 'value' else df['cumulative_return'].to_list()
        y_title = '累積金額' if display_mode == 'value' else '累計報酬率'
        
        fig.add_trace(go.Scatter(
            x=df.index.to_list(),
            y=y_data,
            mode='lines',
            line=dict(color=line_colors[i], width=2),
            name=f'{stock_id} {y_title}'
        ))

    # 圖表美化設定
    fig.update_layout(
        title={
            # 'text': f'{stocks_input[0]} vs {stocks_input[1]} vs {stocks_input[2]}股票累計報酬率比較',
            'text': f'投資組合1 vs 投資組合2 vs 投資組合3 股票累計報酬率比較',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#333')
        },
        xaxis_title='月份',
        yaxis_title='累計報酬率',
        hovermode='x unified', # 鼠標懸停時顯示所有線條的資料
        template='plotly', # 使用黑色背景模板
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#333"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#888888', weight=10),
            bgcolor='#ffffff',
            bordercolor='#ccc',
            borderwidth=0.5
        ),
        margin=dict(l=50, r=50, t=100, b=50),
        xaxis=dict(
            showgrid=False,
            # gridwidth=1,
            # gridcolor='#555555',
            tickformat="%Y-%m" # X軸時間格式
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.3,
            tickformat=".2%" if display_mode == 'return' else "$.0f", # Y軸刻度顯示為百分比
        ),
        hoverlabel=dict(
            # font=dict(color="#030000"),
            # bgcolor='#ffffff',
            font_size=16,
            font_family="Rockwell"
        )
    )

    # 將 Plotly 圖表物件轉換為 JSON 字串，以便傳遞給前端
    # 使用 plotly.io.to_json 來確保輸出標準 JSON 格式
    graph_json = pio.to_json(fig, engine='json')

    return jsonify({
        'graph_json': graph_json,
        'metrics': {
            'stocks_input': ['投資組合1', '投資組合2', '投資組合3'],
            'metrics_data': metrics
        }
    })

@app.route('/get_annuity_chart_data', methods=['POST'])
def get_annuity_chart_data():
    """
    API 路由，處理來自前端的 POST 請求，並回傳圖表資料。
    """
    #取得所有從網頁回傳的資料
    start_date_str = request.form.get('start_date')
    end_date_str = request.form.get('end_date')
    rebalance = request.form.get('rebalance')
    annuity_capital = float(request.form.get('annuity_capital'))
    reinvest_dividends = request.form.get('reinvest_dividends')
    portfolios_data = [
        request.form.get('portfolio1_data'),
        request.form.get('portfolio2_data'),
        request.form.get('portfolio3_data')
    ]

    valid_portfolios = []
    dataframes = []
    metrics = []
    earliest_month = []
    specify_div = None
    start_month = None

    for p_data in portfolios_data: #list of dict-like strings, content is portfolios like stock_a: weight
        if p_data:
            portfolio = json.loads(p_data)
            stocks_in_portfolio = list(portfolio.keys())
            
            portfolio_dfs = {}
            for stock_id in stocks_in_portfolio:
                if stock_id == 'PENSION_FUND':
                    pension_path = os.getenv('PENSION_DATA_PATH', './dataset/pension.parquet')
                    try:
                        df = pd.read_parquet(pension_path)
                        df = df[df.index.between(start_date_str, end_date_str)]
                    except FileNotFoundError:
                        print(f"Warning: Pension data file not found at {pension_path}")
                        df = pd.DataFrame()
                else:
                    df = summary_monthly_data(
                        stock_id=stock_id,
                        market='us' if re.match(r'^[A-Z\^\.]+$', stock_id) else 'tw',
                        start_date=start_date_str,
                        end_date=end_date_str
                    )
                
                # 如果有空的資料，就直接回傳找不到資料
                if not df.empty:
                    earliest_month.append(df.index.min())
                    portfolio_dfs[stock_id] = df
                    print(f'股票:{stock_id}的最早資料年月為{df.index.min().strftime("%Y-%m")}')
                    assert df.index.max().strftime("%Y-%m") != end_date_str[:8], f"{stock_id}資料最後與結束日期對不上"
                else:
                    return jsonify({'error': f'無法找到股票代碼{stock_id}或資料。請重新輸入。'}), 400
            if portfolio_dfs:
                valid_portfolios.append(portfolio_dfs)  
                ### list of portfolios, where every portfolio is a dict having keys as stock_id and 
                ### value of return data df
    
    if not valid_portfolios:
        return jsonify({'error': f'無法找到股票代碼{stock_id}或資料。請重新輸入。'}), 400            
    start_month = max(earliest_month)
    
    for i, portfolio in enumerate(portfolios_data): #list of dict-like strings, content is portfolios like stock_a: weight
        portfolio = json.loads(portfolio)
        
        # 產出一個'month':['monthly_return', 'dividend_yield', ('specific_dividend_yield'), 
        # 'effective_return', 'cumulative_value'] ##年金投資不會有累積報酬率
        if rebalance == 'true': ###### 先用權重乘以月報酬及股利，再依條件得到有效月報酬，最後算出總金額
            df = calculate_portfolio_returns(portfolio, valid_portfolios[i], start_month)
            # 根據下拉式選單選項計算有效報酬率
            if reinvest_dividends == 'specific':
                if i == 0:
                    specify_div = df['dividend_yield']
                    df['effective_return'] = df['monthly_return']
                else:
                    df['specific_dividend_yield'] = specify_div
                    df['effective_return'] = df['monthly_return'] + df['dividend_yield'] - df['specific_dividend_yield']
            elif reinvest_dividends == 'true':
                df['effective_return'] = df['monthly_return'] + df['dividend_yield']
            else:
                df['effective_return'] = df['monthly_return']

            # 計算累積價值
            # 懶得改程式，先reset後再重新set
            df.reset_index(inplace=True)
            df['annuity'] = annuity_capital
            df['cumulative_value'] = 0
            df.loc[0, 'cumulative_value'] = np.int64(round(df.loc[0, 'annuity'] * (1 + df.loc[0, 'effective_return']), 2))
            
            for j in range(1, len(df)):
                previous_cumulative = df.loc[j - 1, 'cumulative_value']
                current_investment = df.loc[j, 'annuity']
                current_return_rate = df.loc[j, 'effective_return']

                df.loc[j, 'cumulative_value'] = round((previous_cumulative + current_investment) * (1 + current_return_rate), 0)

            df.set_index('month', inplace=True)
            
        # 產出一個'month'['monthly_return', 'dividend_yield', ('dividend_value'), 
        # 'effective_return', 'cumulative_value']
        else: ###### 先個別算出總金額，再用權重得到最後總金額，再回推有效報酬
            
            # 從有效投資組合列表中取得對應的資料
            stock_data_dict = valid_portfolios[i]
            
            # 初始化累積報酬和現金股利
            combined_cumulative_value = None
            
            for stock_id, weight in portfolio.items():
                print(stock_id, weight)
                weight = weight / 100
                df_stock = stock_data_dict[stock_id]
                df_stock = df_stock[df_stock.index >= start_month].copy()
                
                # 計算每檔股票的累積價值
                df_stock.reset_index(inplace=True)
                if reinvest_dividends == 'true':
                    # 股息再投入
                    df_stock['effective_return'] = df_stock['monthly_return'] + df_stock['dividend_yield']
                    # 計算累積價值
                    df_stock['annuity'] = annuity_capital * weight
                    df_stock['cumulative_value'] = 0
                    df_stock.loc[0, 'cumulative_value'] = np.int64(df_stock.loc[0, 'annuity'] * (1 + df_stock.loc[0, 'effective_return']))
                    
                    for j in range(1, len(df_stock)):
                        previous_cumulative = df_stock.loc[j - 1, 'cumulative_value']
                        current_investment = df_stock.loc[j, 'annuity']
                        current_return_rate = df_stock.loc[j, 'effective_return']

                        df_stock.loc[j, 'cumulative_value'] = round((previous_cumulative + current_investment) * (1 + current_return_rate), 0)
                    df_stock['dividend_value'] = 0
                    
                else:
                    # 股息不再投入
                    df_stock['effective_return'] = df_stock['monthly_return']
                    # 計算累積價值
                    df_stock['annuity'] = annuity_capital * weight
                    df_stock['cumulative_value'] = 0
                    df_stock.loc[0, 'cumulative_value'] = np.int64(df_stock.loc[0, 'annuity'] * (1 + df_stock.loc[0, 'effective_return']))
                    
                    for j in range(1, len(df_stock)):
                        previous_cumulative = df_stock.loc[j - 1, 'cumulative_value']
                        current_investment = df_stock.loc[j, 'annuity']
                        current_return_rate = df_stock.loc[j, 'effective_return']

                        df_stock.loc[j, 'cumulative_value'] = round((previous_cumulative + current_investment) * (1 + current_return_rate), 0)
                    df_stock['dividend_value'] = df_stock['cumulative_value'].shift(1).fillna(annuity_capital * weight) * df_stock['dividend_yield']
                
                df_stock.set_index('month', inplace=True)
                df_stock = df_stock.drop('stock_id', axis=1)
                # print(df_stock)
                
                if combined_cumulative_value is None:
                    combined_cumulative_value = df_stock.copy()
                else:
                    combined_cumulative_value += df_stock
            
            df = combined_cumulative_value.copy()
            df['dividend_value'] = df['dividend_value'].round(0)
            # df['cumulative_return'] = (df['cumulative_value'] / initial_capital) - 1
            
            # 為了計算指標，我們需要 `effective_return`，可以從 `cumulative_value` 回推
            # 但這是一個簡化方式，可能與實際的 monthly_return 有微小誤差
            # df['effective_return'] = df['cumulative_value'].pct_change().fillna(0)
        
        df.index = df.index.strftime('%Y-%m')
        # df['cumulative_return'] = df['cumulative_return'].round(6)
        df['effective_return'] = df['effective_return'].round(6)
        df['cumulative_value'] = df['cumulative_value'].round(0)
        
        # print(df)
        
        dataframes.append((f"投資組合{i+1}", df))
        # 用df資料計算顯示在表格的資訊
        metrics.append(calculate_metrics(df, reinvest_dividends, annuity_capital, annuity_mode=True))

    if not all(dataframes):
        return jsonify({'error': '無法找到股票代碼或資料。請重新輸入。'}), 400
    
    # 使用 Plotly.express 創建圖表，但這裡使用更靈活的 go.Figure
    fig = go.Figure()
    
    # 定義三種線條顏色
    line_colors = ["#2c4cff", '#d62728', "#2ed728"]
    
    for i, (stock_id, df) in enumerate(dataframes):
        y_data = df['cumulative_value'].to_list()
        y_title = '累積金額'
        
        fig.add_trace(go.Scatter(
            x=df.index.to_list(),
            y=y_data,
            mode='lines',
            line=dict(color=line_colors[i], width=2),
            name=f'{stock_id} {y_title}'
        ))

    # 圖表美化設定
    fig.update_layout(
        title={
            # 'text': f'{stocks_input[0]} vs {stocks_input[1]} vs {stocks_input[2]}股票累計報酬率比較',
            'text': f'投資組合1 vs 投資組合2 vs 投資組合3 股票累計報酬率比較',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#333')
        },
        xaxis_title='月份',
        yaxis_title='累計報酬率',
        hovermode='x unified', # 鼠標懸停時顯示所有線條的資料
        template='plotly',
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#333"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#130000", weight=10),
            bgcolor='#ffffff',
            bordercolor='#ccc',
            borderwidth=0.5
        ),
        margin=dict(l=50, r=50, t=100, b=50),
        xaxis=dict(
            showgrid=False,
            # gridwidth=1,
            # gridcolor='#555555',
            tickformat="%Y-%m" # X軸時間格式
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.3,
            tickformat="$.0f", # Y軸刻度顯示為百分比
        ),
        hoverlabel=dict(
            # font=dict(color='#888888'),
            # bgcolor='#ffffff',
            font_size=16,
            font_family="Rockwell"
        )
    )

    # 將 Plotly 圖表物件轉換為 JSON 字串，以便傳遞給前端
    # 使用 plotly.io.to_json 來確保輸出標準 JSON 格式
    graph_json = pio.to_json(fig, engine='json')

    # print(type(graph_json))
    print(metrics)
    return jsonify({
        'graph_json': graph_json,
        'metrics': {
            'stocks_input': ['投資組合1', '投資組合2', '投資組合3'],
            'metrics_data': metrics
        }
    })

@app.route('/calculate_compound', methods=['POST'])
def calculate_compound():
    pv = request.json.get('pv')
    fv = request.json.get('fv')
    rate = request.json.get('rate')
    n = request.json.get('n')

    if fv is not None and pv is not None and fv < pv:
        return jsonify({'error': '未來值不可小於現值。'}), 400

    if pv is not None and fv is not None and rate is not None and n is None:
        n = np.log(fv / pv) / np.log(1 + rate)
    elif pv is not None and fv is not None and n is not None and rate is None:
        rate = (fv / pv)**(1/n) - 1
    elif pv is not None and rate is not None and n is not None and fv is None:
        fv = pv * (1 + rate)**n
    elif fv is not None and rate is not None and n is not None and pv is None:
        pv = fv / (1 + rate)**n
    else:
        return jsonify({'error': '請提供三個變數。'}), 400

    return jsonify({
        'pv': round(pv, 0),
        'fv': round(fv, 0),
        'rate': round(rate, 6),
        'n': round(n, 2)
    })

@app.route('/calculate_annuity', methods=['POST'])
def calculate_annuity():
    import numpy_financial as npf
    
    target_output = request.json.get('target_output')
    pv_annuity = request.json.get('pv_annuity') or 0
    fv_annuity = request.json.get('fv_annuity') or 0
    pmt = request.json.get('pmt') or 0
    rate_annuity = request.json.get('rate_annuity') or 0
    n_annuity = request.json.get('n_annuity') or 0

    print(pv_annuity, fv_annuity, pmt, rate_annuity, n_annuity)
    
    # 根據不同目標及現有值計算目標的金額
    # 檢查報酬率為 0 的特殊情況
    if rate_annuity == 0 and target_output != 'rate_annuity':
        if fv_annuity is None:
            fv_annuity = pv_annuity + pmt * n_annuity
        if pv_annuity is None:
            pv_annuity = fv_annuity - pmt * n_annuity
        if pmt is None:
            pmt = (fv_annuity - pv_annuity) / n_annuity
        if n_annuity is None:
            n_annuity = (fv_annuity - pv_annuity) / pmt
    else:
        # 修正：添加輸入驗證
        if rate_annuity < 0:
            return jsonify({'error': '報酬率不能為負數'}), 400
        if n_annuity <= 0:
            return jsonify({'error': '期數必須大於0'}), 400
    
    # 根據缺少的變數，套用對應的公式
    if target_output ==  'fv_annuity':
        # 計算未來值 (FV)
        fv_annuity = pv_annuity * (1 + rate_annuity)**n_annuity + pmt * (((1 + rate_annuity)**n_annuity - 1) / rate_annuity)
    
    elif target_output == 'pv_annuity':
        # 計算現在值 (PV)
        term1 = pmt * (((1 + rate_annuity)**n_annuity - 1) / rate_annuity)
        pv_annuity =  (fv_annuity - term1) / (1 + rate_annuity)**n_annuity
        
    elif target_output == 'pmt':
        # 計算年金 (PMT)
        term1 = pv_annuity * (1 + rate_annuity)**n_annuity
        term2 = ((1 + rate_annuity)**n_annuity - 1) / rate_annuity
        pmt = (fv_annuity - term1) / term2
        
    elif target_output == 'n_annuity':
        # 計算年數 (n)，需要使用對數運算
        # 如果 pv * r + pmt 是負數，則無法計算，這代表無法達成目標
        numerator = (fv_annuity * rate_annuity + pmt)
        denominator = (pv_annuity * rate_annuity + pmt)
        
        if denominator == 0 or numerator/denominator <= 0:
             raise ValueError("無法計算年數，請檢查輸入參數。")

        # 使用 NumPy 進行對數運算
        n_annuity = np.log(numerator / denominator) / np.log(1 + rate_annuity)
    
    elif target_output == 'rate_annuity':
        # 實務上會使用 numpy-financial 或 scipy.optimize 等函式庫。
        # raise NotImplementedError("目前無法透過簡單公式計算報酬率，請使用專門的財務計算函式庫。")
        rate_annuity = npf.rate(nper=n_annuity, pmt=-pmt, pv=-pv_annuity, fv=fv_annuity)

    
    return jsonify({
        'pv_annuity': pv_annuity,
        'fv_annuity': fv_annuity,
        'pmt': pmt,
        'rate_annuity': rate_annuity,
        'n_annuity': n_annuity
    })

@app.route('/calculate_table', methods=['POST'])
def calculate_table():
    initial_capital = request.json.get('initial_capital')
    years = [10, 20, 30]
    rates = [0.04, 0.08, 0.12, 0.16]
    results = []

    for year in years:
        row = []
        for rate in rates:
            future_value = initial_capital * (1 + rate)**year
            row.append(future_value)
        results.append(row)

    return jsonify({'results': results})


if __name__ == '__main__':
    # 啟動 Flask 伺服器
    app.run(debug=True)
