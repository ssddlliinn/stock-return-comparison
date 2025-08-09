from flask import Flask, render_template, request, jsonify
from data import summary_monthly_data
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import json
import numpy as np
import re

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
            df = df[df['month'] >= start_month].copy().reset_index()
            if not df.empty:
                df.set_index('month', inplace=True)
                df = df.drop('stock_id', axis=1)
                if total_df is None:
                    total_df = df * weight
                else:
                    total_df += df * weight

    total_df.reset_index(inplace=True)
    
    return total_df #['month', 'monthly_return', 'dividend_yield']

def calculate_metrics(df, reinvest_dividends, initial_capital):
    """
    計算並回傳量化指標的函數。
    """
    df = df.copy()
    # 計算年化報酬率 (Annualized Return)
    # 取累積報酬率最後一個值就是整個區間整體報酬率
    total_periods = len(df)
    total_cumulative_return = df['cumulative_return'].iloc[-1]
    annualized_return = ((1 + total_cumulative_return)**(12/total_periods) - 1).round(6)

    # 計算波動率 (Annualized Volatility)
    annualized_volatility = (df['effective_return'].std() * np.sqrt(12)).round(6)
    
    # 計算最大跌幅(MDD)
    MDD = (df['cumulative_value'] / df['cumulative_value'].cummax() - 1).min()

    # 計算總領股息
    # specific:表示把殖利率比照第一個投資組合做提領
    # true:表示全部再投資，不會有現金被領出
    # false:按照其原本的配發條件做配發
    df['cumulative_value'] = df['cumulative_value'].shift(1).fillna(initial_capital).copy()
    if 'specific_dividend_yield' in df.columns:
        total_dividends = (df['specific_dividend_yield'] * df['cumulative_value']).round(0).sum()
    elif reinvest_dividends == 'true':
        total_dividends = 0
    elif 'dividend_value' in df.columns:
        total_dividends = df['dividend_value'].sum()
    else:
        total_dividends = (df['dividend_yield'] * df['cumulative_value']).round(0).sum()
    
    # 這裡可以加入更多指標，例如夏普比率、最大回撤等
    
    return {
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'MDD': MDD,
        'total_dividends': total_dividends
    }

@app.route('/')
def index():
    """
    首頁路由，渲染 HTML 模板。
    """
    return render_template('index.html')

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
                df = summary_monthly_data(
                    stock_id=stock_id,
                    market='us' if re.match(r'^[A-Z\^\.]+$', stock_id) else 'tw',
                    start_date=start_date_str,
                    end_date=end_date_str
                )
                
                # 如果有空的資料，就直接回傳找不到資料
                if not df.empty:
                    earliest_month.append(df['month'].min())
                    portfolio_dfs[stock_id] = df
                    print(f'股票:{stock_id}的最早資料年月為{df["month"].min().strftime("%Y-%m")}')
                    assert df["month"].max().strftime("%Y-%m") != end_date_str[:8], f"{stock_id}資料最後與結束日期對不上"
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
        
        # 產出一個['month', 'monthly_return', 'dividend_yield', ('specific_dividend_yield'), 
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
                df_stock = df_stock[df_stock['month'] >= start_month].copy().reset_index()
                
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
                df_stock.set_index('month', inplace=True)
                df_stock = df_stock.drop('stock_id', axis=1)
                # print(df_stock)
                
                if combined_cumulative_value is None:
                    combined_cumulative_value = df_stock.copy()
                else:
                    combined_cumulative_value += df_stock
            
            # 回推投資組合的報酬率
            combined_cumulative_value.reset_index(inplace=True)
            df = combined_cumulative_value.copy()
            df['cumulative_return'] = (df['cumulative_value'] / initial_capital) - 1
            
            # 為了計算指標，我們需要 `effective_return`，可以從 `cumulative_return` 回推
            # 但這是一個簡化方式，可能與實際的 monthly_return 有微小誤差
            df['effective_return'] = df['cumulative_value'].pct_change().fillna(0)
        
        if isinstance(df['month'].iloc[0], pd.Period):
            df['month'] = df['month'].dt.strftime('%Y-%m')
        df['cumulative_return'] = df['cumulative_return'].round(6)
        df['effective_return'] = df['effective_return'].round(6)
        df['cumulative_value'] = df['cumulative_value'].round(0)
        df['dividend_value'] = df['dividend_value'].round(0)
        # print(df)
        
        dataframes.append((f"投資組合{i+1}", df))
        # 用df資料計算顯示在表格的資訊
        metrics.append(calculate_metrics(df, reinvest_dividends, initial_capital))

    if not all(dataframes):
        return jsonify({'error': '無法找到股票代碼或資料。請重新輸入。'}), 400
    
    # 使用 Plotly.express 創建圖表，但這裡使用更靈活的 go.Figure
    fig = go.Figure()
    
    # 定義三種線條顏色
    line_colors = ["#00f2ff", '#d62728', "#eeff00"]
    
    for i, (stock_id, df) in enumerate(dataframes):
        y_data = df['cumulative_value'].to_list() if display_mode == 'value' else df['cumulative_return'].to_list()
        y_title = '累積金額' if display_mode == 'value' else '累計報酬率'
        
        fig.add_trace(go.Scatter(
            x=df['month'].to_list(),
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
        template='plotly_dark', # 使用黑色背景模板
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
            gridcolor='#333333',
            tickformat=".2%" if display_mode == 'return' else "$.0f", # Y軸刻度顯示為百分比
        ),
        hoverlabel=dict(
            font=dict(color='#888888'),
            bgcolor='#ffffff',
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

if __name__ == '__main__':
    # 啟動 Flask 伺服器
    app.run(debug=True)
