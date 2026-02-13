<<<<<<< HEAD
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Gold Trend Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈" 
)

# CSS
st.markdown("""
<style>
    .main { font-family: 'Segoe UI', sans-serif; }
    
    /* Card chỉ số (Metric) */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #DAA520;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        transition: transform 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #f8f9fa;
        border-radius: 6px;
        color: #6c757d;
        font-weight: 600;
        font-size: 14px;
        border: 1px solid #e9ecef;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #DAA520;
        color: white;
        border-color: #DAA520;
    }
    
    /* Sidebar Styling */
    .sidebar-logo-container {
        text-align: center; 
        margin-bottom: 25px;
        padding-bottom: 20px;
        border-bottom: 1px solid #e9ecef;
    }
    
    /* Styled Labels in Sidebar */
    .sidebar-label {
        font-size: 11px;
        font-weight: 800;
        color: #495057;
        text-transform: uppercase;
        margin-bottom: 5px;
        display: block;
        letter-spacing: 0.5px;
    }
    
    /* Button Style */
    div.stButton > button { 
        width: 100%; 
        border-radius: 6px; 
        font-weight: 500;
        transition: all 0.3s;
    }
    div.stButton > button[kind="primary"] {
        background-color: #2c3e50;
        border: none;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #DAA520;
        box-shadow: 0 4px 12px rgba(218, 165, 32, 0.4);
    }
    
    /* Date Badge */
    .date-badge {
        background-color: #f1f3f5;
        color: #495057;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

GOLD_TICKER = 'GC=F'    # Vàng thế giới
USDVND_TICKER = 'VND=X' # Tỷ giá
CACHE_TTL = 21600       # Cache 6 tiếng
SJC_Target = 'Hồ Chí Minh'
OUNCE_TO_TAEL = 1.205653 # 1 Ounce = 1.2 Lượng

try:
    from vnstock.explorer.misc import sjc_gold_price
    VNSTOCK_AVAILABLE = True
except ImportError:
    VNSTOCK_AVAILABLE = False


@st.cache_data(ttl=CACHE_TTL)
def fetch_financial_data(start_date, end_date):
    """Lấy dữ liệu từ Yahoo Finance và xử lý cơ bản."""
    try:
        buffer_date = start_date - timedelta(days=100) # Lấy dư ngày để tính chỉ báo kỹ thuật
        tickers = f"{GOLD_TICKER} {USDVND_TICKER}"
        data = yf.download(tickers, start=buffer_date, end=end_date + timedelta(days=1), progress=False, group_by='ticker')
        
        if data.empty: return pd.DataFrame(), "No Data"

        # Tách dữ liệu
        try:
            df_gold = data[GOLD_TICKER].copy()
            # Giữ lại các cột cần thiết và đổi tên
            df_gold = df_gold[['Open', 'High', 'Low', 'Close']].rename(columns={
                'Open': 'Gold_Open', 'High': 'Gold_High', 'Low': 'Gold_Low', 'Close': 'Gold_Close'
            })
            
            df_usd = data[USDVND_TICKER][['Close']].rename(columns={'Close': 'USDVND'})
        except KeyError:
            return pd.DataFrame(), "Structure Error"

        # Gộp dữ liệu (Inner Join)
        df_merge = df_gold.join(df_usd, how='inner')
        df_merge = df_merge.dropna()
        
        # Tính giá quy đổi sang VND (Triệu đồng/Lượng)
        # Công thức: (Giá USD * Tỷ giá * Hệ số Ounce->Lượng) / 1 Triệu
        df_merge['Gold_VND'] = (df_merge['Gold_Close'] * df_merge['USDVND'] * OUNCE_TO_TAEL) / 1e6
        
        # Cắt đúng ngày user chọn
        mask = (df_merge.index.date >= start_date) & (df_merge.index.date <= end_date)
        return df_merge.loc[mask], None

    except Exception as e:
        return pd.DataFrame(), str(e)

@st.cache_data(ttl=CACHE_TTL)
def fetch_sjc_data(start_date, end_date):
    if not VNSTOCK_AVAILABLE: return pd.DataFrame(), "Missing Lib"
    
    all_prices = []
    days_diff = (end_date - start_date).days
    step = 1 if days_diff < 30 else (5 if days_diff < 180 else 15)
    
    current = start_date
    while current <= end_date:
        try:
            time.sleep(0.05) 
            df = sjc_gold_price(date=current.strftime("%Y-%m-%d"))
            if df is not None and not df.empty:
                row = df[df['branch'] == SJC_Target]
                if not row.empty:
                    buy = float(str(row.iloc[0]['buy_price']).replace(',', ''))
                    sell = float(str(row.iloc[0]['sell_price']).replace(',', ''))
                    all_prices.append({'Date': current, 'SJC_Buy': buy, 'SJC_Sell': sell})
        except: pass
        current += timedelta(days=step)
        
    if not all_prices: return pd.DataFrame(), "No Data"
    
    df_sjc = pd.DataFrame(all_prices)
    df_sjc.set_index('Date', inplace=True)
    return df_sjc, None

def get_live_world_price():
    url = "https://tradingeconomics.com/commodities"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, 'html.parser')
            for table in soup.find_all('table'):
                for row in table.find_all('tr'):
                    if 'Gold' in row.text:
                        cols = row.find_all('td')
                        if len(cols) > 1:
                            return float(cols[1].text.strip().replace(',', ''))
    except: return None
    return None


def process_data_with_currency(df, currency_mode):
    """
    Xử lý dữ liệu dựa trên loại tiền tệ được chọn (USD hoặc VND).
    Hàm này tạo cột 'View_Price' để dùng chung cho các biểu đồ.
    """
    df = df.copy()
    
    if currency_mode == 'VND':
        # Chế độ VND(Triệu đồng/Lượng)
        df['View_Price'] = df['Gold_VND']
        df['View_Open'] = (df['Gold_Open'] * df['USDVND'] * OUNCE_TO_TAEL) / 1e6
        df['View_High'] = (df['Gold_High'] * df['USDVND'] * OUNCE_TO_TAEL) / 1e6
        df['View_Low'] = (df['Gold_Low'] * df['USDVND'] * OUNCE_TO_TAEL) / 1e6
    else:
        # Chế độ USD(USD/oz)
        df['View_Price'] = df['Gold_Close']
        df['View_Open'] = df['Gold_Open']
        df['View_High'] = df['Gold_High']
        df['View_Low'] = df['Gold_Low']
        
    return df

def add_technical_indicators(df):
    """Tính chỉ báo kỹ thuật dựa trên cột View_Price đã xử lý tiền tệ"""
    df = df.copy()
    
    # Target column for calculation
    target = df['View_Price']
    
    df['SMA_20'] = target.rolling(window=20).mean()
    df['SMA_50'] = target.rolling(window=50).mean()
    
    delta = target.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['BB_Std'] = target.rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['SMA_20'] - (2 * df['BB_Std'])
    
    ema12 = target.ewm(span=12, adjust=False).mean()
    ema26 = target.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['Daily_Return'] = target.pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod()
    df['Rolling_Max'] = df['Cumulative_Return'].cummax()
    df['Drawdown'] = df['Cumulative_Return'] / df['Rolling_Max'] - 1.0
    
    df.dropna(inplace=True)
    return df

def style_chart(fig):
    """Tối ưu hóa giao diện biểu đồ"""
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI", color="#2c3e50", size=11),
        xaxis=dict(
            showgrid=False,  # Bỏ lưới dọc
            linecolor='#e0e0e0',
            tickformat='%d/%m'
        ),
        yaxis=dict(
            showgrid=True,   # Giữ lưới ngang mờ
            gridcolor='#f5f5f5', 
            gridwidth=1,
            zeroline=False
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(
            orientation="h", y=1.02, x=1, xanchor="right",
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e0e0e0', borderwidth=1
        )
    )
    return fig


with st.sidebar:
    st.markdown("""
        <div class="sidebar-logo-container">
            <h1 style="color: #DAA520; margin:0; font-size: 32px;">GOLD PRO</h1>
            <p style="font-size: 13px; font-weight: 600; color: #7f8c8d; letter-spacing: 1px;">MARKET INTELLIGENCE</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<span class="sidebar-label">💱 ĐƠN VỊ HIỂN THỊ</span>', unsafe_allow_html=True)
        currency_mode = st.radio(
            "Chọn đơn vị tiền tệ",
            ["USD", "VND"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Setup Unit Labels
        if currency_mode == 'USD':
            unit_label = "USD/oz"
            currency_symbol = "$"
        else:
            unit_label = "Triệu VNĐ/Lượng"
            currency_symbol = "₫"
        
        st.markdown("---")
        
        st.markdown('<span class="sidebar-label">🗓️ KHUNG THỜI GIAN</span>', unsafe_allow_html=True)
        range_option = st.selectbox(
            "Chọn thời gian", # Thêm label để tránh lỗi
            ["3 Tháng", "6 Tháng", "1 Năm", "3 Năm", "5 Năm"], 
            index=2, 
            label_visibility="collapsed"
        )
        
        days_map = {"3 Tháng": 90, "6 Tháng": 180, "1 Năm": 365, "3 Năm": 1095, "5 Năm": 1825}
        end_input = datetime.now().date()
        start_input = end_input - timedelta(days=days_map[range_option])
        
        # Hiển thị ngày tháng dạng Badge
        st.markdown(f"""
            <div style="display: flex; gap: 5px; margin-top: 10px;">
                <div class="date-badge">TỪ: {start_input.strftime('%d/%m/%Y')}</div>
                <div class="date-badge">ĐẾN: {end_input.strftime('%d/%m/%Y')}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown('<span class="sidebar-label">⚡ DỮ LIỆU THỰC</span>', unsafe_allow_html=True)
        live_price = None
        if st.button("CẬP NHẬT GIÁ LIVE (USD)", use_container_width=True, type="primary"):
            with st.spinner("Đang kết nối..."):
                live_price = get_live_world_price()
                if live_price: st.toast(f"✅ Giá: ${live_price}")
                else: st.toast("⚠️ Lỗi kết nối")
    
    st.caption("Data Source: Yahoo Finance & Vnstock")

# FETCH DATA
with st.spinner("Đang tải dữ liệu..."):
    df_world, w_err = fetch_financial_data(start_input, end_input)
    df_sjc, s_err = fetch_sjc_data(start_input, end_input)

if not df_world.empty:
    df_processed = process_data_with_currency(df_world, currency_mode)
    df_full = add_technical_indicators(df_processed)
else:
    df_full = pd.DataFrame()

# HEADER
st.markdown(f"## 📊 Bảng Tin Thị Trường Vàng ({currency_mode})")
st.markdown(f"**Cập nhật:** {datetime.now().strftime('%H:%M %d/%m/%Y')}")

tab1, tab2, tab3, tab4 = st.tabs(["TỔNG QUAN", "HIỆU SUẤT ĐẦU TƯ", "PHÂN TÍCH KỸ THUẬT", "MACHINE LEARNING"])

with tab1:
    st.markdown("### 📌 Chỉ Số Quan Trọng")
    c1, c2, c3, c4 = st.columns(4)
    
    if not df_full.empty:
        curr = df_full['View_Price'].iloc[-1]
        prev = df_full['View_Price'].iloc[-2]
        change = curr - prev
        
        # Giá Vàng (Theo đơn vị đã chọn)
        c1.metric(f"Giá Vàng ({currency_mode})", f"{currency_symbol}{curr:,.2f}", f"{change:,.2f} {currency_mode}")
        
        # Tỷ Giá
        c2.metric("Tỷ Giá USD/VND", f"{df_world['USDVND'].iloc[-1]:,.0f} ₫")
        
        # Giá Gốc (Ngược lại)
        if currency_mode == 'USD':
            vn_equiv = df_world['Gold_VND'].iloc[-1]
            c3.metric("Quy Đổi VND", f"{vn_equiv:,.2f} Tr/Lượng")
        else:
            usd_orig = df_world['Gold_Close'].iloc[-1]
            c3.metric("Giá Gốc USD", f"${usd_orig:,.2f}")
        
    sjc_val = df_sjc['SJC_Sell'].iloc[-1]/1e6 if not df_sjc.empty else 0
    c4.metric("SJC Bán Ra (VND)", f"{sjc_val:,.2f} Tr" if sjc_val else "N/A")
    st.divider()

    st.markdown(f"##### 1. Xu Hướng Giá Vàng ({unit_label})")
    fig1 = px.area(df_full, y='View_Price', template="plotly_white")

    chart_color = '#DAA520' if currency_mode == 'USD' else '#2E86C1'
    fig1.update_traces(line_color=chart_color, fillcolor=f"rgba({int(chart_color[1:3],16)}, {int(chart_color[3:5],16)}, {int(chart_color[5:7],16)}, 0.1)")
    fig1.update_layout(yaxis_title=unit_label)
    st.plotly_chart(style_chart(fig1), use_container_width=True)

    col_chart_1, col_chart_2 = st.columns(2)

    with col_chart_1:
        st.markdown("##### 2. Chi Tiết Giá (Candlestick)")
        fig2 = go.Figure(data=[go.Candlestick(x=df_full.index,
                        open=df_full['View_Open'], high=df_full['View_High'],
                        low=df_full['View_Low'], close=df_full['View_Price'])])
        fig2.update_layout(xaxis_rangeslider_visible=False, template="plotly_white", yaxis_title=unit_label)
        st.plotly_chart(style_chart(fig2), use_container_width=True)

    with col_chart_2:
        st.markdown(f"##### 3. Tương Quan Vàng ({currency_mode}) & Tỷ Giá")
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(x=df_full.index, y=df_full['View_Price'], name=f"Vàng ({currency_mode})", line=dict(color=chart_color)), secondary_y=False)
        fig3.add_trace(go.Scatter(x=df_full.index, y=df_full['USDVND'], name="USDVND", line=dict(color="#34495e", dash='dot')), secondary_y=True)
        st.plotly_chart(style_chart(fig3), use_container_width=True)

    st.markdown("##### 4. Hiệu Suất Theo Tháng (Seasonality)")
    df_month = df_full.copy()
    df_month['Month'] = df_month.index.strftime('%m-%Y')
    monthly_ret = df_month.resample('M')['Daily_Return'].sum() * 100
    fig4 = px.bar(x=monthly_ret.index.strftime('%Y-%m'), y=monthly_ret.values, 
                  color=monthly_ret.values, color_continuous_scale="RdYlGn")
    fig4.update_layout(xaxis_title="Tháng", yaxis_title="Lợi nhuận (%)")
    st.plotly_chart(style_chart(fig4), use_container_width=True)

with tab2:
    st.markdown(f"### 💰 Phân Tích Lợi Nhuận ({currency_mode})")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.info(f"👇 **Giả lập đầu tư ({currency_mode})**")
        invest = st.number_input(f"Vốn Đầu Tư ({currency_symbol})", value=1000 if currency_mode=='USD' else 50, step=100 if currency_mode=='USD' else 10)
        buy_date = st.date_input("Ngày mua", value=df_full.index.min().date(), min_value=df_full.index.min().date(), max_value=df_full.index.max().date())
        
        idx = df_full.index.get_indexer([pd.Timestamp(buy_date)], method='nearest')[0]
        buy_p = df_full['View_Price'].iloc[idx]
        curr_p = df_full['View_Price'].iloc[-1]
        profit = (invest / buy_p * curr_p) - invest
        
        st.metric("Lợi Nhuận Dự Tính", f"{currency_symbol}{profit:,.2f}", f"{(profit/invest)*100:.1f}%")

    with c2:
        st.markdown("##### 1. Tăng Trưởng Tài Sản (%)")
        fig_cum = px.line(df_full, y='Cumulative_Return', template="plotly_white")
        fig_cum.add_hline(y=1, line_dash="dash", line_color="grey")
        fig_cum.update_traces(line_color="#27ae60", fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)')
        st.plotly_chart(style_chart(fig_cum), use_container_width=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("##### 2. Mức Độ Sụt Giảm (Max Drawdown)")
        fig_dd = px.area(df_full, y='Drawdown', template="plotly_white")
        fig_dd.update_traces(line_color="#c0392b", fillcolor="rgba(192, 57, 43, 0.3)")
        st.plotly_chart(style_chart(fig_dd), use_container_width=True)
    
    with r2:
        st.markdown("##### 3. Biến Động Giá 30 Ngày")
        df_full['Vol_30'] = df_full['Daily_Return'].rolling(30).std()
        fig_vol = px.line(df_full, y='Vol_30', template="plotly_white")
        fig_vol.update_traces(line_color="#8e44ad")
        st.plotly_chart(style_chart(fig_vol), use_container_width=True)

    st.markdown("##### 4. Phân Phối Lợi Nhuận Ngày")
    fig_hist = px.histogram(df_full, x='Daily_Return', nbins=50, color_discrete_sequence=['#34495e'])
    st.plotly_chart(style_chart(fig_hist), use_container_width=True)

with tab3:
    st.markdown(f"### 🛠️ Chỉ Báo Kỹ Thuật ({unit_label})")
    
    t1, t2 = st.columns(2)
    
    with t1:
        st.markdown("##### 1. Bollinger Bands (20, 2)")
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df_full.index, y=df_full['BB_Upper'], line=dict(color='gray', width=1), name='Upper'))
        fig_bb.add_trace(go.Scatter(x=df_full.index, y=df_full['BB_Lower'], line=dict(color='gray', width=1), fill='tonexty', name='Lower'))
        fig_bb.add_trace(go.Scatter(x=df_full.index, y=df_full['View_Price'], line=dict(color=chart_color, width=2), name='Price'))
        fig_bb.update_layout(showlegend=False, yaxis_title=unit_label)
        st.plotly_chart(style_chart(fig_bb), use_container_width=True)

    with t2:
        st.markdown("##### 2. SMA Crossover (Ngắn vs Dài)")
        fig_sma = go.Figure()
        fig_sma.add_trace(go.Scatter(x=df_full.index, y=df_full['SMA_20'], line=dict(color='#2980b9'), name='SMA 20'))
        fig_sma.add_trace(go.Scatter(x=df_full.index, y=df_full['SMA_50'], line=dict(color='#e74c3c'), name='SMA 50'))
        fig_sma.add_trace(go.Scatter(x=df_full.index, y=df_full['View_Price'], line=dict(color=chart_color, width=1), opacity=0.5, name='Price'))
        st.plotly_chart(style_chart(fig_sma), use_container_width=True)

    t3, t4 = st.columns(2)
    
    with t3:
        st.markdown("##### 3. RSI (Sức Mạnh Tương Đối)")
        fig_rsi = px.line(df_full, y='RSI')
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_traces(line_color='#8e44ad')
        fig_rsi.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(style_chart(fig_rsi), use_container_width=True)

    with t4:
        st.markdown("##### 4. MACD (Động Lượng)")
        fig_macd = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig_macd.add_trace(go.Scatter(x=df_full.index, y=df_full['MACD'], line=dict(color='#2980b9'), name='MACD'), row=1, col=1)
        fig_macd.add_trace(go.Scatter(x=df_full.index, y=df_full['MACD_Signal'], line=dict(color='#e67e22'), name='Signal'), row=1, col=1)
        colors = np.where(df_full['MACD_Hist'] < 0, '#e74c3c', '#27ae60')
        fig_macd.add_trace(go.Bar(x=df_full.index, y=df_full['MACD_Hist'], marker_color=colors, name='Hist'), row=2, col=1)
        fig_macd.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig_macd), use_container_width=True)

with tab4:
    st.markdown(f"### 🤖 Phòng Thí Nghiệm AI ({currency_mode})")
    
    # ML Prep
    df_ml = df_full.copy()
    df_ml['Lag1'] = df_ml['View_Price'].shift(1)
    df_ml['Lag2'] = df_ml['View_Price'].shift(2)
    df_ml['Target'] = df_ml['View_Price'].shift(-1)
    df_ml.dropna(inplace=True)
    
    feats = ['Lag1', 'Lag2', 'SMA_20', 'RSI', 'MACD', 'USDVND']
    X = df_ml[feats]
    y = df_ml['Target']
    
    split = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    c_ai1, c_ai2 = st.columns([1, 2])
    with c_ai1:
        next_pred = rf.predict(df_ml.iloc[[-1]][feats])[0]
        st.markdown(f"""
        <div style="background-color: #2c3e50; color: white; padding: 20px; border-radius: 6px; text-align: center;">
            <p style="margin:0; font-size: 12px; opacity: 0.8;">DỰ BÁO NGÀY MAI (T+1)</p>
            <h2 style="margin: 5px 0; color: #f1c40f;">{currency_symbol}{next_pred:,.2f}</h2>
            <hr style="border-color: rgba(255,255,255,0.2);">
            <div style="font-size: 12px;">MAE: {currency_symbol}{mean_absolute_error(y_test, y_pred):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c_ai2:

        imp = pd.DataFrame({'Feat': feats, 'Imp': rf.feature_importances_}).sort_values('Imp')
        fig_imp = px.bar(imp, x='Imp', y='Feat', orientation='h', title="1. Yếu Tố Ảnh Hưởng Nhất", color_discrete_sequence=['#16a085'])
        st.plotly_chart(style_chart(fig_imp), use_container_width=True)

    a1, a2 = st.columns(2)
    
    with a1:
        st.markdown("##### 2. Kiểm Thử (Backtest)")
        df_res = pd.DataFrame({'Thực tế': y_test, 'Dự báo': y_pred}, index=y_test.index)
        fig_back = go.Figure()
        fig_back.add_trace(go.Scatter(x=df_res.index, y=df_res['Thực tế'], name='Thực tế', line=dict(color='#bdc3c7')))
        fig_back.add_trace(go.Scatter(x=df_res.index, y=df_res['Dự báo'], name='AI', line=dict(color='#f39c12', dash='dot')))
        st.plotly_chart(style_chart(fig_back), use_container_width=True)

    with a2:
        st.markdown("##### 3. Độ Tuyến Tính (Linearity)")
        fig_scat = px.scatter(x=y_test, y=y_pred, labels={'x': 'Giá Thực Tế', 'y': 'Giá Dự Báo'})
        fig_scat.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="red", dash="dash"))
        st.plotly_chart(style_chart(fig_scat), use_container_width=True)
        
    st.markdown("##### 4. Phân Phối Sai Số (Residuals)")
    residuals = y_test - y_pred
    fig_res = px.histogram(x=residuals, nbins=50, color_discrete_sequence=['#c0392b'], labels={'x': f'Mức độ sai lệch ({currency_symbol})'})
    st.plotly_chart(style_chart(fig_res), use_container_width=True)

st.markdown("---")
st.caption("Data Analyst Project | Powered by Streamlit & Yahoo Finance")
=======
import streamlit as st
import pandas as pd
import yfinance as yf # Back to yfinance for historical world gold
import plotly.express as px
from datetime import datetime, timedelta
import time # Required for time.sleep
import requests # For scraping current price
from bs4 import BeautifulSoup # For scraping current price

# Import the specific function if possible, otherwise rely on vnstock being installed
try:
    from vnstock.explorer.misc import sjc_gold_price
except ImportError:
    st.error("Thư viện 'vnstock' chưa được cài đặt hoặc không tìm thấy hàm 'sjc_gold_price'. Vui lòng cài đặt: pip install vnstock")
    st.stop()


# --- Constants ---
GOLD_TICKER = 'GC=F' # World Gold ticker (USD/Ounce)
# LOGO_URL_SIDEBAR = "https://res.cloudinary.com/dd7gti2kn/image/upload/v1745678186/samples/people/LOGO_LQP_msfted.png"
SJC_FETCH_INTERVAL_DAYS = 10
SJC_FETCH_DELAY_SECONDS = 2
SJC_TARGET_BRANCH = 'Hồ Chí Minh'
CACHE_TTL_SECONDS = 21600 # Cache yfinance/vnstock data for 6 hours
SCRAPE_CACHE_TTL_SECONDS = 60 # Cache scraped world gold price for 60 seconds

# --- Set Page Config FIRST ---
st.set_page_config(
    page_title="Biểu đồ Giá Vàng",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🪙"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding: 1.5rem 2rem; }
    [data-testid="stSidebar"] { padding-top: 1rem; }
    .stAlert, [data-testid="stExpander"] { border-radius: 0.5rem; border: 1px solid #eee; }
    [data-testid="stExpander"] summary { font-weight: 600; }
    .footer-caption { color: grey; font-size: 0.85em; }
    .footer-copyright { text-align: right; color: grey; font-size: 0.85em; }
    [data-testid="stMetric"] { background-color: #FFFFFF; border: 1px solid #e6e6e6; border-radius: 0.5rem; padding: 1rem 1.25rem; transition: box-shadow 0.2s ease-in-out; }
    [data-testid="stMetric"]:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    [data-testid="stMetricLabel"] { font-weight: 500; color: #555555; font-size: 0.9em; padding-bottom: 0.25rem; }
    [data-testid="stMetricValue"] { font-weight: 600; font-size: 1.7em; color: #1E1E1E; white-space: nowrap; overflow: hidden; text-overflow: clip; line-height: 1.3; }
    [data-testid="stMetricDelta"] { font-weight: 500; font-size: 0.9em; padding-top: 0.25rem; }
    h2 { margin-bottom: 0.8rem; margin-top: 1.5rem; }
    .stPlotlyChart { margin-bottom: 1.5rem; }
    .sidebar-title { font-size: 1.5em; font-weight: 600; padding-bottom: 1rem; text-align: center; color: #333; }
</style>
""", unsafe_allow_html=True)


# --- Data Fetching Function (World Gold USD/Ounce via yfinance) ---
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_world_gold_usd(start_date, end_date):
    """Fetches world gold data in USD/Ounce using yfinance. Returns (dataframe, raw_dataframe, error_type)"""
    try:
        time.sleep(0.5)
        gold_data_raw = yf.download(GOLD_TICKER, start=start_date, end=end_date + timedelta(days=1), progress=False)
        if gold_data_raw.empty:
            return pd.DataFrame(), None, "nodata"
        processed_data = gold_data_raw[['Close']].copy()
        if isinstance(processed_data.columns, pd.MultiIndex):
            processed_data.columns = processed_data.columns.get_level_values(0)
            processed_data = processed_data.loc[:,~processed_data.columns.duplicated()]
        if 'Close' not in processed_data.columns: raise ValueError("Missing 'Close' column")
        processed_data.rename(columns={'Close': 'Giá TG (USD/oz)'}, inplace=True)
        processed_data.reset_index(inplace=True)
        processed_data.rename(columns={'Date': 'Timestamp'}, inplace=True)
        processed_data['Timestamp'] = pd.to_datetime(processed_data['Timestamp'])
        return processed_data, gold_data_raw, None
    except Exception as e:
        error_str = str(e).lower()
        if 'ratelimit' in error_str or 'too many requests' in error_str:
             print(f"Yahoo Finance Rate Limit Error (World Data): {e}")
             return pd.DataFrame(), None, "ratelimit"
        else:
             print(f"Error fetching world data: {e}")
             return pd.DataFrame(), None, "other"

# --- Web Scraping Functions for Current World Gold Price ---
def fetch_web_data():
    """Tải nội dung HTML từ trang web Trading Economics."""
    url = "https://tradingeconomics.com/commodities"
    headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Network or HTTP error fetching Trading Economics: {e}")
        return None
    except Exception as e:
        print(f"Unknown error fetching web data: {e}")
        return None

def clean_major_name(major):
    return major.split("\n\n")[0].strip() if "\n\n" in major else major.strip()

def format_value(value):
    try:
        f_value = float(str(value).replace(',', ''))
        return f"{f_value:.2f}"
    except ValueError:
        print(f"Cannot format value: {value}")
        return None

@st.cache_data(ttl=SCRAPE_CACHE_TTL_SECONDS)
def get_world_gold_price_scrape():
    """Trích xuất giá vàng thế giới hiện tại từ Trading Economics. Returns (price_float, error_type)"""
    html_content = fetch_web_data()
    if not html_content: return None, "fetch_error"
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        target_table = soup.find('table', {'class': 'table table-hover table-striped table-heatmap'})
        if not target_table:
            tables = soup.find_all('table')
            if len(tables) < 2: return None, "parse_error_no_table"
            target_table = tables[1]
        rows = target_table.find_all('tr')
        if not rows or len(rows) < 2: return None, "parse_error_no_rows"
        data_rows = rows[1:]
        for row in data_rows:
            cols = row.find_all(['td', 'th'], recursive=False)
            if len(cols) > 1:
                commodity_name_element = cols[0].find('b')
                if commodity_name_element:
                    commodity_name = clean_major_name(commodity_name_element.text)
                    if commodity_name == "Gold":
                        price_str = cols[1].text.strip()
                        formatted_price_str = format_value(price_str)
                        if formatted_price_str is None: return None, "format_error"
                        try:
                            price_float = float(formatted_price_str)
                            return price_float, None # Success
                        except (ValueError, TypeError): return None, "conversion_error"
        return None, "not_found"
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return None, "parse_error_exception"

# --- Data Fetching Function (SJC Historical Buy/Sell/Spread via vnstock) ---
@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_sjc_historical_data_buy_sell_spread(start_date, end_date):
    """
    Fetches historical SJC gold buy and sell prices, calculates spread.
    Returns (dataframe, error_type)
    """
    all_sjc_prices = []
    current_date = start_date
    rate_limit_encountered = False; other_error_encountered = False
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        try:
            time.sleep(0.2)
            prices = sjc_gold_price(date=date_str)
            if not prices.empty:
                target_row = prices[prices['branch'] == SJC_TARGET_BRANCH]
                if not target_row.empty:
                    buy_price_str = str(target_row.iloc[0]['buy_price']).replace(',', '')
                    sell_price_str = str(target_row.iloc[0]['sell_price']).replace(',', '')
                    buy_price = pd.to_numeric(buy_price_str, errors='coerce')
                    sell_price = pd.to_numeric(sell_price_str, errors='coerce')
                    if pd.notna(buy_price) and pd.notna(sell_price):
                         all_sjc_prices.append({
                             'Timestamp': pd.to_datetime(current_date),
                             'Giá Mua SJC (VND/cây)': buy_price,
                             'Giá Bán SJC (VND/cây)': sell_price
                         })
        except Exception as e:
            error_str = str(e).lower();
            if 'ratelimit' in error_str or 'too many requests' in error_str: rate_limit_encountered = True
            else: other_error_encountered = True
            print(f"Error fetching SJC on {date_str}: {e}")
        current_date += timedelta(days=SJC_FETCH_INTERVAL_DAYS)
        if current_date <= end_date: time.sleep(SJC_FETCH_DELAY_SECONDS)
    if not all_sjc_prices and (rate_limit_encountered or other_error_encountered):
        error_type = "ratelimit" if rate_limit_encountered else "other"
        return pd.DataFrame(), error_type
    elif not all_sjc_prices: return pd.DataFrame(), "nodata"
    else:
         df = pd.DataFrame(all_sjc_prices)
         if 'Giá Bán SJC (VND/cây)' in df.columns and 'Giá Mua SJC (VND/cây)' in df.columns and \
            pd.api.types.is_numeric_dtype(df['Giá Bán SJC (VND/cây)']) and \
            pd.api.types.is_numeric_dtype(df['Giá Mua SJC (VND/cây)']):
             df['Chênh lệch Mua/Bán SJC'] = df['Giá Bán SJC (VND/cây)'] - df['Giá Mua SJC (VND/cây)']
         else: df['Chênh lệch Mua/Bán SJC'] = pd.NA
         return df, None

# --- Removed Calculation Function for World Gold VND ---

# --- Streamlit App Layout ---

# --- Sidebar for Controls - Rearranged ---
with st.sidebar:
    st.markdown("<p class='sidebar-title'>Gold Trend Chart</p>", unsafe_allow_html=True)

    # Section for Current World Gold Price Update
    st.header("🔄 Giá TG Hiện tại")
    update_world_button = st.button("Lấy giá TG mới nhất", key="update_world", help="Lấy giá vàng thế giới hiện tại từ Trading Economics (có thể không ổn định).")
    st.divider()

    # Section for Historical Data Selection
    st.header("📅 Thời gian Lịch sử")
    predefined_ranges = { "1 Tháng": 30, "3 Tháng": 90, "6 Tháng": 180, "1 Năm": 365, "Từ đầu năm (YTD)": "YTD", "Tất cả (Tối đa 10 năm)": "Max" }
    selected_range_label = st.selectbox("Chọn nhanh:", options=list(predefined_ranges.keys()), index=2)
    st.markdown("**Hoặc chọn ngày:**")
    today = datetime.now().date()
    if selected_range_label == "Tất cả (Tối đa 10 năm)": default_start_date_calc = max(today - timedelta(days=10*365), datetime(2015, 1, 1).date())
    elif selected_range_label == "Từ đầu năm (YTD)": default_start_date_calc = datetime(today.year, 1, 1).date()
    else: default_start_date_calc = today - timedelta(days=predefined_ranges[selected_range_label])
    default_end_date_calc = today
    start_date_input = st.date_input("Từ ngày", default_start_date_calc, label_visibility="collapsed")
    end_date_input = st.date_input("Đến ngày", default_end_date_calc, label_visibility="collapsed")
    start_date = start_date_input; end_date = end_date_input
    final_label_hist = f"{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}"
    if start_date > end_date: st.error("Lỗi: Ngày bắt đầu không được sau ngày kết thúc."); st.stop()
    st.divider()

    # SJC Fetch Info Caption
    st.caption(f"SJC lấy mỗi {SJC_FETCH_INTERVAL_DAYS} ngày ({SJC_TARGET_BRANCH}).")

# --- Main Page Layout ---
st.title("📊 Biểu đồ Lịch sử Giá Vàng")
st.caption(f"Giá TG (USD/oz) & Giá SJC (VND/cây) | Khoảng thời gian lịch sử: {final_label_hist}")
st.write("")

# --- Initialize variables ---
world_data_error = False
sjc_data_error = False
world_gold_usd_hist = pd.DataFrame()
sjc_hist = pd.DataFrame()
gold_hist_raw = None
scraped_world_price = None
scraped_world_fetch_error = None

# --- Handle Scrape Button Click ---
if update_world_button:
    with st.spinner("Đang lấy giá TG mới nhất..."):
        scraped_world_price, scraped_world_fetch_error = get_world_gold_price_scrape()
        if scraped_world_fetch_error:
            if scraped_world_fetch_error == "fetch_error": st.error("⚠️ Lỗi mạng hoặc không thể kết nối đến Trading Economics để lấy giá TG.", icon="🌐")
            elif scraped_world_fetch_error == "not_found": st.error("⚠️ Không tìm thấy giá vàng ('Gold') trong dữ liệu từ Trading Economics.", icon="🔍")
            else: st.error(f"❌ Đã xảy ra lỗi khi xử lý dữ liệu giá TG từ Trading Economics (code: {scraped_world_fetch_error}).")
            scraped_world_price = None
        elif scraped_world_price is not None:
            st.toast(f"Đã lấy giá TG hiện tại: ${scraped_world_price:.2f}", icon="✅") # Use toast for success
        else:
             st.error("Không thể lấy hoặc xử lý giá vàng thế giới lần này.")

# --- Fetch Historical World Data (yfinance) ---
world_fetch_error_type = None
fetch_world_success = False
with st.spinner(f"Đang tải dữ liệu lịch sử giá TG (USD/oz)..."):
    world_gold_usd_hist, gold_hist_raw, world_fetch_error_type = fetch_world_gold_usd(start_date, end_date)
    if world_fetch_error_type: world_data_error = True
    elif world_gold_usd_hist.empty: world_data_error = True
    else: fetch_world_success = True

# Display yfinance fetch status (only if scraping wasn't just attempted or failed)
if world_fetch_error_type and not update_world_button:
    if world_fetch_error_type == "ratelimit": st.warning("⚠️ **Giới hạn truy cập (Lịch sử TG):** Máy chủ Yahoo Finance đang tạm thời giới hạn truy cập. Biểu đồ lịch sử giá TG có thể không hiển thị. Vui lòng thử lại sau ít phút.", icon="⏳")
    elif world_fetch_error_type == "nodata": st.info("ℹ️ Không tìm thấy dữ liệu lịch sử giá thế giới cho khoảng thời gian này.")
    elif world_fetch_error_type == "other": st.error("❌ Đã xảy ra lỗi khi tải dữ liệu lịch sử giá thế giới.")

# --- Fetch SJC Data ---
sjc_fetch_error_type = None
fetch_sjc_success = False
with st.spinner(f"Đang tải dữ liệu giá SJC (Mua/Bán/Chênh lệch)..."):
     sjc_hist, sjc_fetch_error_type = fetch_sjc_historical_data_buy_sell_spread(start_date, end_date)
     if sjc_fetch_error_type: sjc_data_error = True
     elif sjc_hist.empty:
         sjc_data_error = True
         if sjc_fetch_error_type is None: sjc_fetch_error_type = "nodata"
     else:
         sjc_hist['Timestamp'] = pd.to_datetime(sjc_hist['Timestamp'])
         fetch_sjc_success = True

# Display SJC status message outside spinner (only show errors/warnings)
if sjc_fetch_error_type == "ratelimit": st.warning(f"⚠️ **Giới hạn truy cập (Giá SJC):** Có thể đã gặp giới hạn khi lấy dữ liệu SJC. Dữ liệu SJC có thể không đầy đủ hoặc không hiển thị. Vui lòng thử lại sau.", icon="⏳")
elif sjc_fetch_error_type == "nodata": st.info(f"ℹ️ Không tìm thấy dữ liệu SJC nào cho khoảng thời gian đã chọn ({final_label_hist}).")
elif sjc_fetch_error_type == "other": st.error("❌ Đã xảy ra lỗi khi tải dữ liệu SJC.")


# --- Display Metrics ---
col1, col2, col3, col4 = st.columns(4)

# Function to format delta string (USD and VND)
def format_delta(delta_value, currency="VND"):
    if delta_value is None or pd.isna(delta_value): return None
    sign = "+" if delta_value > 0 else ""
    if currency == "USD": return f"{sign}{delta_value:,.2f} USD"
    else: return f"{sign}{delta_value:,.0f} VND"

# Helper function to safely get scalar value
def get_scalar(value):
     if isinstance(value, (pd.Series, pd.DataFrame)):
         if not value.empty:
             try: return value.iloc[0]
             except IndexError: return None
         else: return None
     if pd.isna(value): return None
     return value

# Metric 1: World Gold (USD) - Show scraped price if available, else historical
latest_world_price_usd = None; latest_world_date = None; latest_world_date_str = "N/A"; delta_world_usd = None; metric1_label = "Giá TG (Lịch sử)"
if scraped_world_price is not None: # Prioritize scraped price if available
    latest_world_price_usd = scraped_world_price
    latest_world_date_str = datetime.now().strftime('%H:%M')
    metric1_label = f"Giá TG ({latest_world_date_str})"
    if not world_data_error and not world_gold_usd_hist.empty:
        last_hist_price = get_scalar(world_gold_usd_hist.iloc[-1]['Giá TG (USD/oz)'])
        if pd.notna(latest_world_price_usd) and pd.notna(last_hist_price):
            delta_world_usd = latest_world_price_usd - last_hist_price
elif not world_data_error and not world_gold_usd_hist.empty: # Fallback to historical data
    try:
        latest_row_world = world_gold_usd_hist.iloc[-1]
        latest_world_price_usd = get_scalar(latest_row_world['Giá TG (USD/oz)'])
        latest_world_date = get_scalar(latest_row_world['Timestamp'])
        if isinstance(latest_world_date, pd.Timestamp): latest_world_date_str = latest_world_date.strftime('%d/%m')
        metric1_label = f"Giá TG ({latest_world_date_str})"
        if len(world_gold_usd_hist) > 1:
            prev_world_price_usd = get_scalar(world_gold_usd_hist.iloc[-2]['Giá TG (USD/oz)'])
            if pd.notna(latest_world_price_usd) and isinstance(latest_world_price_usd, (int, float, complex)) and \
               pd.notna(prev_world_price_usd) and isinstance(prev_world_price_usd, (int, float, complex)):
                 delta_world_usd = latest_world_price_usd - prev_world_price_usd
    except Exception as e: print(f"Error processing world gold metric: {e}"); latest_world_price_usd = None; latest_world_date_str = "N/A"; delta_world_usd = None; metric1_label = "Giá TG (Lỗi)"

with col1: st.metric(label=metric1_label, value=f"{latest_world_price_usd:,.2f} USD" if pd.notna(latest_world_price_usd) else "N/A", delta=format_delta(delta_world_usd, "USD"), help="Giá vàng thế giới (USD/Ounce)")

# Metrics 2, 3, 4 for SJC remain the same
# ... (SJC metric code is unchanged) ...
latest_sjc_sell = None; latest_sjc_date = None; latest_sjc_date_str = "N/A"; delta_sjc_sell = None
if not sjc_data_error and not sjc_hist.empty:
    try:
        latest_row_sjc = sjc_hist.iloc[-1]
        latest_sjc_sell = get_scalar(latest_row_sjc['Giá Bán SJC (VND/cây)'])
        latest_sjc_date = get_scalar(latest_row_sjc['Timestamp'])
        if isinstance(latest_sjc_date, pd.Timestamp): latest_sjc_date_str = latest_sjc_date.strftime('%d/%m')
        if len(sjc_hist) > 1:
             prev_sjc_sell = get_scalar(sjc_hist.iloc[-2]['Giá Bán SJC (VND/cây)'])
             if pd.notna(latest_sjc_sell) and isinstance(latest_sjc_sell, (int, float, complex)) and \
                pd.notna(prev_sjc_sell) and isinstance(prev_sjc_sell, (int, float, complex)):
                 delta_sjc_sell = latest_sjc_sell - prev_sjc_sell
    except Exception as e: print(f"Error processing SJC sell metric: {e}"); latest_sjc_sell = None; latest_sjc_date_str = "N/A"; delta_sjc_sell = None
with col2: st.metric(label=f"SJC Bán ({latest_sjc_date_str})", value=f"{latest_sjc_sell:,.0f}" if pd.notna(latest_sjc_sell) else "N/A", delta=format_delta(delta_sjc_sell), help=f"Giá bán SJC tại {SJC_TARGET_BRANCH}")
latest_sjc_buy = None; delta_sjc_buy = None
if not sjc_data_error and not sjc_hist.empty:
    try:
        latest_sjc_buy = get_scalar(sjc_hist.iloc[-1]['Giá Mua SJC (VND/cây)'])
        if len(sjc_hist) > 1:
             prev_sjc_buy = get_scalar(sjc_hist.iloc[-2]['Giá Mua SJC (VND/cây)'])
             if pd.notna(latest_sjc_buy) and isinstance(latest_sjc_buy, (int, float, complex)) and \
                pd.notna(prev_sjc_buy) and isinstance(prev_sjc_buy, (int, float, complex)):
                 delta_sjc_buy = latest_sjc_buy - prev_sjc_buy
    except Exception as e: print(f"Error processing SJC buy metric: {e}"); latest_sjc_buy = None; delta_sjc_buy = None
with col3: st.metric(label=f"SJC Mua ({latest_sjc_date_str})", value=f"{latest_sjc_buy:,.0f}" if pd.notna(latest_sjc_buy) else "N/A", delta=format_delta(delta_sjc_buy), help=f"Giá mua SJC tại {SJC_TARGET_BRANCH}")
latest_sjc_spread = None; delta_sjc_spread = None
if not sjc_data_error and not sjc_hist.empty and 'Chênh lệch Mua/Bán SJC' in sjc_hist.columns:
    try:
        latest_sjc_spread = get_scalar(sjc_hist.iloc[-1]['Chênh lệch Mua/Bán SJC'])
        if len(sjc_hist) > 1:
             prev_sjc_spread = get_scalar(sjc_hist.iloc[-2]['Chênh lệch Mua/Bán SJC'])
             if pd.notna(latest_sjc_spread) and isinstance(latest_sjc_spread, (int, float, complex)) and \
                pd.notna(prev_sjc_spread) and isinstance(prev_sjc_spread, (int, float, complex)):
                 delta_sjc_spread = latest_sjc_spread - prev_sjc_spread
    except Exception as e: print(f"Error processing SJC spread metric: {e}"); latest_sjc_spread = None; delta_sjc_spread = None
with col4: st.metric(label=f"Chênh lệch SJC ({latest_sjc_date_str})", value=f"{latest_sjc_spread:,.0f}" if pd.notna(latest_sjc_spread) else "N/A", delta=format_delta(delta_sjc_spread), help="Chênh lệch Mua/Bán SJC")

st.divider()

# --- Display World Gold Chart (USD/oz) ---
st.subheader(f"🌍 Giá Vàng Thế Giới (USD/oz) - Lịch sử ({final_label_hist})")
if world_data_error: st.info("Không có dữ liệu lịch sử giá vàng thế giới để hiển thị.")
elif not world_gold_usd_hist.empty:
    fig_world = px.line(world_gold_usd_hist, x='Timestamp', y='Giá TG (USD/oz)', labels={'Timestamp': 'Thời gian', 'Giá TG (USD/oz)': 'Giá (USD/oz)'})
    fig_world.update_traces(line_color='#0d6efd', hovertemplate="Ngày: %{x|%d/%m/%Y}<br>Giá: %{y:,.2f} USD<extra></extra>")
    fig_world.update_layout(hovermode="x unified", margin=dict(t=10, b=0, l=0, r=0))
    st.plotly_chart(fig_world, use_container_width=True)
else: st.info("Không có dữ liệu lịch sử giá vàng thế giới để hiển thị.")

# --- Display SJC Chart (Buy, Sell, Spread) ---
st.subheader(f"🇻🇳 Giá Vàng SJC (VND/cây) - Lịch sử ({final_label_hist})")
if sjc_data_error: st.info(f"Không có dữ liệu SJC để hiển thị.")
elif not sjc_hist.empty:
    plot_sjc_df = sjc_hist.melt(id_vars=['Timestamp'], value_vars=['Giá Mua SJC (VND/cây)', 'Giá Bán SJC (VND/cây)', 'Chênh lệch Mua/Bán SJC'], var_name='Loại Giá', value_name='Giá (VND/cây)')
    plot_sjc_df.dropna(subset=['Giá (VND/cây)'], inplace=True)
    if not plot_sjc_df.empty:
        fig_sjc = px.line(plot_sjc_df, x='Timestamp', y='Giá (VND/cây)', color='Loại Giá', labels={'Timestamp': 'Thời gian', 'Giá (VND/cây)': 'Giá (VND/cây)', 'Loại Giá': 'Loại Giá'}, markers=True, color_discrete_map={'Giá Mua SJC (VND/cây)': '#198754', 'Giá Bán SJC (VND/cây)': '#dc3545', 'Chênh lệch Mua/Bán SJC': '#6c757d'})
        fig_sjc.update_traces(hovertemplate="Ngày: %{x|%d/%m/%Y}<br>Giá: %{y:,.0f}<extra></extra>")
        fig_sjc.update_layout(hovermode="x unified", margin=dict(t=10, b=0, l=0, r=0), legend_title_text='Loại Giá')
        st.plotly_chart(fig_sjc, use_container_width=True)
    else: st.info(f"Không có dữ liệu SJC hợp lệ để vẽ biểu đồ.")
else: st.info(f"Không có dữ liệu SJC để hiển thị.")

# --- Removed Separate Spread Chart ---

# --- Display Raw Data (Optional Expander) ---
expander_title_parts = []
if not world_data_error: expander_title_parts.append("TG (USD)")
if not sjc_data_error: expander_title_parts.append("SJC Mua/Bán")

if expander_title_parts:
    with st.expander(f"🔍 Xem dữ liệu gốc ({' & '.join(expander_title_parts)})"):
        num_cols = len(expander_title_parts)
        cols = st.columns(num_cols)
        col_index = 0
        # Display raw world data (gold_hist_raw contains raw yfinance data)
        if not world_data_error and gold_hist_raw is not None:
            with cols[col_index]:
                st.caption("Vàng TG (USD/oz) - Raw");
                st.dataframe(gold_hist_raw[['Open', 'High', 'Low', 'Close', 'Volume']].style.format("{:,.2f}"), use_container_width=True, height=300)
            col_index += 1
        # Display SJC data
        if not sjc_data_error and not sjc_hist.empty:
             with cols[col_index]:
                st.caption(f"Vàng SJC (mỗi {SJC_FETCH_INTERVAL_DAYS} ngày)");
                st.dataframe(sjc_hist.set_index('Timestamp').style.format({'Giá Mua SJC (VND/cây)': '{:,.0f}', 'Giá Bán SJC (VND/cây)': '{:,.0f}', 'Chênh lệch Mua/Bán SJC': '{:,.0f}'}), use_container_width=True, height=300)

# --- Footer ---
st.divider()
col_left, col_right = st.columns([0.7, 0.3])
with col_left:
     st.markdown(f"<p class='footer-caption'>Nguồn: Yahoo Finance (TG Lịch sử), Trading Economics (TG Hiện tại), vnstock (SJC). Tải lúc: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}</p>", unsafe_allow_html=True) # Updated source
with col_right:
     st.markdown("<p class='footer-copyright'>Copyright ©LeQuyPhat</p>", unsafe_allow_html=True)
>>>>>>> f112762b5a73e4b7342d6681bd6fe9cbf5fbd89f
