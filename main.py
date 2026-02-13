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
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------
# 1. SETUP CƠ BẢN
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gold Trend Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈" 
)

# Phần này chỉnh CSS để giao diện nhìn đẹp hơn xíu
# Note: Mình dùng card style cho các metric nhìn cho nó nổi
st.markdown("""
<style>
    .main { font-family: 'Segoe UI', sans-serif; }
    
    /* Chỉnh cái card metric có viền vàng bên trái */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #DAA520;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    
    /* Chỉnh lại mấy cái Tab cho gọn */
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
    
    /* Phần Sidebar */
    .sidebar-logo-container {
        text-align: center; 
        margin-bottom: 25px;
        padding-bottom: 20px;
        border-bottom: 1px solid #e9ecef;
    }
    
    .sidebar-label {
        font-size: 11px;
        font-weight: 800;
        color: #495057;
        text-transform: uppercase;
        margin-bottom: 5px;
        display: block;
        letter-spacing: 0.5px;
    }
    
    /* Custom lại nút bấm cho nó đồng bộ màu */
    div.stButton > button[kind="primary"] {
        background-color: #2c3e50;
        border: none;
        width: 100%;
        border-radius: 6px;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #DAA520;
    }
    
    /* Badge ngày tháng nhỏ nhỏ ở sidebar */
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

# ---------------------------------------------------------
# 2. KHAI BÁO BIẾN
# ---------------------------------------------------------
# Mã chứng khoán trên Yahoo Finance
GOLD_TICKER = 'GC=F'    
USDVND_TICKER = 'VND=X' 
CACHE_TTL = 21600 # Cache 6 tiếng cho đỡ tốn request
SJC_Target = 'Hồ Chí Minh'
OUNCE_TO_TAEL = 1.205653 # Hệ số quy đổi ounce sang lượng

# Check xem có thư viện vnstock không, nếu không có thì báo lỗi sau
try:
    from vnstock.explorer.misc import sjc_gold_price
    VNSTOCK_AVAILABLE = True
except ImportError:
    VNSTOCK_AVAILABLE = False

# ---------------------------------------------------------
# 3. LẤY DỮ LIỆU (DATA FETCHING)
# ---------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL)
def fetch_financial_data(start_date, end_date):
    """
    Hàm này lấy dữ liệu từ Yahoo Finance.
    Note: Cần lấy dư ngày ra để tính mấy cái chỉ số MA, RSI đỡ bị NaN ở đầu.
    """
    try:
        buffer_date = start_date - timedelta(days=100) 
        tickers = f"{GOLD_TICKER} {USDVND_TICKER}"
        
        # Tải dữ liệu về
        data = yf.download(tickers, start=buffer_date, end=end_date + timedelta(days=1), progress=False, group_by='ticker')
        
        if data.empty:
            return pd.DataFrame(), "Không có dữ liệu"

        # Xử lý dữ liệu trả về (Yahoo hay đổi cấu trúc MultiIndex nên phải try-except)
        try:
            df_gold = data[GOLD_TICKER].copy()
            # Đổi tên cột cho dễ gọi
            df_gold = df_gold[['Open', 'High', 'Low', 'Close']].rename(columns={
                'Open': 'Gold_Open', 'High': 'Gold_High', 'Low': 'Gold_Low', 'Close': 'Gold_Close'
            })
            
            df_usd = data[USDVND_TICKER][['Close']].rename(columns={'Close': 'USDVND'})
        except KeyError:
            return pd.DataFrame(), "Lỗi cấu trúc dữ liệu Yahoo"

        # Gộp bảng vàng và bảng tỷ giá lại (Inner join)
        df_merge = df_gold.join(df_usd, how='inner')
        df_merge = df_merge.dropna()
        
        # Tính cột giá quy đổi sang VND (Triệu đồng/Lượng)
        # Công thức: (Giá USD * Tỷ giá * Hệ số Ounce->Lượng) / 1 Triệu
        df_merge['Gold_VND'] = (df_merge['Gold_Close'] * df_merge['USDVND'] * OUNCE_TO_TAEL) / 1e6
        
        # Lọc lại đúng khoảng ngày user chọn
        mask = (df_merge.index.date >= start_date) & (df_merge.index.date <= end_date)
        return df_merge.loc[mask], None

    except Exception as e:
        return pd.DataFrame(), str(e)

@st.cache_data(ttl=CACHE_TTL)
def fetch_sjc_data(start_date, end_date):
    """
    Lấy giá SJC. API này free nên hơi chậm, phải handle kỹ.
    """
    if not VNSTOCK_AVAILABLE:
        return pd.DataFrame(), "Thiếu thư viện vnstock"
    
    all_prices = []
    # Nếu user chọn khoảng thời gian dài quá thì mình lấy thưa ra (cách 5-15 ngày) cho nhanh
    days_diff = (end_date - start_date).days
    step = 1 if days_diff < 30 else (5 if days_diff < 180 else 15)
    
    current = start_date
    while current <= end_date:
        try:
            time.sleep(0.05) # Ngủ tí để không bị block IP
            df = sjc_gold_price(date=current.strftime("%Y-%m-%d"))
            if df is not None and not df.empty:
                row = df[df['branch'] == SJC_Target]
                if not row.empty:
                    # Clean data: bỏ dấu phẩy trong số
                    buy = float(str(row.iloc[0]['buy_price']).replace(',', ''))
                    sell = float(str(row.iloc[0]['sell_price']).replace(',', ''))
                    all_prices.append({'Date': current, 'SJC_Buy': buy, 'SJC_Sell': sell})
        except:
            pass # Lỗi thì bỏ qua ngày đó
        current += timedelta(days=step)
        
    if not all_prices:
        return pd.DataFrame(), "Không lấy được data SJC"
    
    df_sjc = pd.DataFrame(all_prices)
    df_sjc.set_index('Date', inplace=True)
    return df_sjc, None

def get_live_world_price():
    """
    Cào (Scrape) giá live từ TradingEconomics.
    """
    url = "https://tradingeconomics.com/commodities"
    headers = {"User-Agent": "Mozilla/5.0"} # Giả lập trình duyệt
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, 'html.parser')
            # Tìm trong table xem chỗ nào có chữ 'Gold'
            for table in soup.find_all('table'):
                for row in table.find_all('tr'):
                    if 'Gold' in row.text:
                        cols = row.find_all('td')
                        if len(cols) > 1:
                            return float(cols[1].text.strip().replace(',', ''))
    except:
        return None
    return None

# ---------------------------------------------------------
# 4. XỬ LÝ & TÍNH TOÁN CHỈ SỐ (FEATURE ENGINEERING)
# ---------------------------------------------------------

def process_data_with_currency(df, currency_mode):
    """
    Hàm này chuẩn bị data để vẽ biểu đồ tùy theo user chọn USD hay VND
    """
    df = df.copy()
    
    if currency_mode == 'VND':
        # Dùng giá quy đổi
        df['View_Price'] = df['Gold_VND']
        df['View_Open'] = (df['Gold_Open'] * df['USDVND'] * OUNCE_TO_TAEL) / 1e6
        df['View_High'] = (df['Gold_High'] * df['USDVND'] * OUNCE_TO_TAEL) / 1e6
        df['View_Low'] = (df['Gold_Low'] * df['USDVND'] * OUNCE_TO_TAEL) / 1e6
    else:
        # Dùng giá gốc USD
        df['View_Price'] = df['Gold_Close']
        df['View_Open'] = df['Gold_Open']
        df['View_High'] = df['Gold_High']
        df['View_Low'] = df['Gold_Low']
        
    return df

def add_technical_indicators(df):
    """
    Tính mấy cái chỉ số kỹ thuật (TA) để phân tích và chạy ML.
    Công thức tham khảo trên Investopedia.
    """
    df = df.copy()
    target = df['View_Price']
    
    # 1. Moving Averages (Đường trung bình)
    df['SMA_20'] = target.rolling(window=20).mean()
    df['SMA_50'] = target.rolling(window=50).mean()
    
    # 2. RSI (Relative Strength Index)
    delta = target.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. Bollinger Bands
    df['BB_Std'] = target.rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['SMA_20'] - (2 * df['BB_Std'])
    
    # 4. MACD
    ema12 = target.ewm(span=12, adjust=False).mean()
    ema26 = target.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 5. Returns & Drawdown (Dùng cho tab Hiệu suất)
    df['Daily_Return'] = target.pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod()
    df['Rolling_Max'] = df['Cumulative_Return'].cummax()
    df['Drawdown'] = df['Cumulative_Return'] / df['Rolling_Max'] - 1.0
    
    df.dropna(inplace=True) # Xóa mấy dòng đầu bị NaN do tính toán
    return df

# Hàm helper để style biểu đồ cho nhanh
def style_chart(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI", color="#2c3e50", size=11),
        xaxis=dict(showgrid=False, linecolor='#e0e0e0', tickformat='%d/%m'),
        yaxis=dict(showgrid=True, gridcolor='#f5f5f5', zeroline=False),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", bgcolor='rgba(255,255,255,0.8)')
    )
    return fig

# ---------------------------------------------------------
# 5. GIAO DIỆN CHÍNH (STREAMLIT UI)
# ---------------------------------------------------------

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div class="sidebar-logo-container">
            <h1 style="color: #DAA520; margin:0; font-size: 32px;">GOLD PRO</h1>
            <p style="font-size: 13px; font-weight: 600; color: #7f8c8d; letter-spacing: 1px;">MARKET INTELLIGENCE</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Filter 1: Cấu hình
    with st.container(border=True):
        st.markdown('<span class="sidebar-label">💱 ĐƠN VỊ HIỂN THỊ</span>', unsafe_allow_html=True)
        currency_mode = st.radio(
            "Chọn đơn vị tiền tệ",
            ["USD", "VND"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # Set nhãn đơn vị
        if currency_mode == 'USD':
            unit_label = "USD/oz"
            currency_symbol = "$"
        else:
            unit_label = "Triệu VNĐ/Lượng"
            currency_symbol = "₫"
        
        st.markdown("---")
        
        st.markdown('<span class="sidebar-label">🗓️ KHUNG THỜI GIAN</span>', unsafe_allow_html=True)
        range_option = st.selectbox(
            "Chọn thời gian", 
            ["3 Tháng", "6 Tháng", "1 Năm", "3 Năm", "5 Năm"], 
            index=2, 
            label_visibility="collapsed"
        )
        
        days_map = {"3 Tháng": 90, "6 Tháng": 180, "1 Năm": 365, "3 Năm": 1095, "5 Năm": 1825}
        end_input = datetime.now().date()
        start_input = end_input - timedelta(days=days_map[range_option])
        
        st.markdown(f"""
            <div style="display: flex; gap: 5px; margin-top: 10px;">
                <div class="date-badge">TỪ: {start_input.strftime('%d/%m/%Y')}</div>
                <div class="date-badge">ĐẾN: {end_input.strftime('%d/%m/%Y')}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Filter 2: Update giá
    with st.container(border=True):
        st.markdown('<span class="sidebar-label">⚡ DỮ LIỆU THỰC</span>', unsafe_allow_html=True)
        live_price = None
        if st.button("CẬP NHẬT GIÁ LIVE (USD)", use_container_width=True, type="primary"):
            with st.spinner("Đang kết nối..."):
                live_price = get_live_world_price()
                if live_price: st.toast(f"✅ Giá: ${live_price}")
                else: st.toast("⚠️ Lỗi kết nối")
    
    st.caption("Data Source: Yahoo Finance & Vnstock")

# --- LOAD DATA ---
with st.spinner("Đang tải dữ liệu..."):
    df_world, w_err = fetch_financial_data(start_input, end_input)
    df_sjc, s_err = fetch_sjc_data(start_input, end_input)

# Xử lý data để vẽ chart
if not df_world.empty:
    df_processed = process_data_with_currency(df_world, currency_mode)
    df_full = add_technical_indicators(df_processed)
else:
    df_full = pd.DataFrame()

# --- HEADER TRANG ---
st.markdown(f"## 📊 Bảng Tin Thị Trường Vàng ({currency_mode})")
st.markdown(f"**Cập nhật:** {datetime.now().strftime('%H:%M %d/%m/%Y')}")

# Tạo 4 Tab chính
tab1, tab2, tab3, tab4 = st.tabs(["TỔNG QUAN", "HIỆU SUẤT ĐẦU TƯ", "PHÂN TÍCH KỸ THUẬT", "DỰ BÁO AI"])

# ==========================================
# TAB 1: TỔNG QUAN (MARKET OVERVIEW)
# ==========================================
with tab1:
    st.markdown("### 📌 Chỉ Số Quan Trọng")
    c1, c2, c3, c4 = st.columns(4)
    
    if not df_full.empty:
        curr = df_full['View_Price'].iloc[-1]
        prev = df_full['View_Price'].iloc[-2]
        change = curr - prev
        
        c1.metric(f"Giá Vàng ({currency_mode})", f"{currency_symbol}{curr:,.2f}", f"{change:,.2f} {currency_mode}")
        c2.metric("Tỷ Giá USD/VND", f"{df_world['USDVND'].iloc[-1]:,.0f} ₫")
        
        # Tính giá tham chiếu ngược lại để so sánh
        if currency_mode == 'USD':
            vn_equiv = df_world['Gold_VND'].iloc[-1]
            c3.metric("Quy Đổi VND", f"{vn_equiv:,.2f} Tr/Lượng")
        else:
            usd_orig = df_world['Gold_Close'].iloc[-1]
            c3.metric("Giá Gốc USD", f"${usd_orig:,.2f}")
        
    sjc_val = df_sjc['SJC_Sell'].iloc[-1]/1e6 if not df_sjc.empty else 0
    c4.metric("SJC Bán Ra (VND)", f"{sjc_val:,.2f} Tr" if sjc_val else "N/A")
    st.divider()

    # Chart 1: Xu hướng chính
    st.markdown(f"##### 1. Xu Hướng Giá Vàng ({unit_label})")
    fig1 = px.area(df_full, y='View_Price', template="plotly_white")
    chart_color = '#DAA520' if currency_mode == 'USD' else '#2E86C1'
    fig1.update_traces(line_color=chart_color, fillcolor=f"rgba({int(chart_color[1:3],16)}, {int(chart_color[3:5],16)}, {int(chart_color[5:7],16)}, 0.1)")
    fig1.update_layout(yaxis_title=unit_label)
    st.plotly_chart(style_chart(fig1), use_container_width=True)

    col_chart_1, col_chart_2 = st.columns(2)
    
    # Chart 2: Nến Nhật
    with col_chart_1:
        st.markdown("##### 2. Chi Tiết Giá (Candlestick)")
        fig2 = go.Figure(data=[go.Candlestick(x=df_full.index,
                        open=df_full['View_Open'], high=df_full['View_High'],
                        low=df_full['View_Low'], close=df_full['View_Price'])])
        fig2.update_layout(xaxis_rangeslider_visible=False, template="plotly_white", yaxis_title=unit_label)
        st.plotly_chart(style_chart(fig2), use_container_width=True)

    # Chart 3: Tương quan
    with col_chart_2:
        st.markdown(f"##### 3. Tương Quan Vàng ({currency_mode}) & Tỷ Giá")
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(x=df_full.index, y=df_full['View_Price'], name=f"Vàng ({currency_mode})", line=dict(color=chart_color)), secondary_y=False)
        fig3.add_trace(go.Scatter(x=df_full.index, y=df_full['USDVND'], name="USDVND", line=dict(color="#34495e", dash='dot')), secondary_y=True)
        st.plotly_chart(style_chart(fig3), use_container_width=True)

    # Chart 4: Mùa vụ
    st.markdown("##### 4. Hiệu Suất Theo Tháng (Seasonality)")
    df_month = df_full.copy()
    df_month['Month'] = df_month.index.strftime('%m-%Y')
    monthly_ret = df_month.resample('M')['Daily_Return'].sum() * 100
    fig4 = px.bar(x=monthly_ret.index.strftime('%Y-%m'), y=monthly_ret.values, 
                  color=monthly_ret.values, color_continuous_scale="RdYlGn")
    fig4.update_layout(xaxis_title="Tháng", yaxis_title="Lợi nhuận (%)")
    st.plotly_chart(style_chart(fig4), use_container_width=True)

# ==========================================
# TAB 2: HIỆU SUẤT & RỦI RO (VOLATILITY & ROI)
# ==========================================
with tab2:
    st.markdown(f"### 💰 Phân Tích Lợi Nhuận ({currency_mode})")
    
    # Chart 1: ROI Simulator
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
        # Chart 2: Cumulative Return
        st.markdown("##### 1. Tăng Trưởng Tài Sản (%)")
        fig_cum = px.line(df_full, y='Cumulative_Return', template="plotly_white")
        fig_cum.add_hline(y=1, line_dash="dash", line_color="grey")
        fig_cum.update_traces(line_color="#27ae60", fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)')
        st.plotly_chart(style_chart(fig_cum), use_container_width=True)

    r1, r2 = st.columns(2)
    with r1:
        # Chart 3: Max Drawdown
        st.markdown("##### 2. Mức Độ Sụt Giảm (Max Drawdown)")
        fig_dd = px.area(df_full, y='Drawdown', template="plotly_white")
        fig_dd.update_traces(line_color="#c0392b", fillcolor="rgba(192, 57, 43, 0.3)")
        st.plotly_chart(style_chart(fig_dd), use_container_width=True)
    
    with r2:
        # Chart 4: Volatility
        st.markdown("##### 3. Biến Động Giá 30 Ngày")
        df_full['Vol_30'] = df_full['Daily_Return'].rolling(30).std()
        fig_vol = px.line(df_full, y='Vol_30', template="plotly_white")
        fig_vol.update_traces(line_color="#8e44ad")
        st.plotly_chart(style_chart(fig_vol), use_container_width=True)

    # Chart 5: Histogram
    st.markdown("##### 4. Phân Phối Lợi Nhuận Ngày")
    fig_hist = px.histogram(df_full, x='Daily_Return', nbins=50, color_discrete_sequence=['#34495e'])
    st.plotly_chart(style_chart(fig_hist), use_container_width=True)

# ==========================================
# TAB 3: PHÂN TÍCH KỸ THUẬT (TECHNICAL ANALYSIS)
# ==========================================
with tab3:
    st.markdown(f"### 🛠️ Chỉ Báo Kỹ Thuật ({unit_label})")
    
    t1, t2 = st.columns(2)
    
    # Chart 1: Bollinger Bands
    with t1:
        st.markdown("##### 1. Bollinger Bands (20, 2)")
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df_full.index, y=df_full['BB_Upper'], line=dict(color='gray', width=1), name='Upper'))
        fig_bb.add_trace(go.Scatter(x=df_full.index, y=df_full['BB_Lower'], line=dict(color='gray', width=1), fill='tonexty', name='Lower'))
        fig_bb.add_trace(go.Scatter(x=df_full.index, y=df_full['View_Price'], line=dict(color=chart_color, width=2), name='Price'))
        fig_bb.update_layout(showlegend=False, yaxis_title=unit_label)
        st.plotly_chart(style_chart(fig_bb), use_container_width=True)

    # Chart 2: SMA Crossover
    with t2:
        st.markdown("##### 2. SMA Crossover (Ngắn vs Dài)")
        fig_sma = go.Figure()
        fig_sma.add_trace(go.Scatter(x=df_full.index, y=df_full['SMA_20'], line=dict(color='#2980b9'), name='SMA 20'))
        fig_sma.add_trace(go.Scatter(x=df_full.index, y=df_full['SMA_50'], line=dict(color='#e74c3c'), name='SMA 50'))
        fig_sma.add_trace(go.Scatter(x=df_full.index, y=df_full['View_Price'], line=dict(color=chart_color, width=1), opacity=0.5, name='Price'))
        st.plotly_chart(style_chart(fig_sma), use_container_width=True)

    t3, t4 = st.columns(2)
    
    # Chart 3: RSI
    with t3:
        st.markdown("##### 3. RSI (Sức Mạnh Tương Đối)")
        fig_rsi = px.line(df_full, y='RSI')
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_traces(line_color='#8e44ad')
        fig_rsi.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(style_chart(fig_rsi), use_container_width=True)

    # Chart 4: MACD
    with t4:
        st.markdown("##### 4. MACD (Động Lượng)")
        fig_macd = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        fig_macd.add_trace(go.Scatter(x=df_full.index, y=df_full['MACD'], line=dict(color='#2980b9'), name='MACD'), row=1, col=1)
        fig_macd.add_trace(go.Scatter(x=df_full.index, y=df_full['MACD_Signal'], line=dict(color='#e67e22'), name='Signal'), row=1, col=1)
        colors = np.where(df_full['MACD_Hist'] < 0, '#e74c3c', '#27ae60')
        fig_macd.add_trace(go.Bar(x=df_full.index, y=df_full['MACD_Hist'], marker_color=colors, name='Hist'), row=2, col=1)
        fig_macd.update_layout(showlegend=False)
        st.plotly_chart(style_chart(fig_macd), use_container_width=True)

# ==========================================
# TAB 4: AI FORECAST (DỰ BÁO)
# ==========================================
with tab4:
    st.markdown(f"### 🤖 Phòng Thí Nghiệm AI ({currency_mode})")
    
    # Prepare Data
    df_ml = df_full.copy()
    df_ml['Lag1'] = df_ml['View_Price'].shift(1)
    df_ml['Lag2'] = df_ml['View_Price'].shift(2)
    df_ml['Target'] = df_ml['View_Price'].shift(-1)
    df_ml.dropna(inplace=True)
    
    feats = ['Lag1', 'Lag2', 'SMA_20', 'RSI', 'MACD', 'USDVND']
    X = df_ml[feats]
    y = df_ml['Target']
    
    # Train Model (Random Forest)
    split = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Chart 1: Kết quả dự báo
    c_ai1, c_ai2 = st.columns([1, 2])
    with c_ai1:
        next_pred = rf.predict(df_ml.iloc[[-1]][feats])[0]
        st.markdown(f"""
        <div style="background-color: #2c3e50; color: white; padding: 20px; border-radius: 6px; text-align: center;">
            <p style="margin:0; font-size: 12px; opacity: 0.8;">DỰ BÁO NGÀY MAI (T+1)</p>
            <h2 style="margin: 5px 0; color: #f1c40f;">{currency_symbol}{next_pred:,.2f}</h2>
            <hr style="border-color: rgba(255,255,255,0.2);">
            <div style="font-size: 12px;">Sai số (MAE): {currency_symbol}{mean_absolute_error(y_test, y_pred):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c_ai2:
        # Chart 2: Feature Importance
        imp = pd.DataFrame({'Feat': feats, 'Imp': rf.feature_importances_}).sort_values('Imp')
        fig_imp = px.bar(imp, x='Imp', y='Feat', orientation='h', title="1. Yếu Tố Ảnh Hưởng Nhất", color_discrete_sequence=['#16a085'])
        st.plotly_chart(style_chart(fig_imp), use_container_width=True)

    a1, a2 = st.columns(2)
    
    # Chart 3: Backtest
    with a1:
        st.markdown("##### 2. Kiểm Thử (Backtest)")
        df_res = pd.DataFrame({'Thực tế': y_test, 'Dự báo': y_pred}, index=y_test.index)
        fig_back = go.Figure()
        fig_back.add_trace(go.Scatter(x=df_res.index, y=df_res['Thực tế'], name='Thực tế', line=dict(color='#bdc3c7')))
        fig_back.add_trace(go.Scatter(x=df_res.index, y=df_res['Dự báo'], name='AI', line=dict(color='#f39c12', dash='dot')))
        st.plotly_chart(style_chart(fig_back), use_container_width=True)

    # Chart 4: Scatter Plot
    with a2:
        st.markdown("##### 3. Độ Tuyến Tính (Linearity)")
        fig_scat = px.scatter(x=y_test, y=y_pred, labels={'x': 'Giá Thực Tế', 'y': 'Giá Dự Báo'})
        fig_scat.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(color="red", dash="dash"))
        st.plotly_chart(style_chart(fig_scat), use_container_width=True)
        
    # Chart 5: Residuals
    st.markdown("##### 4. Phân Phối Sai Số (Residuals)")
    residuals = y_test - y_pred
    fig_res = px.histogram(x=residuals, nbins=50, color_discrete_sequence=['#c0392b'], labels={'x': f'Mức độ sai lệch ({currency_symbol})'})
    st.plotly_chart(style_chart(fig_res), use_container_width=True)

st.markdown("---")
st.caption("Data Analyst Project | Powered by Streamlit & Yahoo Finance")