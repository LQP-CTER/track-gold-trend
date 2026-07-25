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
import json
from bs4 import BeautifulSoup
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from narrative import generate_ai_insight

# Security: Get API Key from Streamlit secrets or environment
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, KeyError):
    GROQ_API_KEY = ""

st.set_page_config(
    page_title="Gold Trend Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Icon-free Minimalist Enterprise Fintech Style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp {
    background-color: #F8F9FA !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #F1F3F5 !important;
    border-right: 1px solid rgba(0, 0, 0, 0.06) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }

.brand-header {
    padding: 30px 20px 20px;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
}
.brand-header h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 25px;
    font-weight: 600;
    color: #0F172A;
    margin: 0;
    letter-spacing: -0.3px;
}
.brand-header p {
    font-size: 10px;
    color: #888888;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 4px 0 0;
}

.sidebar-label {
    font-size: 10px !important;
    color: #94A3B8 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    display: block !important;
    margin-bottom: 8px !important;
}

.date-text {
    color: #475569;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 8px;
}

[data-testid="stSidebar"] div.stButton > button {
    background: #0F172A !important;
    color: #FFFFFF !important;
    border: 1px solid #0F172A !important;
    border-radius: 3px !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
[data-testid="stSidebar"] div.stButton > button:hover {
    background: #1E293B !important;
    border-color: #1E293B !important;
    transform: translateY(-1px) !important;
}

/* MAIN TEXT */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { 
    font-family: 'Outfit', sans-serif !important;
    color: #0F172A !important; 
    font-weight: 600 !important;
    letter-spacing: -0.5px !important;
}

p:not(.quant-terminal *):not(.ai-insight-container *),
span:not(.quant-terminal *):not(.ai-insight-container *):not(.sidebar-label),
label:not(.quant-terminal *):not(.ai-insight-container *) {
    color: #475569 !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    padding: 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}
.stTabs [data-baseweb="tab"] {
    height: 48px;
    background: transparent;
    border-radius: 0;
    color: #64748B;
    font-weight: 500;
    font-size: 14px;
    border: none;
    border-bottom: 2px solid transparent;
    margin-right: 35px;
    padding: 0 4px;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #0F172A; }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #0F172A;
    border-bottom: 2px solid #0F172A;
}

/* HERO HEADER */
.hero-header {
    padding: 10px 0 25px;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    margin-bottom: 30px;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 30px;
    font-weight: 600;
    color: #0F172A;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.hero-subtitle { 
    font-size: 12px; 
    color: #64748B; 
    margin: 0;
    font-family: 'JetBrains Mono', monospace;
}

/* METRIC CARDS */
.metric-container {
    padding: 22px 24px;
    background: rgba(255, 255, 255, 0.75) !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
    border-radius: 6px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    box-shadow: 0 2px 12px rgba(9, 30, 66, 0.03) !important;
    height: 100%;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.metric-container:hover {
    border-color: rgba(0, 0, 0, 0.12) !important;
    box-shadow: 0 8px 30px rgba(9, 30, 66, 0.08) !important;
    transform: translateY(-2px) !important;
}
.metric-label {
    font-size: 11px;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px;
    font-weight: 500;
    color: #0F172A;
    margin-bottom: 8px;
    line-height: 1;
}
.metric-sub {
    font-size: 12.5px;
    font-family: 'JetBrains Mono', monospace;
}
.text-up { color: #059669; }
.text-down { color: #DC2626; }
.text-neutral { color: #64748B; }

/* SJC SPREAD CARD */
.spread-card {
    padding: 20px 24px;
    background: #FFFFFF !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    border-left: 4px solid #0F172A !important;
    border-radius: 6px !important;
    margin-bottom: 25px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
}
.spread-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.badge-normal { background: #E6F4EA; color: #137333; }
.badge-high { background: #FEF7E0; color: #B06000; }
.badge-very-high { background: #FCE8E6; color: #C5221F; }

/* SECTION DIVIDER */
.section-divider {
    height: 1px;
    background: rgba(0, 0, 0, 0.05);
    margin: 30px 0;
    border: none;
}

/* AI CARD */
.ai-card {
    background: rgba(255, 255, 255, 0.75) !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
    border-radius: 6px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    box-shadow: 0 2px 12px rgba(9, 30, 66, 0.03) !important;
    padding: 24px;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.ai-card:hover {
    border-color: rgba(0, 0, 0, 0.12) !important;
    box-shadow: 0 8px 30px rgba(9, 30, 66, 0.08) !important;
    transform: translateY(-2px) !important;
}
.ai-header {
    font-size: 11px;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 20px;
}
.ai-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 34px;
    color: #0F172A;
    margin-bottom: 10px;
}
.ai-stat {
    font-size: 12.5px;
    color: #64748B;
    font-family: 'JetBrains Mono', monospace;
}

/* AI INSIGHT CARDS */
.ai-insight-container {
    background: rgba(255, 255, 255, 0.8) !important;
    border: 1px solid rgba(0, 0, 0, 0.05) !important;
    border-left: 3px solid #D97706 !important;
    border-radius: 0 6px 6px 0 !important;
    padding: 16px 20px !important;
    margin-top: -15px !important;
    margin-bottom: 25px !important;
    box-shadow: 0 2px 12px rgba(9, 30, 66, 0.02) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}
.ai-insight-title {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    color: #64748B !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    font-weight: 600 !important;
    margin-bottom: 6px !important;
}
.ai-insight-body, .ai-insight-body p, .ai-insight-body span {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    color: #1E293B !important;
    line-height: 1.6 !important;
    font-weight: 400 !important;
    margin: 0 !important;
}
.ai-insight-body strong {
    color: #0F172A !important;
    font-weight: 600 !important;
}

/* QUANT TERMINAL */
.quant-terminal {
    background-color: #0F172A !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 25px 30px !important;
    border-radius: 8px !important;
    box-shadow: 0 10px 45px rgba(0, 0, 0, 0.15) !important;
    font-family: 'JetBrains Mono', monospace !important;
    position: relative !important;
}
.terminal-header {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding-bottom: 12px !important;
    margin-bottom: 20px !important;
}
.dot-red, .dot-yellow, .dot-green {
    display: inline-block !important;
    width: 8px !important;
    height: 8px !important;
    border-radius: 50% !important;
}
.dot-red { background-color: #EF4444 !important; }
.dot-yellow { background-color: #F59E0B !important; }
.dot-green { background-color: #10B981 !important; }
.terminal-title {
    color: #94A3B8 !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    margin-left: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.status-active {
    color: #10B981 !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.terminal-body, .terminal-body p, .terminal-body span, .terminal-body br {
    color: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13.5px !important;
    line-height: 1.8 !important;
}
.terminal-body strong {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* FOOTER */
.footer {
    text-align: left;
    padding: 20px 0;
    margin-top: 40px;
    border-top: 1px solid rgba(0, 0, 0, 0.05);
    color: #94A3B8;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

GOLD_TICKER = 'GC=F'
USDVND_TICKER = 'VND=X'
CACHE_TTL = 21600
SJC_Target = 'Hồ Chí Minh'
OUNCE_TO_TAEL = 1.205653

try:
    from vnstock.explorer.misc import sjc_gold_price
    VNSTOCK_AVAILABLE = True
except ImportError:
    VNSTOCK_AVAILABLE = False

def t(vi_text, en_text):
    if st.session_state.get('lang', 'Tiếng Việt') == 'English':
        return en_text
    return vi_text

def render_chart_insight(data_dict, chart_name):
    """Helper to generate and display AI insight per chart (Light Glassmorphism Theme)"""
    lang_code = 'VN' if st.session_state.get('lang', 'Tiếng Việt') == 'Tiếng Việt' else 'EN'
    spinner_text = "Phân tích AI..." if lang_code == 'VN' else "AI Analyzing..."
    with st.spinner(spinner_text):
        insight = generate_ai_insight(
            json.dumps(data_dict, ensure_ascii=False),
            dashboard_section=chart_name,
            lang=lang_code
        )
    st.markdown(f"""
    <div class="ai-insight-container">
        <div class="ai-insight-title">AI Market Insight</div>
        <div class="ai-insight-body">{insight}</div>
    </div>
    """, unsafe_allow_html=True)

def get_groq_analysis(current_price, predicted_price, rsi, macd, sma20, sma50, currency):
    lang_code = st.session_state.get('lang', 'Tiếng Việt')
    is_en = (lang_code == 'English')

    if not GROQ_API_KEY:
        return t("Lỗi: Chưa cấu hình GROQ_API_KEY trong secrets.", "Error: GROQ_API_KEY is not configured in secrets.")
        
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    if is_en:
        sys_prompt = "You are a professional financial analyst. Write your analysis in English. DO NOT USE ANY EMOJIS OR ICONS."
        prompt = f"""
        You are a Senior Quantitative Financial Analyst.
        Below is the latest technical data for the Gold market (Currency: {currency}):
        - Current Price: {current_price:,.2f}
        - Tomorrow Forecast (Quantitative Model): {predicted_price:,.2f}
        - RSI (14 days): {rsi:.2f}
        - MACD: {macd:.2f}
        - SMA 20: {sma20:,.2f}
        - SMA 50: {sma50:,.2f}
        
        Write a structured summary in 3-4 bullet points:
        Start with plain text tags:
        [SIGNAL: POSITIVE / NEUTRAL / CAUTIOUS]
        [RISK LEVEL: LOW / MEDIUM / HIGH]

        Provide clear, objective analysis based on the numbers above. DO NOT USE ANY EMOJIS OR ICONS AT ALL.
        """
    else:
        sys_prompt = "You are a professional financial analyst. Write your analysis in Vietnamese. DO NOT USE ANY EMOJIS OR ICONS."
        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính định lượng chuyên nghiệp.
        Dưới đây là các dữ liệu kỹ thuật mới nhất của thị trường Vàng (đơn vị: {currency}):
        - Giá hiện tại: {current_price:,.2f}
        - Giá dự báo ngày mai (Mô hình định lượng): {predicted_price:,.2f}
        - Chỉ báo RSI (14 ngày): {rsi:.2f}
        - Chỉ báo MACD: {macd:.2f}
        - Đường trung bình SMA 20 ngày: {sma20:,.2f}
        - Đường trung bình SMA 50 ngày: {sma50:,.2f}
        
        Viết một bản tóm tắt phân tích súc tích (3-4 dòng gạch đầu dòng):
        Mở đầu bằng các tag văn bản:
        [TÍN HIỆU: TÍCH CỰC / TRUNG LẬP / THẬN TRỌNG]
        [MỨC RỦI RO: THẤP / TRUNG BÌNH / CAO]

        TUÂN THỦ TUYỆT ĐỐI KHÔNG DÙNG BẤT KỲ ICON HOẶC EMOJI NÀO.
        """
    
    payload = {
        "model": "qwen/qwen3-32b",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return t(f"Lỗi khi kết nối Groq API: {str(e)}", f"Error connecting to Groq API: {str(e)}")

@st.cache_data(ttl=CACHE_TTL)
def fetch_financial_data(start_date, end_date):
    try:
        buffer_date = start_date - timedelta(days=100) 
        tickers = f"{GOLD_TICKER} {USDVND_TICKER}"
        data = yf.download(tickers, start=buffer_date, end=end_date + timedelta(days=1), progress=False, group_by='ticker')
        
        if data.empty:
            return pd.DataFrame(), "No data returned."

        try:
            df_gold = data[GOLD_TICKER].copy()
            df_gold = df_gold[['Open', 'High', 'Low', 'Close']].rename(columns={
                'Open': 'Gold_Open', 'High': 'Gold_High', 'Low': 'Gold_Low', 'Close': 'Gold_Close'
            })
            df_usd = data[USDVND_TICKER][['Close']].rename(columns={'Close': 'USDVND'})
        except KeyError:
            return pd.DataFrame(), "Data structure error from Yahoo Finance."

        df_merge = df_gold.join(df_usd, how='outer')
        df_merge['USDVND'] = df_merge['USDVND'].ffill().bfill()
        df_merge = df_merge.dropna(subset=['Gold_Close'])
        
        df_merge['Gold_VND'] = (df_merge['Gold_Close'] * df_merge['USDVND'] * OUNCE_TO_TAEL) / 1e6
        
        mask = (df_merge.index.date >= start_date) & (df_merge.index.date <= end_date)
        return df_merge.loc[mask], None

    except Exception as e:
        return pd.DataFrame(), str(e)

import os
import re
SJC_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sjc_history_cache.csv')

def normalize_sjc_price(val):
    """Normalize any price value to raw VND float (e.g., 137,500,000.0)"""
    if val is None or pd.isna(val):
        return 0.0
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '').strip()
    try:
        f_val = float(val)
        if f_val <= 0:
            return 0.0
        if f_val < 1000: # E.g., 137.5 (Million VND)
            return f_val * 1e6
        elif f_val < 10000000: # E.g., 13.750.000 (VND per chi)
            return f_val * 10
        return f_val
    except Exception:
        return 0.0

def load_sjc_cache():
    if os.path.exists(SJC_CACHE_FILE):
        try:
            df = pd.read_csv(SJC_CACHE_FILE)
            if 'index' in df.columns and 'Date' not in df.columns:
                df.rename(columns={'index': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            df.set_index('Date', inplace=True)
            df['SJC_Buy'] = df['SJC_Buy'].apply(normalize_sjc_price)
            df['SJC_Sell'] = df['SJC_Sell'].apply(normalize_sjc_price)
            return df
        except Exception:
            return pd.DataFrame(columns=['SJC_Buy', 'SJC_Sell'])
    return pd.DataFrame(columns=['SJC_Buy', 'SJC_Sell'])

def save_sjc_cache(df_cache):
    try:
        df_cache = df_cache.copy()
        df_cache.index.name = 'Date'
        df_to_save = df_cache.reset_index()
        df_to_save.to_csv(SJC_CACHE_FILE, index=False)
    except Exception:
        pass

def fetch_sjc_tier1_vangtoday():
    """Tier 1: High speed Vang.today API"""
    try:
        url = "https://www.vang.today/api/prices"
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3)
        if r.status_code == 200:
            data = r.json()
            vngsjc = data.get('prices', {}).get('VNGSJC', {})
            buy = normalize_sjc_price(vngsjc.get('buy', 0))
            sell = normalize_sjc_price(vngsjc.get('sell', 0))
            if buy > 0 and sell > 0:
                return buy, sell
    except Exception:
        pass
    return None, None

def fetch_sjc_tier2_webgia_live():
    """Tier 2A: WebGia HTML Live Scraper"""
    try:
        url = "https://webgia.com/gia-vang/sjc/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r = requests.get(url, headers=headers, timeout=3)
        soup = BeautifulSoup(r.text, 'html.parser')
        tables = soup.find_all('table')
        if tables:
            for row in tables[0].find_all('tr'):
                cols = [c.text.strip() for c in row.find_all(['td', 'th'])]
                if len(cols) >= 4 and 'Hồ Chí Minh' in cols[0] and '1L' in cols[1]:
                    buy = normalize_sjc_price(cols[2])
                    sell = normalize_sjc_price(cols[3])
                    if buy > 0 and sell > 0:
                        return buy, sell
    except Exception:
        pass
    return None, None

def fetch_sjc_tier2_webgia_history():
    """Tier 2B: WebGia Highcharts Javascript regex history scraper"""
    history_dict = {}
    try:
        url = "https://webgia.com/gia-vang/sjc/bieu-do-1-thang.html"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r = requests.get(url, headers=headers, timeout=3)
        pairs = re.findall(r'\[(\d{12,13}),\s*([\d\.]+)\]', r.text)
        for ts_str, price_m in pairs:
            dt = datetime.fromtimestamp(int(ts_str) / 1000).date()
            price_vnd = normalize_sjc_price(price_m)
            if price_vnd > 0:
                history_dict[dt] = {'SJC_Buy': price_vnd * 0.97, 'SJC_Sell': price_vnd}
    except Exception:
        pass
    return history_dict

def fetch_sjc_tier3_vnstock(target_date):
    """Tier 3: Vnstock library fallback for historical date"""
    if not VNSTOCK_AVAILABLE:
        return None, None
    try:
        from vnstock.explorer.misc import sjc_gold_price
        df = sjc_gold_price(date=target_date.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            row = df[df['branch'] == SJC_Target]
            if not row.empty:
                buy = normalize_sjc_price(row.iloc[0]['buy_price'])
                sell = normalize_sjc_price(row.iloc[0]['sell_price'])
                if buy > 0 and sell > 0:
                    return buy, sell
    except Exception:
        pass
    return None, None

def estimate_sjc_failsafe(target_date):
    """Tier 4 Failsafe Estimator using World Gold + USD/VND parity"""
    try:
        ticker_g = yf.Ticker(GOLD_TICKER)
        ticker_u = yf.Ticker(USDVND_TICKER)
        g_price = float(ticker_g.fast_info['lastPrice'])
        u_price = float(ticker_u.fast_info['lastPrice'])
        est_sell = (g_price * u_price * OUNCE_TO_TAEL) * 1.12
        est_buy = est_sell * 0.97
        return normalize_sjc_price(est_buy), normalize_sjc_price(est_sell)
    except Exception:
        return 137000000.0, 141000000.0

@st.cache_data(ttl=CACHE_TTL)
def fetch_sjc_data(start_date, end_date):
    df_cache = load_sjc_cache()
    today = datetime.now().date()
    
    live_buy, live_sell = fetch_sjc_tier1_vangtoday()
    if not live_buy or not live_sell:
        live_buy, live_sell = fetch_sjc_tier2_webgia_live()
    if not live_buy or not live_sell:
        live_buy, live_sell = estimate_sjc_failsafe(today)

    if live_buy and live_sell:
        df_today = pd.DataFrame({'Date': [today], 'SJC_Buy': [live_buy], 'SJC_Sell': [live_sell]})
        df_today.set_index('Date', inplace=True)
        df_cache = pd.concat([df_cache, df_today])
        df_cache = df_cache[~df_cache.index.duplicated(keep='last')].sort_index()

    days_diff = (end_date - start_date).days
    step = 2 if days_diff < 30 else (7 if days_diff < 180 else (20 if days_diff < 365 else 30))
    target_dates = []
    current = start_date
    while current <= end_date:
        target_dates.append(current)
        current += timedelta(days=step)

    missing_dates = [d for d in target_dates if d not in df_cache.index]

    if missing_dates:
        webgia_hist = fetch_sjc_tier2_webgia_history()
        new_data = []
        for d in missing_dates:
            if d in webgia_hist:
                new_data.append({'Date': d, 'SJC_Buy': webgia_hist[d]['SJC_Buy'], 'SJC_Sell': webgia_hist[d]['SJC_Sell']})
            else:
                b, s = fetch_sjc_tier3_vnstock(d)
                if b and s:
                    new_data.append({'Date': d, 'SJC_Buy': b, 'SJC_Sell': s})

        if new_data:
            df_new = pd.DataFrame(new_data)
            df_new.set_index('Date', inplace=True)
            df_cache = pd.concat([df_cache, df_new])
            df_cache = df_cache[~df_cache.index.duplicated(keep='last')].sort_index()

    save_sjc_cache(df_cache)

    mask = (df_cache.index >= start_date) & (df_cache.index <= end_date)
    df_result = df_cache.loc[mask].copy()

    if df_result.empty:
        est_b, est_s = estimate_sjc_failsafe(today)
        df_result = pd.DataFrame({'Date': [start_date, end_date], 'SJC_Buy': [est_b, est_b], 'SJC_Sell': [est_s, est_s]})
        df_result.set_index('Date', inplace=True)

    return df_result, None

def get_live_world_price():
    try:
        ticker = yf.Ticker(GOLD_TICKER)
        return float(ticker.fast_info['lastPrice'])
    except Exception:
        try:
            hist = ticker.history(period='1d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception:
            pass
    return None

def process_data_with_currency(df, currency_mode):
    df = df.copy()
    if currency_mode == 'VND':
        df['View_Price'] = df['Gold_VND']
        df['View_Open'] = (df['Gold_Open'] * df['USDVND'] * OUNCE_TO_TAEL) / 1e6
        df['View_High'] = (df['Gold_High'] * df['USDVND'] * OUNCE_TO_TAEL) / 1e6
        df['View_Low'] = (df['Gold_Low'] * df['USDVND'] * OUNCE_TO_TAEL) / 1e6
    else:
        df['View_Price'] = df['Gold_Close']
        df['View_Open'] = df['Gold_Open']
        df['View_High'] = df['Gold_High']
        df['View_Low'] = df['Gold_Low']
    return df

def add_technical_indicators(df):
    df = df.copy()
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
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#64748B", size=11),
        xaxis=dict(
            showgrid=False,
            linecolor='rgba(0,0,0,0.06)',
            tickfont=dict(color='#64748B', family="JetBrains Mono"),
            tickformat='%d/%m',
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.04)',
            zeroline=False,
            linecolor='rgba(0,0,0,0.06)',
            tickfont=dict(color='#64748B', family="JetBrains Mono"),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="rgba(0,0,0,0.08)",
            font=dict(color="#0F172A", family="JetBrains Mono", size=11),
        ),
        legend=dict(
            orientation="h", y=1.05, x=1, xanchor="right",
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#475569', family="Inter"),
        )
    )
    return fig

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown(f"""
        <div class="brand-header">
            <h1>GoldTrend</h1>
            <p>{t('Phân Tích Định Lượng', 'Quantitative Analytics')}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<span class="sidebar-label">{t("NGÔN NGỮ", "LANGUAGE")}</span>', unsafe_allow_html=True)
    lang_selection = st.radio(
        "Language",
        ["Tiếng Việt", "English"],
        index=0 if st.session_state.get('lang', 'Tiếng Việt') == 'Tiếng Việt' else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    if lang_selection != st.session_state.get('lang'):
        st.session_state['lang'] = lang_selection
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container():
        st.markdown(f'<span class="sidebar-label">{t("CHẾ ĐỘ XEM", "VIEW MODE")}</span>', unsafe_allow_html=True)
        view_mode = st.radio(
            "View Mode",
            [t("Phổ thông (Basic)", "Basic"), t("Chuyên sâu (Pro / Quant)", "Pro / Quant")],
            index=0 if st.session_state.get('view_mode', 'Basic') in ['Basic', 'Phổ thông (Basic)'] else 1,
            label_visibility="collapsed"
        )
        st.session_state['view_mode'] = view_mode

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<span class="sidebar-label">{t("LOẠI TIỀN TỆ", "CURRENCY")}</span>', unsafe_allow_html=True)
        currency_mode = st.radio(
            "Currency",
            ["USD", "VND"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if currency_mode == 'USD':
            unit_label = "USD/oz"
            currency_symbol = "$"
        else:
            unit_label = t("Tr.VNĐ / Lượng", "VND (Mil) / Tael")
            currency_symbol = "₫"
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f'<span class="sidebar-label">{t("KHUNG THỜI GIAN", "TIMEFRAME")}</span>', unsafe_allow_html=True)
        time_opts = [t("3 Tháng", "3 Months"), t("6 Tháng", "6 Months"), t("1 Năm", "1 Year"), t("3 Năm", "3 Years"), t("5 Năm", "5 Years")]
        range_option = st.selectbox(
            "Timeframe", 
            time_opts, 
            index=2, 
            label_visibility="collapsed"
        )
        
        days_map = {t("3 Tháng", "3 Months"): 90, t("6 Tháng", "6 Months"): 180, t("1 Năm", "1 Year"): 365, t("3 Năm", "3 Years"): 1095, t("5 Năm", "5 Years"): 1825}
        end_input = datetime.now().date()
        start_input = end_input - timedelta(days=days_map[range_option])
        
        st.markdown(f"""
            <div class="date-text">
                {start_input.strftime('%Y-%m-%d')} &mdash; {end_input.strftime('%Y-%m-%d')}
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        st.markdown(f'<span class="sidebar-label">{t("DỮ LIỆU LIVE", "REAL-TIME DATA")}</span>', unsafe_allow_html=True)
        live_price = None
        if st.button(t("CẬP NHẬT GIÁ USD LIVE", "FETCH LIVE USD PRICE"), use_container_width=True):
            with st.spinner(t("Đang kết nối...", "Connecting...")):
                live_price = get_live_world_price()
                if live_price: 
                    st.success(f"{t('Mới nhất', 'Latest')}: ${live_price}")
                else: 
                    st.error(t("Lỗi kết nối.", "Connection failed."))
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption(f"{t('Nguồn', 'Source')}: Yahoo Finance & Vnstock")

# --- LOAD DATA ---
with st.spinner(t("Đang tải dữ liệu thị trường...", "Loading market data...")):
    df_world, w_err = fetch_financial_data(start_input, end_input)
    df_sjc, s_err = fetch_sjc_data(start_input, end_input)

# --- PROCESS ---
if not df_world.empty:
    df_processed = process_data_with_currency(df_world, currency_mode)
    df_full = add_technical_indicators(df_processed)
else:
    df_full = pd.DataFrame()

# --- HEADER ---
is_basic_mode = "Basic" in view_mode or "Phổ thông" in view_mode
header_mode_text = t("Chế độ Phổ thông", "Basic Mode") if is_basic_mode else t("Chế độ Chuyên sâu", "Pro/Quant Mode")

st.markdown(f"""
<div class="hero-header">
    <div class="hero-title">{t("Phân tích Thị trường Vàng", "Gold Market Analytics")} / {currency_mode}</div>
    <div class="hero-subtitle">[{header_mode_text}] &middot; {t("Giờ hệ thống", "System Time")}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>
""", unsafe_allow_html=True)

if w_err:
    st.error(f"{t('Lỗi Dữ liệu Toàn cầu', 'Global Data Error')}: {w_err}")
if s_err:
    st.warning(f"{t('Cảnh báo Dữ liệu Nội địa', 'Local Data Warning')}: {s_err}. {t('Các chỉ số SJC có thể không có.', 'SJC metrics may be N/A.')}")


# --- TABS DEFINITION ---
if is_basic_mode:
    tab1, tab2, tab3 = st.tabs([
        t("TỔNG QUAN THỊ TRƯỜNG", "MARKET OVERVIEW"),
        t("MÔ PHỎNG LỢI NHUẬN", "P&L SIMULATOR"),
        t("BÁO CÁO AI", "EXECUTIVE AI REPORT")
    ])
else:
    tab1, tab2, tab3, tab4 = st.tabs([
        t("TỔNG QUAN", "OVERVIEW"),
        t("HIỆU SUẤT", "PERFORMANCE"),
        t("KỸ THUẬT", "TECHNICALS"),
        t("MÔ HÌNH AI", "AI MODELS")
    ])

# ==================== TAB 1: OVERVIEW ====================
with tab1:
    if not df_full.empty:
        curr = df_full['View_Price'].iloc[-1]
        prev = df_full['View_Price'].iloc[-2]
        change = curr - prev
        change_pct = (change / prev) * 100
        
        delta_class = "text-up" if change >= 0 else "text-down"
        delta_sign = "+" if change >= 0 else ""
        usdvnd = df_world['USDVND'].iloc[-1]

        if currency_mode == 'USD':
            ref_label = t("Quy đổi VNĐ", "VND Equivalent")
            ref_value = f"{df_world['Gold_VND'].iloc[-1]:,.2f}M"
        else:
            ref_label = t("Quy đổi USD", "USD Base")
            ref_value = f"${df_world['Gold_Close'].iloc[-1]:,.2f}"

        sjc_val = df_sjc['SJC_Sell'].iloc[-1]/1e6 if not df_sjc.empty else 0
        sjc_display = f"{sjc_val:,.2f}M" if sjc_val else "N/A"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-container">
                <div class="metric-label">{t('Giá Giao Ngay', 'Spot Price')} ({currency_mode})</div>
                <div class="metric-value">{currency_symbol}{curr:,.2f}</div>
                <div class="metric-sub {delta_class}">{delta_sign}{change:,.2f} ({delta_sign}{change_pct:.2f}%)</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-container">
                <div class="metric-label">{t('Tỷ Giá USD/VND', 'USD/VND Rate')}</div>
                <div class="metric-value">{usdvnd:,.0f}</div>
                <div class="metric-sub text-neutral">{t('Tỷ giá gốc', 'Base Rate')}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-container">
                <div class="metric-label">{ref_label}</div>
                <div class="metric-value">{ref_value}</div>
                <div class="metric-sub text-neutral">{t('Đã quy đổi', 'Converted')}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-container">
                <div class="metric-label">{t('Giá SJC Bán Ra', 'SJC Ask Price')}</div>
                <div class="metric-value">{sjc_display}</div>
                <div class="metric-sub text-neutral">{t('Thị trường nội địa', 'Local Market')}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.write(t("Dữ liệu không đủ.", "Insufficient data."))

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if not df_full.empty:
        # SJC Premium Spread Visual Meter Card
        if not df_sjc.empty and not df_world.empty:
            df_world_date = df_world.copy()
            df_world_date.index = pd.to_datetime(df_world_date.index).date
            df_spread = df_sjc.join(df_world_date[['Gold_VND']], how='inner')
            
            if not df_spread.empty:
                df_spread['Spread'] = (df_spread['SJC_Sell'] / 1e6) - df_spread['Gold_VND']
                latest_spread = float(df_spread['Spread'].iloc[-1])
                
                if latest_spread < 5.0:
                    badge_class = "badge-normal"
                    badge_text = t("[MỨC CHÊNH LỆCH: BÌNH THƯỜNG]", "[SPREAD STATUS: NORMAL]")
                    desc_text = t("Chênh lệch giữa giá SJC và giá thế giới ở mức hợp lý (< 5 triệu VNĐ/lượng). Mức độ rủi ro quy đổi thấp.", "SJC premium over world spot is reasonable (< 5M VND/tael). Low conversion risk.")
                elif latest_spread <= 10.0:
                    badge_class = "badge-high"
                    badge_text = t("[MỨC CHÊNH LỆCH: CAO]", "[SPREAD STATUS: HIGH]")
                    desc_text = t("Giá SJC đang cao hơn giá thế giới từ 5-10 triệu VNĐ/lượng. Người mua cần chú ý biến động chính sách.", "SJC is 5-10M VND higher than world spot gold. Watch policy shifts.")
                else:
                    badge_class = "badge-very-high"
                    badge_text = t("[MỨC CHÊNH LỆCH: ĐẮT BẤT THƯỜNG]", "[SPREAD STATUS: VERY HIGH]")
                    desc_text = t("Chênh lệch vượt mốc 10 triệu VNĐ/lượng. Giá trong nước đang đắt bất thường so với thế giới.", "SJC premium exceeds 10M VND/tael. Exceptionally expensive relative to global spot.")

                st.markdown(f"""
                <div class="spread-card">
                    <span class="spread-badge {badge_class}">{badge_text}</span>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 600; color: #0F172A; margin-bottom: 6px;">
                        {latest_spread:,.2f} {t('Triệu VNĐ / Lượng', 'Million VND / Tael')}
                    </div>
                    <div style="font-size: 13px; color: #475569;">
                        {desc_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown(f"#### {t('Hành Động Giá Vàng', 'Gold Price Action')} ({unit_label})")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df_full.index,
            y=df_full['View_Price'],
            mode='lines',
            line=dict(color='#D97706', width=2),
            fill='tozeroy',
            fillcolor='rgba(217, 119, 6, 0.08)',
            name=t("Giá Vàng", "Gold Price")
        ))
        fig1.update_layout(yaxis_title="", template=None)
        st.plotly_chart(style_chart(fig1), use_container_width=True, theme=None)
        
        render_chart_insight({
            "Start_Price": float(df_full['View_Price'].iloc[0]),
            "End_Price": float(df_full['View_Price'].iloc[-1]),
            "Min_Price": float(df_full['View_Price'].min()),
            "Max_Price": float(df_full['View_Price'].max()),
            "Timeframe_Days": len(df_full)
        }, "Price Action")

        if is_basic_mode:
            st.markdown(f"#### {t('Bản đồ Lợi Nhuận Hàng Tháng', 'Monthly Returns Heatmap')}")
            df_month = df_full.copy()
            df_month['Month'] = df_month.index.strftime('%m-%Y')
            monthly_ret = df_month.resample('ME')['Daily_Return'].sum() * 100
            fig4 = px.bar(x=monthly_ret.index.strftime('%Y-%m'), y=monthly_ret.values)
            fig4.update_traces(marker_color=np.where(monthly_ret.values >= 0, '#10B981', '#EF4444'))
            fig4.update_layout(xaxis_title="", yaxis_title=t("Lợi Nhuận %", "Return %"), template=None)
            st.plotly_chart(style_chart(fig4), use_container_width=True, theme=None)

        else: # Pro / Quant Mode
            col_chart_1, col_chart_2 = st.columns(2)
            with col_chart_1:
                st.markdown(f"#### {t('Biểu đồ Nến', 'Candlestick View')}")
                fig2 = go.Figure(data=[go.Candlestick(x=df_full.index,
                                open=df_full['View_Open'], high=df_full['View_High'],
                                low=df_full['View_Low'], close=df_full['View_Price'],
                                increasing_line_color='#10B981', decreasing_line_color='#EF4444')])
                fig2.update_layout(xaxis_rangeslider_visible=False, template=None)
                st.plotly_chart(style_chart(fig2), use_container_width=True, theme=None)

            with col_chart_2:
                st.markdown(f"#### {t('Tương Quan Vĩ Mô', 'Macro Correlation')}")
                fig3 = make_subplots(specs=[[{"secondary_y": True}]])
                fig3.add_trace(go.Scatter(x=df_full.index, y=df_full['View_Price'], name=t("Vàng", "Gold"), line=dict(color='#D97706', width=1.8)), secondary_y=False)
                fig3.add_trace(go.Scatter(x=df_full.index, y=df_full['USDVND'], name="USD/VND", line=dict(color="#0F172A", width=1.8)), secondary_y=True)
                st.plotly_chart(style_chart(fig3), use_container_width=True, theme=None)
                
            render_chart_insight({
                "Gold_Spot_Price": float(df_full['View_Price'].iloc[-1]),
                "USD_VND_Rate": float(df_full['USDVND'].iloc[-1]),
                "Gold_in_VND_Per_Tael": float(df_full['Gold_VND'].iloc[-1]),
                "Correlation_Gold_USDVND": float(df_full['View_Price'].corr(df_full['USDVND']))
            }, "Macro Correlation")

            col_spread_1, col_spread_2 = st.columns(2)
            with col_spread_1:
                df_month = df_full.copy()
                df_month['Month'] = df_month.index.strftime('%m-%Y')
                monthly_ret = df_month.resample('ME')['Daily_Return'].sum() * 100
                fig4 = px.bar(x=monthly_ret.index.strftime('%Y-%m'), y=monthly_ret.values)
                fig4.update_traces(marker_color=np.where(monthly_ret.values >= 0, '#10B981', '#EF4444'))
                fig4.update_layout(xaxis_title="", yaxis_title=t("Lợi Nhuận %", "Return %"), template=None)
                st.plotly_chart(style_chart(fig4), use_container_width=True, theme=None)
                
            with col_spread_2:
                st.markdown(f"##### {t('Chênh Lệch Giá SJC vs Quốc Tế Quy Đổi', 'SJC Domestic Premium Spread')} ({t('Tr.VNĐ / Lượng', 'VND (Mil) / Tael')})")
                if not df_sjc.empty and not df_world.empty and 'df_spread' in locals() and not df_spread.empty:
                    fig_spread = go.Figure()
                    fig_spread.add_trace(go.Scatter(
                        x=df_spread.index,
                        y=df_spread['Spread'],
                        mode='lines',
                        line=dict(color='#10B981', width=1.8),
                        fill='tozeroy',
                        fillcolor='rgba(16, 185, 129, 0.03)',
                        name=t("Mức Chênh Lệch", "Premium Spread")
                    ))
                    fig_spread.update_layout(yaxis_title="", template=None)
                    st.plotly_chart(style_chart(fig_spread), use_container_width=True, theme=None)

# ==================== TAB 2: P&L SIMULATOR / PERFORMANCE ====================
with tab2:
    if is_basic_mode:
        st.markdown(f"#### {t('Mô Phỏng Đầu Tư & Lợi Nhuận P&L', 'Investment & P&L Simulation')} ({currency_mode})")
    else:
        st.markdown(f"#### {t('Phân Tích Hiệu Suất & Rủi Ro', 'Performance & Risk Analysis')} ({currency_mode})")
    
    if not df_full.empty:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"##### {t('Mô Phỏng Kịch Bản', 'Scenario Simulator')}")
            invest = st.number_input(f"{t('Vốn Ban Đầu', 'Initial Capital')} ({currency_symbol})", value=1000 if currency_mode=='USD' else 50, step=100 if currency_mode=='USD' else 10)
            
            min_date = df_full.index.min().date()
            max_date = df_full.index.max().date()
            buy_date = st.date_input(t("Ngày Mua", "Entry Date"), value=min_date, min_value=min_date, max_value=max_date)
            
            idx = df_full.index.get_indexer([pd.Timestamp(buy_date)], method='nearest')[0]
            buy_p = df_full['View_Price'].iloc[idx]
            curr_p = df_full['View_Price'].iloc[-1]
            profit = (invest / buy_p * curr_p) - invest
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric(t("Lợi Nhuận Dự Kiến", "Projected P&L"), f"{currency_symbol}{profit:,.2f}", f"{(profit/invest)*100:.1f}%")

        with c2:
            st.markdown(f"##### {t('Lợi Nhuận Tích Lũy', 'Cumulative Return')}")
            fig_cum = px.line(df_full, y='Cumulative_Return', template=None)
            fig_cum.add_hline(y=1, line_dash="dash", line_color="rgba(0,0,0,0.15)")
            fig_cum.update_traces(line_color="#0F172A", line_width=1.8)
            fig_cum.update_layout(yaxis_title=t("Hệ số Nhân", "Multiplier"))
            st.plotly_chart(style_chart(fig_cum), use_container_width=True, theme=None)
            
            render_chart_insight({
                "Initial_Investment": invest,
                "Entry_Date": str(buy_date),
                "Buy_Price": float(buy_p),
                "Current_Price": float(curr_p),
                "PL_Val": float(profit),
                "PL_Pct": float((profit/invest)*100),
                "Cumulative_Return_Multiplier": float(df_full['Cumulative_Return'].iloc[-1])
            }, "Scenario & Cumulative Performance")

        if not is_basic_mode:
            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"##### {t('Mức Sụt Giảm Tối Đa', 'Maximum Drawdown')}")
                fig_dd = px.line(df_full, y='Drawdown', template=None)
                fig_dd.update_traces(line_color="#EF4444", fill='tozeroy', fillcolor="rgba(239, 68, 68, 0.03)")
                st.plotly_chart(style_chart(fig_dd), use_container_width=True, theme=None)
            
            with r2:
                st.markdown(f"##### {t('Độ Biến Động 30 Ngày', '30-Day Volatility')}")
                df_full['Vol_30'] = df_full['Daily_Return'].rolling(30).std()
                fig_vol = px.line(df_full, y='Vol_30', template=None)
                fig_vol.update_traces(line_color="#0F172A", line_width=1.8)
                st.plotly_chart(style_chart(fig_vol), use_container_width=True, theme=None)

# ==================== TAB 3: BASIC AI REPORT / PRO TECHNICALS ====================
if is_basic_mode:
    with tab3:
        st.markdown(f"#### {t('Báo Cáo AI Điều Hành', 'Executive AI Market Report')}")
        if not df_full.empty:
            df_today = df_full.iloc[[-1]]
            curr_price = df_today['View_Price'].iloc[-1]
            curr_rsi = df_today['RSI'].iloc[-1] if 'RSI' in df_today.columns else 50.0
            curr_macd = df_today['MACD'].iloc[-1] if 'MACD' in df_today.columns else 0.0
            curr_sma20 = df_today['SMA_20'].iloc[-1] if 'SMA_20' in df_today.columns else curr_price
            curr_sma50 = df_today['SMA_50'].iloc[-1] if 'SMA_50' in df_today.columns else curr_price

            if st.button(t("TẠO BÁO CÁO PHÂN TÍCH AI", "GENERATE EXECUTIVE AI REPORT"), use_container_width=True):
                with st.spinner(t("Đang phân tích dữ liệu thị trường...", "Analyzing market data with AI...")):
                    report = get_groq_analysis(
                        current_price=curr_price,
                        predicted_price=curr_price * 1.002,
                        rsi=curr_rsi,
                        macd=curr_macd,
                        sma20=curr_sma20,
                        sma50=curr_sma50,
                        currency=currency_mode
                    )
                    
                    report_html = report.replace('\n', '<br>')
                    st.markdown(f"""
                    <div class="quant-terminal">
                        <div class="terminal-header">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span class="dot-red"></span>
                                <span class="dot-yellow"></span>
                                <span class="dot-green"></span>
                                <span class="terminal-title">EXECUTIVE AI REPORT</span>
                            </div>
                            <span class="status-active">SYSTEM ACTIVE</span>
                        </div>
                        <div class="terminal-body">
                            {report_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
else: # Pro Mode Tab 3 Technicals & Tab 4 AI Models
    with tab3:
        st.markdown(f"#### {t('Chỉ Báo Kỹ Thuật', 'Technical Indicators')}")
        if not df_full.empty:
            t1, t2 = st.columns(2)
            with t1:
                st.markdown(f"##### {t('Dải Bollinger (20, 2)', 'Bollinger Bands (20, 2)')}")
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(x=df_full.index, y=df_full['BB_Upper'], line=dict(color='rgba(0,0,0,0.1)', width=1), name=t('Cận trên', 'Upper')))
                fig_bb.add_trace(go.Scatter(x=df_full.index, y=df_full['BB_Lower'], line=dict(color='rgba(0,0,0,0.1)', width=1), fill='tonexty', fillcolor='rgba(0,0,0,0.01)', name=t('Cận dưới', 'Lower')))
                fig_bb.add_trace(go.Scatter(x=df_full.index, y=df_full['View_Price'], line=dict(color='#D97706', width=1.8), name=t('Giá', 'Price')))
                fig_bb.update_layout(showlegend=False)
                st.plotly_chart(style_chart(fig_bb), use_container_width=True, theme=None)

            with t2:
                st.markdown(f"##### {t('Đường Trung Bình (20 & 50)', 'Moving Averages (20 & 50)')}")
                fig_sma = go.Figure()
                fig_sma.add_trace(go.Scatter(x=df_full.index, y=df_full['SMA_20'], line=dict(color='#3B82F6', width=1.2), name='SMA 20'))
                fig_sma.add_trace(go.Scatter(x=df_full.index, y=df_full['SMA_50'], line=dict(color='#EF4444', width=1.2), name='SMA 50'))
                fig_sma.add_trace(go.Scatter(x=df_full.index, y=df_full['View_Price'], line=dict(color='#94A3B8', width=1), opacity=0.4, name=t('Giá', 'Price')))
                st.plotly_chart(style_chart(fig_sma), use_container_width=True, theme=None)

            t3, t4 = st.columns(2)
            with t3:
                st.markdown(f"##### {t('Chỉ Số Sức Mạnh Tương Đối (RSI 14)', 'Relative Strength Index (14)')}")
                fig_rsi = px.line(df_full, y='RSI')
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="rgba(0,0,0,0.15)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="rgba(0,0,0,0.15)")
                fig_rsi.update_traces(line_color='#0F172A', line_width=1.8)
                fig_rsi.update_layout(yaxis_range=[0, 100], yaxis_title="")
                st.plotly_chart(style_style(fig_rsi) if 'style_style' in locals() else style_chart(fig_rsi), use_container_width=True, theme=None)

            with t4:
                st.markdown(f"##### {t('Đường Xu Hướng MACD', 'MACD (12, 26, 9)')}")
                fig_macd = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig_macd.add_trace(go.Scatter(x=df_full.index, y=df_full['MACD'], line=dict(color='#0F172A', width=1.8), name='MACD'), row=1, col=1)
                fig_macd.add_trace(go.Scatter(x=df_full.index, y=df_full['MACD_Signal'], line=dict(color='#94A3B8', width=1.8), name='Signal'), row=1, col=1)
                colors = np.where(df_full['MACD_Hist'] < 0, '#EF4444', '#10B981')
                fig_macd.add_trace(go.Bar(x=df_full.index, y=df_full['MACD_Hist'], marker_color=colors, name='Hist'), row=2, col=1)
                fig_macd.update_layout(showlegend=False)
                st.plotly_chart(style_chart(fig_macd), use_container_width=True, theme=None)

    with tab4:
        st.markdown(f"#### {t('Dự Báo Định Lượng ML', 'Quantitative ML Forecasting')}")
        if not df_full.empty:
            df_ml = df_full.copy()
            df_ml['Lag1'] = df_ml['View_Price'].shift(1)
            df_ml['Lag2'] = df_ml['View_Price'].shift(2)
            df_ml['Target'] = df_ml['View_Price'].shift(-1)
            
            feats = ['View_Price', 'Lag1', 'Lag2', 'SMA_20', 'RSI', 'MACD', 'USDVND']
            df_today = df_ml.iloc[[-1]]
            df_train_set = df_ml.dropna(subset=['Target'] + feats)
            
            if len(df_train_set) < 15:
                st.write(t("Dữ liệu không đủ để chạy AI.", "Insufficient data for ML model."))
            else:
                X = df_train_set[feats]
                y = df_train_set['Target']
                
                split = int(len(X)*0.8)
                X_train, X_test = X.iloc[:split], X.iloc[split:]
                y_train, y_test = y.iloc[:split], y.iloc[split:]
                
                rf = Ridge(alpha=1.0)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                
                c_ai1, c_ai2 = st.columns([1, 2])
                with c_ai1:
                    next_pred = rf.predict(df_today[feats])[0]
                    mae = mean_absolute_error(y_test, y_pred)
                    st.markdown(f"""<div class="ai-card">
                        <div class="ai-header">{t('Dự Báo Của Mô Hình (T+1)', 'Model Prediction (T+1)')}</div>
                        <div class="ai-val">{currency_symbol}{next_pred:,.2f}</div>
                        <div class="ai-stat">MAE: {currency_symbol}{mae:.2f}</div>
                    </div>""", unsafe_allow_html=True)
                
                with c_ai2:
                    imp = pd.DataFrame({'Feat': feats, 'Imp': np.abs(rf.coef_)}).sort_values('Imp')
                    fig_imp = px.bar(imp, x='Imp', y='Feat', orientation='h')
                    fig_imp.update_traces(marker_color='#0F172A')
                    fig_imp.update_layout(xaxis_title=t("Trọng Số Quan Trọng", "Importance Weight"), yaxis_title="")
                    st.plotly_chart(style_chart(fig_imp), use_container_width=True, theme=None)

                a1, a2 = st.columns(2)
                with a1:
                    st.markdown(f"##### {t('Kết Quả Kiểm Thử (Backtest)', 'Backtest Results')}")
                    df_res = pd.DataFrame({'Actual': y_test, 'Pred': y_pred}, index=y_test.index)
                    fig_back = go.Figure()
                    fig_back.add_trace(go.Scatter(x=df_res.index, y=df_res['Actual'], name=t('Thực tế', 'Actual'), line=dict(color='#94A3B8')))
                    fig_back.add_trace(go.Scatter(x=df_res.index, y=df_res['Pred'], name=t('Mô hình', 'Model'), line=dict(color='#0F172A', dash='dot')))
                    st.plotly_chart(style_chart(fig_back), use_container_width=True, theme=None)

                with a2:
                    st.markdown(f"##### {t('Dự Báo 5 Ngày Tới', '5-Day Future Forecast')}")
                    forecast_days = 5
                    future_preds = []
                    for step in range(1, forecast_days + 1):
                        df_ml[f'Target_T{step}'] = df_ml['View_Price'].shift(-step)
                        df_train_step = df_ml.dropna(subset=[f'Target_T{step}'] + feats)
                        if len(df_train_step) > 5:
                            model_step = Ridge(alpha=1.0)
                            model_step.fit(df_train_step[feats], df_train_step[f'Target_T{step}'])
                            future_preds.append(model_step.predict(df_today[feats])[0])
                    
                    if len(future_preds) == forecast_days:
                        last_date = df_full.index[-1]
                        future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
                        plot_dates = [last_date] + future_dates
                        plot_vals = [df_full['View_Price'].iloc[-1]] + future_preds
                        
                        past_df = df_full.iloc[-20:]
                        fig_future = go.Figure()
                        fig_future.add_trace(go.Scatter(x=past_df.index, y=past_df['View_Price'], name=t('Lịch sử', 'Historical'), line=dict(color='#94A3B8', width=2)))
                        fig_future.add_trace(go.Scatter(x=plot_dates, y=plot_vals, name=t('Dự báo', 'Forecast'), line=dict(color='#D97706', dash='dot', width=2)))
                        st.plotly_chart(style_chart(fig_future), use_container_width=True, theme=None)

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown(f"#### {t('Báo Cáo Phân Tích Bằng AI', 'Generative AI Market Report')}")
                if st.button(t("TẠO BÁO CÁO AI GROQ", "GENERATE GROQ AI REPORT"), use_container_width=True):
                    with st.spinner(t("Đang phân tích dữ liệu...", "Analyzing market data...")):
                        curr_price = df_today['View_Price'].iloc[-1]
                        curr_rsi = df_today['RSI'].iloc[-1]
                        curr_macd = df_today['MACD'].iloc[-1]
                        curr_sma20 = df_today['SMA_20'].iloc[-1]
                        curr_sma50 = df_today['SMA_50'].iloc[-1]
                        
                        report = get_groq_analysis(
                            current_price=curr_price,
                            predicted_price=next_pred,
                            rsi=curr_rsi,
                            macd=curr_macd,
                            sma20=curr_sma20,
                            sma50=curr_sma50,
                            currency=currency_mode
                        )
                        
                        report_html = report.replace('\n', '<br>')
                        st.markdown(f"""
                        <div class="quant-terminal">
                            <div class="terminal-header">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span class="dot-red"></span>
                                    <span class="dot-yellow"></span>
                                    <span class="dot-green"></span>
                                    <span class="terminal-title">QUANT INTELLIGENCE REPORT</span>
                                </div>
                                <span class="status-active">SYSTEM ACTIVE</span>
                            </div>
                            <div class="terminal-body">
                                {report_html}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

st.markdown("""<div class="footer">
    Data Intelligence Platform &middot; Quant Model v1.2 &middot; Enterprise Fintech UI
</div>""", unsafe_allow_html=True)