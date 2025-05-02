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

@st.cache_data(ttl=SCRAPE_CACHE_TTL_SECONDS) # Cache the scraped price briefly
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

# --- Sidebar for Controls ---
with st.sidebar:
    st.markdown("<p class='sidebar-title'>Le Quy Phat</p>", unsafe_allow_html=True)
    st.header("📅 Thời gian Lịch sử") # Clarify date range is for historical charts
    st.write("")
    predefined_ranges = { "1 Tháng": 30, "3 Tháng": 90, "6 Tháng": 180, "1 Năm": 365, "Từ đầu năm (YTD)": "YTD", "Tất cả (Tối đa 10 năm)": "Max" }
    selected_range_label = st.selectbox("Chọn nhanh:", options=list(predefined_ranges.keys()), index=2)
    st.divider()
    st.markdown("**Hoặc chọn ngày:**")
    today = datetime.now().date()
    if selected_range_label == "Tất cả (Tối đa 10 năm)": default_start_date_calc = max(today - timedelta(days=10*365), datetime(2015, 1, 1).date())
    elif selected_range_label == "Từ đầu năm (YTD)": default_start_date_calc = datetime(today.year, 1, 1).date()
    else: default_start_date_calc = today - timedelta(days=predefined_ranges[selected_range_label])
    default_end_date_calc = today
    start_date_input = st.date_input("Từ ngày", default_start_date_calc, label_visibility="collapsed")
    end_date_input = st.date_input("Đến ngày", default_end_date_calc, label_visibility="collapsed")
    start_date = start_date_input; end_date = end_date_input
    final_label_hist = f"{start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}" # Label for historical range
    if start_date > end_date: st.error("Lỗi: Ngày bắt đầu không được sau ngày kết thúc."); st.stop()
    st.divider()
    st.caption(f"SJC lấy mỗi {SJC_FETCH_INTERVAL_DAYS} ngày ({SJC_TARGET_BRANCH}).")
    st.divider()
    st.header("🔄 Giá TG Hiện tại")
    update_world_button = st.button("Lấy giá TG mới nhất", key="update_world", help="Lấy giá vàng thế giới hiện tại từ Trading Economics (có thể không ổn định).")

# --- Main Page Layout ---
st.title("📊 Biểu đồ Lịch sử Giá Vàng")
st.caption(f"Giá TG (USD/oz) & Giá SJC (VND/cây) | Khoảng thời gian lịch sử: {final_label_hist}")
st.write("")

# --- Initialize variables ---
world_data_error = False
sjc_data_error = False
world_gold_usd_hist = pd.DataFrame() # For historical yfinance data
sjc_hist = pd.DataFrame()
gold_hist_raw = None # Store raw yfinance data
scraped_world_price = None # Store the manually scraped price
scraped_world_fetch_error = None # Store scraping error type

# --- Handle Scrape Button Click ---
if update_world_button:
    with st.spinner("Đang lấy giá TG mới nhất..."):
        scraped_world_price, scraped_world_fetch_error = get_world_gold_price_scrape()
        if scraped_world_fetch_error:
            if scraped_world_fetch_error == "fetch_error": st.error("⚠️ Lỗi mạng hoặc không thể kết nối đến Trading Economics để lấy giá TG.", icon="🌐")
            elif scraped_world_fetch_error == "not_found": st.error("⚠️ Không tìm thấy giá vàng ('Gold') trong dữ liệu từ Trading Economics.", icon="🔍")
            else: st.error(f"❌ Đã xảy ra lỗi khi xử lý dữ liệu giá TG từ Trading Economics (code: {scraped_world_fetch_error}).")
            scraped_world_price = None # Ensure price is None on error
        elif scraped_world_price is not None:
            st.success(f"Đã lấy giá TG hiện tại: ${scraped_world_price:.2f}")
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
if world_fetch_error_type and not update_world_button: # Don't show yfinance error if user just tried scraping
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

if sjc_fetch_error_type == "ratelimit": st.warning(f"⚠️ **Giới hạn truy cập (Giá SJC):** Có thể đã gặp giới hạn khi lấy dữ liệu SJC. Dữ liệu SJC có thể không đầy đủ hoặc không hiển thị. Vui lòng thử lại sau.", icon="⏳")
elif sjc_fetch_error_type == "nodata": st.info(f"ℹ️ Không tìm thấy dữ liệu SJC nào cho khoảng thời gian này ({final_label_hist}).")
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
if scraped_world_price is not None:
    latest_world_price_usd = scraped_world_price
    latest_world_date_str = datetime.now().strftime('%H:%M') # Show current time
    metric1_label = f"Giá TG ({latest_world_date_str})"
    # Calculate delta based on comparison with last historical price
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
st.subheader(f"🌍 Giá Vàng Thế Giới (USD/oz) - Lịch sử ({final_label_hist})") # Updated subheader
if world_data_error: st.info("Không có dữ liệu lịch sử giá vàng thế giới để hiển thị.")
elif not world_gold_usd_hist.empty:
    fig_world = px.line(world_gold_usd_hist, x='Timestamp', y='Giá TG (USD/oz)', labels={'Timestamp': 'Thời gian', 'Giá TG (USD/oz)': 'Giá (USD/oz)'})
    fig_world.update_traces(line_color='#0d6efd', hovertemplate="Ngày: %{x|%d/%m/%Y}<br>Giá: %{y:,.2f} USD<extra></extra>")
    fig_world.update_layout(hovermode="x unified", margin=dict(t=10, b=0, l=0, r=0))
    st.plotly_chart(fig_world, use_container_width=True)
else: st.info("Không có dữ liệu lịch sử giá vàng thế giới để hiển thị.")

# --- Display SJC Chart (Buy, Sell, Spread) ---
st.subheader(f"🇻🇳 Giá Vàng SJC (VND/cây) - Lịch sử ({final_label_hist})") # Updated subheader
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
