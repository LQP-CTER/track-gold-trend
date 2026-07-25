import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime, timedelta

def test_tier1_vangtoday():
    print("\n--- Tier 1: Vang.today API ---")
    try:
        r = requests.get('https://www.vang.today/api/prices', timeout=5)
        if r.status_code == 200:
            data = r.json()
            vngsjc = data.get('prices', {}).get('VNGSJC', {})
            buy = float(vngsjc.get('buy', 0))
            sell = float(vngsjc.get('sell', 0))
            print(f"SUCCESS: VNGSJC Buy={buy:,.0f} VND, Sell={sell:,.0f} VND")
            return buy, sell
    except Exception as e:
        print(f"FAILED: {e}")
    return None, None

def test_tier2_webgia_live():
    print("\n--- Tier 2A: WebGia Live Scraping ---")
    try:
        url = "https://webgia.com/gia-vang/sjc/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, 'html.parser')
        tables = soup.find_all('table')
        if tables:
            for row in tables[0].find_all('tr'):
                cols = [c.text.strip() for c in row.find_all(['td', 'th'])]
                if len(cols) >= 4 and 'Hồ Chí Minh' in cols[0] and '1L' in cols[1]:
                    # Format: 13.750.000 (VND/chỉ) -> x10 -> 137.500.000 (VND/lượng)
                    buy_chi = float(cols[2].replace('.', '').replace(',', ''))
                    sell_chi = float(cols[3].replace('.', '').replace(',', ''))
                    buy = buy_chi * 10 if buy_chi < 100000000 else buy_chi
                    sell = sell_chi * 10 if sell_chi < 100000000 else sell_chi
                    print(f"SUCCESS: WebGia Live Buy={buy:,.0f} VND, Sell={sell:,.0f} VND")
                    return buy, sell
    except Exception as e:
        print(f"FAILED: {e}")
    return None, None

def test_tier2_webgia_history():
    print("\n--- Tier 2B: WebGia Highcharts History ---")
    try:
        url = "https://webgia.com/gia-vang/sjc/bieu-do-1-thang.html"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r = requests.get(url, headers=headers, timeout=5)
        pairs = re.findall(r'\[(\d{12,13}),\s*([\d\.]+)\]', r.text)
        if pairs:
            history = []
            for ts_str, price_m in pairs:
                dt = datetime.fromtimestamp(int(ts_str) / 1000).date()
                price_vnd = float(price_m) * 1e6 # Triệu VNĐ -> VNĐ
                history.append((dt, price_vnd))
            print(f"SUCCESS: Extracted {len(history)} historical data points. Latest: {history[-1]}")
            return history
    except Exception as e:
        print(f"FAILED: {e}")
    return []

def test_tier3_vnstock():
    print("\n--- Tier 3: Vnstock ---")
    try:
        from vnstock.explorer.misc import sjc_gold_price
        df = sjc_gold_price(date=datetime.now().strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            row = df[df['branch'] == 'Hồ Chí Minh']
            if not row.empty:
                buy = float(str(row.iloc[0]['buy_price']).replace(',', ''))
                sell = float(str(row.iloc[0]['sell_price']).replace(',', ''))
                if buy < 1000000: buy *= 1e6
                if sell < 1000000: sell *= 1e6
                print(f"SUCCESS: Vnstock Buy={buy:,.0f} VND, Sell={sell:,.0f} VND")
                return buy, sell
    except Exception as e:
        print(f"FAILED: {e}")
    return None, None

def test_tier4_failsafe():
    print("\n--- Tier 4: Failsafe Estimator ---")
    try:
        import yfinance as yf
        gold_price = float(yf.Ticker('GC=F').fast_info['lastPrice'])
        usdvnd = float(yf.Ticker('VND=X').fast_info['lastPrice'])
        # 1 Ounce = 1.205653 Tael. Premium ~12%
        est_base = (gold_price * usdvnd * 1.205653)
        est_sell = est_base * 1.12
        est_buy = est_sell * 0.97
        print(f"SUCCESS: Estimated SJC Buy={est_buy:,.0f} VND, Sell={est_sell:,.0f} VND (World={gold_price}, USDVND={usdvnd})")
        return est_buy, est_sell
    except Exception as e:
        print(f"FAILED: {e}")
    return None, None

if __name__ == "__main__":
    print("=========================================")
    print("   TESTING 4-TIER SJC SCRAPING PIPELINE  ")
    print("=========================================")
    test_tier1_vangtoday()
    test_tier2_webgia_live()
    test_tier2_webgia_history()
    test_tier3_vnstock()
    test_tier4_failsafe()
