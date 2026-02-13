<div align="center">

# 📈 GOLD TREND CHART

**Interactive Gold Price Visualization Dashboard**

Streamlit | Python | Yahoo Finance | vnstock | Plotly

---

Visualizes historical and current gold prices, including world gold prices (USD/oz) and SJC gold prices (VND/cây) in Vietnam.

</div>

---

## 📊 Project Overview

**Gold Trend Chart** is a Streamlit-based web application that provides comprehensive gold price analytics:

- **World Gold Prices**: Historical data from Yahoo Finance (USD/oz)
- **SJC Gold Prices**: Real-time buy/sell prices from Vietnam (VND/cây)
- **Current Market Data**: Live world gold prices from Trading Economics

The app delivers interactive visualizations using Plotly with customizable time ranges and detailed metrics.

---

## ✨ Features

### 📉 Historical Data Visualization
- World gold prices (USD/oz) from Yahoo Finance
- SJC gold buy, sell, and spread (VND/cây) from `vnstock`
- Interactive line charts showing price trends over user-selected time ranges

### 💰 Current World Gold Price
- Scrapes real-time world gold prices from Trading Economics
- Updates on demand via a button in the sidebar

### 🎛️ Customizable Time Range
- **Predefined ranges**: 1 month, 3 months, 6 months, 1 year, YTD, or up to 10 years
- **Manual selection**: Custom date picker for specific periods

### 📊 Metrics Display
- Latest world gold price (USD/oz) with delta from previous data point
- Latest SJC buy, sell, and spread (VND/cây) with deltas

### 📋 Raw Data Access
- Expandable section to view raw data tables
- Separate views for world and SJC gold prices

### 🎨 Responsive Design
- Custom CSS with Inter font
- Light and dark theme support
- Optimized layout with sidebar controls

---

## 🔧 Prerequisites

- **Python**: Version 3.8 or higher
- **Internet Connection**: Required for fetching data from Yahoo Finance, Trading Economics, and `vnstock`

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd gold-trend-chart
```

### Step 2: Set Up a Virtual Environment

*(Recommended)*

```bash
python -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages (`requirements.txt`):**
```
streamlit
pandas
yfinance
plotly
requests
beautifulsoup4
vnstock
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`.

---

## 📖 Usage

### 1. Access the Application
Open the app in your web browser after running the command above.

### 2. Select Time Range (Sidebar)
- Under **Thời gian Lịch sử**, choose a predefined range:
  - "1 Tháng", "3 Tháng", "6 Tháng", "1 Năm", "YTD", "Tất cả"
- Or manually select start and end dates
- The app ensures the start date is not after the end date

### 3. Fetch Current World Gold Price
- Click **Lấy giá TG mới nhất** in the sidebar
- Toast notification confirms success or shows error message

### 4. View Metrics
Four key metrics display at the top:
- **World Gold Price (USD/oz)**: Latest price with delta
- **SJC Sell Price (VND/cây)**: Latest SJC sell price
- **SJC Buy Price (VND/cây)**: Latest SJC buy price
- **SJC Spread (VND/cây)**: Difference between sell and buy prices

### 5. View Charts
- **World Gold Chart**: Historical prices (USD/oz) as interactive line chart
- **SJC Gold Chart**: Buy, sell, and spread prices with color-coded legend
- Hover over charts for precise values and dates

### 6. View Raw Data
- Expand **Xem dữ liệu gốc** section
- View formatted tables for world gold (Open, High, Low, Close, Volume)
- View SJC gold data (Buy, Sell, Spread)

---

## 📝 Technical Notes

### Data Sources
- **World Gold (Historical)**: Yahoo Finance using `GC=F` ticker (USD/oz)
- **World Gold (Current)**: Scraped from Trading Economics
- **SJC Gold**: Fetched from `vnstock` for Hồ Chí Minh branch

### Caching Strategy
- **Historical data**: Cached for 6 hours (`CACHE_TTL_SECONDS`)
- **Scraped prices**: Cached for 60 seconds (`SCRAPE_CACHE_TTL_SECONDS`)
- **SJC data**: Sampled every 10 days to avoid rate limits

### Error Handling
- Handles rate limits, empty data, and parsing errors
- Appropriate warnings and info messages for failures
- Graceful fallback when data fetching fails

### SJC Data Specifics
- Fetched every 10 days to reduce API load
- Only includes Hồ Chí Minh branch data (`SJC_TARGET_BRANCH`)

### Limitations
- Trading Economics scraping may be unreliable due to website changes
- Rate limits from Yahoo Finance or `vnstock` may temporarily prevent data retrieval

---

## 💡 Example Output

**Metrics Example** (hypothetical):
- **Giá TG (12:30)**: $2,500.00 USD, Δ: +10.50 USD
- **SJC Bán (15/06/2025)**: 92,000,000 VND, Δ: +500,000 VND
- **SJC Mua (15/06/2025)**: 90,000,000 VND, Δ: +400,000 VND
- **Chênh lệch SJC**: 2,000,000 VND, Δ: +100,000 VND

**Charts**: Interactive Plotly line charts showing trends for world gold and SJC prices.

**Raw Data**: Tables with historical values (e.g., world gold Open: 2,490.50, Close: 2,500.00).

---

## 🔍 Troubleshooting

### "vnstock not installed" Error
```bash
pip install vnstock
```
Ensure the `sjc_gold_price` function is available in your `vnstock` version.

### Rate Limit Errors
- Yahoo Finance or `vnstock` may impose rate limits
- Wait a few minutes and retry
- Use cached data (6-hour TTL) to reduce fetch frequency

### Scraping Failures
- Check internet connection
- Trading Economics website structure may have changed
- Try again later or check console logs

### Empty Charts
- Ensure selected time range contains available data
- Verify that `start_date` is not after `end_date`

### Data Formatting Issues
- Check console logs for specific errors
- Ensure numeric columns are correctly formatted in source data

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch
   ```bash
   git push origin feature/your-feature
   ```
5. Open a Pull Request

---

<div align="center">

**Developed by LeQuyPhat**  
*Data Analyst Portfolio*

[GitHub](https://github.com/LQP-CTER) • [Email](mailto:Lequyphat0123@gmail.com) • [Portfolio](https://lequyphat.wuaze.com/)

---

Copyright © LeQuyPhat. All rights reserved.

</div>
