# Gold Trend Chart

**Gold Trend Chart** is a Streamlit-based web application that visualizes historical and current gold prices, including world gold prices (USD/oz) and SJC gold prices (VND/cây) in Vietnam. It fetches historical world gold data from Yahoo Finance, current world gold prices from Trading Economics, and SJC buy/sell prices from the `vnstock` library. The app displays interactive line charts using Plotly and provides metrics for the latest prices and buy/sell spreads.

## Features
- **Historical Data Visualization**:
  - World gold prices (USD/oz) from Yahoo Finance.
  - SJC gold buy, sell, and spread (VND/cây) from `vnstock`.
  - Interactive line charts showing price trends over user-selected time ranges.
- **Current World Gold Price**:
  - Scrapes real-time world gold prices from Trading Economics.
  - Updates on demand via a button in the sidebar.
- **Customizable Time Range**:
  - Predefined ranges (1 month, 3 months, 6 months, 1 year, YTD, or up to 10 years).
  - Manual date selection for custom periods.
- **Metrics Display**:
  - Latest world gold price (USD/oz) with delta from previous data point.
  - Latest SJC buy, sell, and spread (VND/cây) with deltas.
- **Raw Data Access**:
  - Expandable section to view raw data tables for world and SJC gold prices.
- **Responsive Design**:
  - Custom CSS with Inter font, supporting light and dark themes.
  - Optimized layout with sidebar controls and metrics.

## Prerequisites
- **Python**: Version 3.8 or higher.
- **Dependencies**: Install required Python packages listed in `requirements.txt`.
- **Internet Connection**: Required for fetching data from Yahoo Finance, Trading Economics, and `vnstock`.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd gold-trend-chart
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Create a `requirements.txt` file with the following content:
   ```
   streamlit
   pandas
   yfinance
   plotly
   requests
   beautifulsoup4
   vnstock
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   The app will open in your default web browser.

## Usage

1. **Access the Application**:
   - Open the app in a web browser after running the command above.
   - The app displays a title, caption, and sidebar for controls.

2. **Select Time Range (Sidebar)**:
   - Under **Thời gian Lịch sử**, choose a predefined range (e.g., "1 Tháng", "1 Năm", "Tất cả") or manually select start and end dates.
   - The app ensures the start date is not after the end date.

3. **Fetch Current World Gold Price**:
   - Click **Lấy giá TG mới nhất** in the sidebar to scrape the current world gold price from Trading Economics.
   - A toast notification confirms success, or an error message appears if the fetch fails.

4. **View Metrics**:
   - Four metrics display at the top:
     - **World Gold Price (USD/oz)**: Shows the latest scraped price (if available) or the most recent historical price, with delta.
     - **SJC Sell Price (VND/cây)**: Latest SJC sell price for Hồ Chí Minh branch.
     - **SJC Buy Price (VND/cây)**: Latest SJC buy price.
     - **SJC Spread (VND/cây)**: Difference between sell and buy prices.
   - Metrics include deltas compared to the previous data point (if available).

5. **View Charts**:
   - **World Gold Chart**: Displays historical world gold prices (USD/oz) as a line chart for the selected time range.
   - **SJC Gold Chart**: Shows SJC buy, sell, and spread prices (VND/cây) as overlaid lines, with color-coded legend.
   - Hover over charts to see precise values and dates.

6. **View Raw Data**:
   - Expand the **Xem dữ liệu gốc** section to view raw data tables for world gold (Open, High, Low, Close, Volume) and SJC gold (Buy, Sell, Spread).
   - Data is formatted for clarity and displayed in separate columns.

## Notes
- **Data Sources**:
  - **World Gold (Historical)**: Fetched from Yahoo Finance using the `GC=F` ticker (USD/oz).
  - **World Gold (Current)**: Scraped from Trading Economics, cached for 60 seconds.
  - **SJC Gold**: Fetched from `vnstock` for the Hồ Chí Minh branch, sampled every 10 days to avoid rate limits.
- **Caching**:
  - Historical data (Yahoo Finance and `vnstock`) is cached for 6 hours (`CACHE_TTL_SECONDS`).
  - Scraped world gold prices are cached for 60 seconds (`SCRAPE_CACHE_TTL_SECONDS`).
- **Error Handling**:
  - The app handles rate limits, empty data, and parsing errors with appropriate warnings or info messages.
  - If data fetching fails, charts are not displayed, and error messages guide the user.
- **SJC Data**:
  - Fetched every 10 days to reduce API load.
  - Only includes data for the Hồ Chí Minh branch (`SJC_TARGET_BRANCH`).
- **Limitations**:
  - Scraping from Trading Economics may be unreliable due to website changes or network issues.
  - Rate limits from Yahoo Finance or `vnstock` may temporarily prevent data retrieval.

## Example Output
- **Metrics Example** (hypothetical):
  - Giá TG (12:30): $2,500.00 USD, Δ: +10.50 USD
  - SJC Bán (15/06/2025): 92,000,000 VND, Δ: +500,000 VND
  - SJC Mua (15/06/2025): 90,000,000 VND, Δ: +400,000 VND
  - Chênh lệch SJC: 2,000,000 VND, Δ: +100,000 VND
- **Charts**: Interactive Plotly line charts showing trends for world gold and SJC prices.
- **Raw Data**: Tables with historical values, e.g., world gold (Open: 2,490.50, Close: 2,500.00) and SJC (Buy: 90,000,000, Sell: 92,000,000).

## Troubleshooting
- **"vnstock not installed" Error**:
  - Install the library: `pip install vnstock`.
  - Ensure the `sjc_gold_price` function is available in your `vnstock` version.
- **Rate Limit Errors**:
  - Yahoo Finance or `vnstock` may impose rate limits. Wait a few minutes and retry.
  - Reduce the frequency of data fetches by using cached data (6-hour TTL).
- **Scraping Failures**:
  - If Trading Economics scraping fails, check your internet connection or try again later.
  - The website’s structure may have changed, requiring code updates.
- **Empty Charts**:
  - Ensure the selected time range contains available data.
  - Verify that start_date is not after end_date.
- **Data Formatting Issues**:
  - Check console logs for specific errors (e.g., parsing or conversion issues).
  - Ensure numeric columns (e.g., prices) are correctly formatted in the source data.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License
Copyright © LeQuyPhat. All rights reserved.
