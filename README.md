# Gold Trend Pro 📈 - Quantitative Forecasting Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![Groq](https://img.shields.io/badge/Groq-Qwen_3-black.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-blueviolet.svg)

A real-time, interactive **Streamlit dashboard** designed to track, analyze, and forecast the Vietnamese (SJC) and Global gold markets using **Quantitative Machine Learning** and **Generative AI**.

## 📊 Overview

The Gold Trend Pro Dashboard provides financial analysts and investors with a comprehensive, enterprise-grade view of the gold market. It seamlessly integrates real-time web scraping for domestic prices, Yahoo Finance for global trends, and advanced predictive modeling to deliver actionable financial insights.

### Key Features

*   **Real-time Synchronization:** Directly scrapes real-time SJC gold prices via `BeautifulSoup` and fetches global metrics via `yfinance`.
*   **Quantitative Forecasting:** Utilizes **Ridge Regression** to perform Direct Multi-step Forecasting, projecting the future price trajectory for the next 5 days.
*   **Generative AI Market Report:** Integrates the ultra-fast **Qwen 3 32B** model (via Groq API) to act as a professional financial analyst, generating instant, localized (Vietnamese) risk assessments based on live technical indicators.
*   **Advanced Technical Indicators:** Interactive Plotly charts including Bollinger Bands, SMA (20 & 50), RSI (14), and MACD.
*   **Minimalist Enterprise UI:** A strict dark-mode, icon-free design tailored for high-focus fintech environments.
*   **Robust State & Caching:** Utilizes `@st.cache_data` with automatic TTL to manage API rate limits and ensure snappy UI performance.

## 🎯 Target Audience

The dashboard is designed for the following user groups:
*   **Quantitative Analysts:** Needing quick access to multi-variable regression models (Lagging, SMA, RSI) and feature importance weights.
*   **Retail Investors:** Looking for easy-to-understand, AI-generated summaries of complex market conditions.
*   **Financial Researchers:** Requiring a unified view of domestic (VND) vs. global (USD) gold price parity and historical backtesting.

## ⚙️ Architecture & Technical Stack

*   **Framework**: Streamlit (Frontend & Backend integration)
*   **Data Processing**: Pandas, NumPy
*   **Data Visualization**: Plotly (Graph Objects, Subplots, Express)
*   **Machine Learning**: `sklearn.linear_model.Ridge`
*   **LLM Provider**: Groq Cloud API (Qwen 3)
*   **Data Sourcing**: `yfinance`, `bs4`

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/gold-trend-pro.git
   cd gold-trend-pro
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Groq API Secrets:**
   Create a `.streamlit/secrets.toml` file and securely add your Groq API Key:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

5. **Run the application:**
   ```bash
   streamlit run main.py
   ```

## 🔒 Security Notice

*   **API Key Privacy**: The `.streamlit/secrets.toml` and `.env` files are strictly ignored via `.gitignore` to prevent accidental exposure of your Groq API credentials.
*   **Cloud Deployment**: When deploying to Streamlit Community Cloud, do NOT upload the secrets file. Instead, inject the `GROQ_API_KEY` directly via the Streamlit App Settings -> Secrets dashboard.

---
*Developed as a Quantitative Fintech Project*
