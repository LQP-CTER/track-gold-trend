"""
AI Narrative Generator — Gold Trend Pro
Auto-generate insight paragraphs using Groq API (LLM) based on quantitative gold market data.
"""

import streamlit as st
import os
from groq import Groq

# Initialize Groq Client
try:
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
    client = Groq(api_key=api_key)
except:
    client = None

@st.cache_data(ttl=3600)
def generate_ai_insight(data_json, dashboard_section="Overview", lang='VN'):
    """Generic AI Insight Generator reading JSON data for any gold market dashboard section."""
    if not client:
        return "Warning: GROQ_API_KEY not found in secrets or environment variables." if lang == 'EN' else "⚠️ Cảnh báo: Chưa cấu hình GROQ_API_KEY trong cấu hình."
        
    prompt = f"""
    You are a Senior Quantitative Gold Strategist and Financial Analyst.
    Below is the raw market data (aggregated in JSON format) for the section '{dashboard_section}' of the Gold Trend Pro analysis platform.

    JSON Data:
    {data_json}

    Task:
    Analyze this quantitative data, identify any key trends, risks, anomalies, or support/resistance points (e.g. price trends, RSI momentum levels, MACD crossovers, forecasting errors, or volatility changes), and write a concise, professional analysis (strictly 2-3 sentences max) summarizing the situation and recommending a logical action or stance.

    Strict Requirements:
    - Write the analysis in {'Vietnamese' if lang == 'VN' else 'English'}.
    - Keep the tone highly professional, precise, quantitative, and institutional-grade.
    - DO NOT USE ANY EMOJIS OR ICONS AT ALL (strictly prohibited).
    - Base your claims strictly on the actual numbers in the JSON data, avoiding generic advice or vague statements.
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="qwen-2.5-32b",
            temperature=0.3,
            max_tokens=350,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback to llama-3.1-8b-instant if qwen fails
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=350,
            )
            return response.choices[0].message.content.strip()
        except Exception as e2:
            return f"Groq API connection error: {str(e2)}" if lang == 'EN' else f"Lỗi kết nối Groq API: {str(e2)}"
