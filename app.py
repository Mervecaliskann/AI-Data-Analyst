"""
AI Data Analyst (Automated Data Quality & Cleaning Consultant)
--------------------------------------------------------------
Author: Merve Çalışkan
Description: 
A hybrid data analysis tool that uses Pandas for deterministic statistical profiling 
and Llama-3 (Groq) for qualitative interpretation and cleaning recommendations.
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from io import StringIO

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. AYARLAR
load_dotenv()
st.set_page_config(page_title="📊 AI Data Analyst", layout="wide")

if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ HATA: .env dosyasında GROQ_API_KEY bulunamadı!")
    st.stop()

# 2. PANDAS ANALİZ MOTORU (Deterministic Logic)
# AI'a ham veriyi vermeden önce, Python ile kesin istatistikleri çıkarıyoruz.
def analyze_data_quality(df: pd.DataFrame):
    """
    Pandas kullanarak verinin röntgenini çeker.
    Bu kısım AI değildir, saf matematiktir (Kesin sonuç verir).
    """
    analysis = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "numerical_summary": df.describe().to_string(), # Sayısal özet
        "column_names": list(df.columns)
    }
    return analysis

# 3. AI YORUMLAMA MOTORU (Probabilistic Logic)
# Pandas'tan gelen istatistikleri alıp, bir "Veri Bilimci" gibi yorumlar.
def get_ai_recommendation(analysis_result, sample_data):
    """
    İstatistiksel özeti Llama-3'e gönderir ve temizlik önerileri alır.
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.2 # Yaratıcılık az olsun, teknik konuşsun.
    )

    prompt_template = """
    You are a Senior Data Scientist Expert. 
    Analyze the following dataset profile and provide actionable cleaning recommendations.

    DATASET STATISTICS:
    - Total Rows: {rows}
    - Total Columns: {columns}
    - Duplicate Rows: {duplicates}
    - Missing Values per Column: {missing_values}
    - Data Types: {data_types}

    SAMPLE DATA (First 5 rows):
    {sample_data}

    YOUR TASK:
    1. Identify critical data quality issues (Missing values, duplicates, wrong types).
    2. Explain WHY these are problems.
    3. Write specific PYTHON (Pandas) code to fix these issues.

    OUTPUT FORMAT:
    - **Issue 1:** [Explanation]
    - **Code:** [Pandas Code]
    ...
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    # AI'a sadece özeti gönderiyoruz (Token tasarrufu)
    response = chain.invoke({
        "rows": analysis_result["rows"],
        "columns": analysis_result["columns"],
        "duplicates": analysis_result["duplicates"],
        "missing_values": analysis_result["missing_values"],
        "data_types": analysis_result["data_types"],
        "sample_data": sample_data
    })
    
    return response.content

# 4. ARAYÜZ (STREAMLIT)
st.title("📊 AI Data Quality Analyst")
st.markdown("""
Bu araç, CSV dosyalarınızı analiz eder, eksik/hatalı verileri bulur ve 
**Llama-3** destekli temizlik kodları önerir.
""")

uploaded_file = st.file_uploader("CSV Dosyanı Yükle", type=["csv"])

if uploaded_file is not None:
    # Dosyayı Oku
    df = pd.read_csv(uploaded_file)
    
    # 1. Ham Veriyi Göster
    with st.expander("🔍 Ham Veriyi İncele (İlk 5 Satır)", expanded=True):
        st.dataframe(df.head())

    # 2. Analiz Yap (Pandas)
    with st.spinner("İstatistiksel analiz yapılıyor..."):
        profile = analyze_data_quality(df)
        
        # Metrikleri Göster
        col1, col2, col3 = st.columns(3)
        col1.metric("Satır Sayısı", profile["rows"])
        col2.metric("Sütun Sayısı", profile["columns"])
        col3.metric("Tekrar Eden Satır", profile["duplicates"], 
                    delta_color="inverse" if profile["duplicates"] > 0 else "normal")

    # 3. AI Tavsiyesi Al (Groq)
    if st.button("🤖 Yapay Zeka ile Detaylı Analiz Et"):
        with st.spinner("Llama-3 veriyi yorumluyor ve temizlik kodu yazıyor..."):
            # AI'a sadece ilk 5 satırı ve istatistikleri atıyoruz (Tüm veriyi değil!)
            csv_preview = df.head().to_string()
            
            ai_advice = get_ai_recommendation(profile, csv_preview)
            
            st.markdown("### 💡 AI Temizlik Önerileri ve Kodları")
            st.markdown(ai_advice)
            st.success("Analiz tamamlandı!")

else:
    st.info("Lütfen analiz etmek için bir CSV dosyası yükleyin.")