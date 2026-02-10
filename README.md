# 📊 AI Data Quality Analyst (Hybrid Architecture)

An intelligent data cleaning assistant that combines **deterministic statistical profiling** (Pandas) with **probabilistic reasoning** (Llama-3). This tool analyzes CSV datasets, identifies quality issues (missing values, duplicates, type mismatches), and automatically generates Python code to fix them.

## 🚀 Features

* **Hybrid Architecture:** Uses Pandas for accurate calculations (counting rows, nulls) and LLM for qualitative interpretation.
* **Automated EDA:** Instantly profiles datasets to find outliers and anomalies.
* **Code Generation:** Instead of just pointing out errors, it writes the exact `pandas` code to fix them.
* **Scalable:** Since only the metadata (statistics) is sent to the LLM, it can handle large datasets without hitting token limits.

## 🛠️ Tech Stack

* **Analysis Engine:** Python (Pandas)
* **Reasoning Engine:** Llama-3.3-70b (via Groq API)
* **UI:** Streamlit
* **Architecture:** Hybrid (Deterministic + Generative)

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Mervecaliskann/AI-Data-Analyst.git](https://github.com/Mervecaliskann/AI-Data-Analyst.git)
    cd AI-Data-Analyst
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 📂 Usage

1.  Upload a CSV file (e.g., raw sales data, customer logs).
2.  View the automated statistical profile (missing values, duplicates).
3.  Click **"Analyze with AI"** to get a detailed cleaning report and copy-pasteable Python code fixes.

---
*Developed by Merve Çalışkan*