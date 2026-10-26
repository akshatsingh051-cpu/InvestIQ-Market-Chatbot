# 🤖 InvestIQ: AI-Powered Market Intelligence & Trading Chatbot

InvestIQ is a multimodal financial AI agent that synthesizes natural language processing (NLP) and quantitative technical analysis to deliver real-time stock recommendations. 

### 🚀 Key Features
* **Hybrid Entity Resolution:** Maps complex company names to global and Indian (NSE/BSE) stock tickers using a custom Alias + BERT-NER pipeline.
* **FinBERT Sentiment Engine:** Calculates Softmax probability distributions on unstructured financial news headlines.
* **Algorithmic Trend Analysis:** Integrates 5-day and 15-day Simple Moving Average (SMA) crossovers for technical validation.
* **Dynamic Competitor Momentum:** Scans sector peers to detect industry-wide contagion risks.
* **5-Page Streamlit GUI:** Features secure login, conversational AI interface, live market dashboard, and real-time model evaluation metrics.

### 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Machine Learning:** PyTorch, Hugging Face Transformers (`ProsusAI/finbert`, `dslim/bert-base-NER`)
* **Market Data & Vis:** yfinance, mplfinance, Pandas
* **Evaluation:** Scikit-learn, Matplotlib, NumPy

### 💻 How to Run Locally
1. Clone the repository: `git clone https://github.com/your-username/InvestIQ-Market-Chatbot.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the application: `streamlit run app.py`

*Built for the B.Tech CSE Capstone Project at MIT World Peace University.*