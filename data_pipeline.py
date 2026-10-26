import yfinance as yf
import mplfinance as mpf
import numpy as np
import pandas as pd
import os

COMMON_COMPANY_ALIASES = {
    # Indian Equities (NSE/BSE)
    "INDIAN OIL": "IOC.NS",
    "INDIAN OIL CORP": "IOC.NS",
    "IOC": "IOC.NS",
    "RELIANCE": "RELIANCE.NS",
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "TCS": "TCS.NS",
    "TATA CONSULTANCY": "TCS.NS",
    "TATA CONSULTANCY SERVICES": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "INFY": "INFY.NS",
    "STATE BANK OF INDIA": "SBIN.NS",
    "SBI": "SBIN.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "L&T": "LT.NS",
    "LARSEN & TOUBRO": "LT.NS",
    "MAHINDRA": "M&M.NS",
    "M&M": "M&M.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "ZOMATO": "ZOMATO.NS",
    
    # Global Equities (US/NASDAQ/NYSE)
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "TESLA": "TSLA",
    "AMAZON": "AMZN",
    "NVIDIA": "NVDA",
    "META": "META",
    "AMD": "AMD"
}

def get_currency_symbol(ticker):
    ticker_clean = ticker.upper()
    if ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO"):
        return "₹"
    return "$"

def universal_ticker_resolver(company_name):
    if not company_name:
        return None
        
    clean_query = company_name.strip().upper()
    
    # 1. Alias Match
    for alias, ticker in COMMON_COMPANY_ALIASES.items():
        if alias == clean_query or alias in clean_query or clean_query in alias:
            return ticker
            
    # 2. Direct Suffix Check
    for suffix in ["", ".NS", ".BO"]:
        test_symbol = f"{clean_query}{suffix}"
        try:
            test_df = yf.download(test_symbol, period="1d", progress=False)
            if len(test_df) > 0:
                return test_symbol
        except Exception:
            pass
            
    # 3. Yahoo Finance Search Fallback
    try:
        search = yf.Search(company_name, max_results=5)
        if search.quotes:
            for quote in search.quotes:
                symbol = quote.get('symbol', '')
                if symbol:
                    test_df = yf.download(symbol, period="1d", progress=False)
                    if len(test_df) > 0:
                        return symbol.upper()
    except Exception:
        pass
        
    return None

def fetch_and_generate_chart(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if len(df) == 0:
        return df, None, 0
        
    # 1. Flatten MultiIndex columns and drop duplicate column labels
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
        
    # 2. Forward-fill missing price data and drop remaining NaNs
    df = df.ffill().bfill()
    df = df.dropna(subset=['Close'])
    
    if len(df) == 0:
        return df, None, 0

    # 3. Explicitly convert OHLCV columns to 1D Pandas Series
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.Series(df[col].values.flatten(), index=df.index).astype(float)
            
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0)
        valid_volume = bool(df['Volume'].sum() > 0)
    else:
        valid_volume = False

    close_s = df['Close']
    low_s = df['Low']
    high_s = df['High']

    df['SMA_5'] = close_s.rolling(window=5).mean()
    df['SMA_15'] = close_s.rolling(window=15).mean()
    
    buy_signals, sell_signals = [], []
    sma_5, sma_15 = df['SMA_5'], df['SMA_15']
    
    for i in range(len(df)):
        if pd.isna(sma_5.iloc[i]) or pd.isna(sma_15.iloc[i]):
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
            continue
            
        if sma_5.iloc[i] > sma_15.iloc[i] and sma_5.iloc[i-1] <= sma_15.iloc[i-1]:
            buy_signals.append(low_s.iloc[i] * 0.96)
            sell_signals.append(np.nan)
        elif sma_5.iloc[i] < sma_15.iloc[i] and sma_5.iloc[i-1] >= sma_15.iloc[i-1]:
            buy_signals.append(np.nan)
            sell_signals.append(high_s.iloc[i] * 1.04)
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)
            
    apds = [
        mpf.make_addplot(df['SMA_5'], color='#00d2ff', width=2),
        mpf.make_addplot(df['SMA_15'], color='#ff007f', width=2)
    ]
    
    if not all(np.isnan(buy_signals)):
        apds.append(mpf.make_addplot(buy_signals, type='scatter', markersize=220, marker='^', color='#00ff88'))
    if not all(np.isnan(sell_signals)):
        apds.append(mpf.make_addplot(sell_signals, type='scatter', markersize=220, marker='v', color='#ff3366'))
        
    os.makedirs('charts', exist_ok=True)
    chart_path = f"charts/{ticker.replace('.', '_')}_candlestick_signals.png"
    
    custom_style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        facecolor='#0e1117',
        figcolor='#0e1117',
        gridcolor='#1e2638',
        gridstyle='--',
        y_on_right=True
    )
    
    mpf.plot(
        df, type='candle', style=custom_style, volume=valid_volume,
        addplot=apds, savefig=dict(fname=chart_path, dpi=150, bbox_inches='tight', facecolor='#0e1117'),
        figsize=(11, 5)
    )
    
    latest_trend = 1 if sma_5.iloc[-1] > sma_15.iloc[-1] else -1
    return df, chart_path, latest_trend

def get_dynamic_competitors(active_ticker):
    ticker_clean = active_ticker.upper()
    is_indian = ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO")
    
    try:
        t = yf.Ticker(ticker_clean)
        sector = t.info.get('sector', '')
        
        if is_indian:
            sector_fallbacks = {
                'Industrials': ['LT.NS', 'BHEL.NS'],
                'Technology': ['TCS.NS', 'INFY.NS', 'WIPRO.NS'],
                'Financial Services': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS'],
                'Energy': ['RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS']
            }
            peers = sector_fallbacks.get(sector, ['LT.NS', 'RELIANCE.NS', 'TCS.NS'])
        else:
            sector_fallbacks = {
                'Technology': ['MSFT', 'AAPL', 'GOOGL', 'NVDA', 'AMD'],
                'Energy': ['XOM', 'CVX', 'COP']
            }
            peers = sector_fallbacks.get(sector, ['SPY', 'QQQ'])
    except Exception:
        peers = ['LT.NS', 'RELIANCE.NS'] if is_indian else ['SPY', 'QQQ']
        
    peers = [p for p in peers if p != ticker_clean][:4]
    insights, peer_returns = [], []
    
    for peer in peers:
        try:
            peer_df = yf.download(peer, period="1mo", progress=False)
            if isinstance(peer_df.columns, pd.MultiIndex):
                peer_df.columns = peer_df.columns.get_level_values(0)
            peer_df = peer_df.loc[:, ~peer_df.columns.duplicated()]
            
            # Clean missing values
            peer_df = peer_df.ffill().bfill().dropna(subset=['Close'])
            
            if len(peer_df) >= 5:
                peer_close = pd.Series(peer_df['Close'].values.flatten(), index=peer_df.index).astype(float)
                peer_df['SMA_5'] = peer_close.rolling(window=5).mean()
                peer_df['SMA_15'] = peer_close.rolling(window=15).mean()
                
                curr_sym = get_currency_symbol(peer)
                current_price = float(peer_close.iloc[-1])
                price_5d_ago = float(peer_close.iloc[-5])
                
                if np.isnan(current_price) or np.isnan(price_5d_ago) or price_5d_ago == 0:
                    continue

                pct_change_5d = ((current_price - price_5d_ago) / price_5d_ago) * 100
                is_bullish = peer_df['SMA_5'].iloc[-1] > peer_df['SMA_15'].iloc[-1]
                
                insights.append({
                    'ticker': peer, 'price': round(current_price, 2),
                    'currency': curr_sym, 'change_5d': round(pct_change_5d, 2),
                    'is_bullish': is_bullish
                })
                peer_returns.append(pct_change_5d)
        except Exception:
            continue
            
    insights = sorted(insights, key=lambda x: x['change_5d'], reverse=True)
    avg_peer_return = np.mean(peer_returns) if peer_returns else 0.0
    return insights, round(float(avg_peer_return), 2), bool(avg_peer_return < -1.0)