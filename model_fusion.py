import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
from data_pipeline import COMMON_COMPANY_ALIASES

_nlp_tokenizer = None
_nlp_model = None
_ner_pipeline = None

HIGH_IMPACT_KEYWORDS = [
    'earnings', 'revenue', 'profit', 'loss', 'guidance', 'bankruptcy', 
    'lawsuit', 'antitrust', 'sec', 'fda', 'ban', 'sanction', 'recall', 
    'merger', 'acquisition', 'takeover', 'slash', 'delays', 'contract', 'sell-off', 'plunges'
]

LOW_IMPACT_KEYWORDS = ['routine', 'patch', 'update', 'meeting', 'schedule', 'minor']

def get_nlp_models():
    global _nlp_tokenizer, _nlp_model, _ner_pipeline
    if _nlp_tokenizer is None or _nlp_model is None:
        _nlp_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _nlp_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    if _ner_pipeline is None:
        _ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return _nlp_tokenizer, _nlp_model, _ner_pipeline

def analyze_sentiment(headline):
    tokenizer, model, _ = get_nlp_models()
    inputs = tokenizer(headline, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions[0].numpy()

def extract_company_entity(headline):
    headline_upper = headline.upper()
    
    for alias in COMMON_COMPANY_ALIASES.keys():
        if alias == headline_upper.strip() or alias in headline_upper:
            return alias.title()
            
    try:
        _, _, ner = get_nlp_models()
        entities = ner(headline)
        org_tokens = [ent['word'].replace('##', '') for ent in entities if ent['entity_group'] == 'ORG']
        if org_tokens:
            return " ".join(org_tokens)
    except Exception:
        pass
        
    words = headline.split()
    capitalized_words = [
        word.strip(".,!?\"'()") for word in words 
        if word[0].isupper() and word.upper() not in [
            'THE', 'A', 'AN', 'IN', 'ON', 'AT', 'FOR', 'NEW', 'STOCK', 'NEWS', 
            'BUY', 'SELL', 'SHARES', 'PROFIT', 'LOSS'
        ]
    ]
    return " ".join(capitalized_words) if capitalized_words else headline

def calculate_headline_priority(headline, sentiment_probs):
    headline_lower = headline.lower()
    max_conf = max(sentiment_probs)
    
    if any(kw in headline_lower for kw in HIGH_IMPACT_KEYWORDS) or max_conf > 0.85:
        return "HIGH", 2.0, "🚨 High Priority Impact"
    elif any(kw in headline_lower for kw in LOW_IMPACT_KEYWORDS) or max_conf < 0.50:
        return "LOW", 0.5, "ℹ️ Low Priority Impact"
    else:
        return "MEDIUM", 1.0, "⚡ Medium Priority Impact"

def generate_trade_recommendation(cumulative_probs, technical_trend):
    pos_pct, neg_pct, neu_pct = cumulative_probs
    
    if pos_pct > 0.52 and technical_trend == 1:
        return "STRONG BUY 🟢", f"Positive sentiment ({round(pos_pct*100, 1)}%) aligns with a Bullish chart trend."
    elif pos_pct > 0.48 and technical_trend == -1:
        return "HOLD / WATCH 🟡", f"Positive sentiment ({round(pos_pct*100, 1)}%) detected, but chart remains Bearish. Await breakout."
    elif neg_pct > 0.48 and technical_trend == -1:
        return "STRONG SELL 🔴", f"Negative sentiment ({round(neg_pct*100, 1)}%) reinforced by a Bearish chart trend."
    elif neg_pct > 0.45 and technical_trend == 1:
        return "HOLD / RISK WARNING 🟡", f"Chart shows upward momentum, but news sentiment is Negative ({round(neg_pct*100, 1)}%). Risk of reversal."
    elif neu_pct > 0.45:
        return ("BUY / MODERATE 🟢" if technical_trend == 1 else "HOLD / NEUTRAL 🟡"), "Dominant neutral sentiment with no strong catalyst."
    else:
        return "HOLD 🟡", "Mixed sentiment signals across headline history."