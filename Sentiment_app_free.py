# sentiment_app_free.py
# Full app: sentiment + option probability + smart strike suggestions
# Dependencies: flask, requests, pandas, numpy, python-dateutil, yfinance, scipy

from flask import Flask, render_template_string, request
import requests, time, math
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import yfinance as yf
from scipy.stats import norm

app = Flask(__name__)

# ---------- CONFIG ----------
NSE_OPTION_CHAIN_URL = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/plain, */*"
}
TICKER_MAP = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK"
}
DEFAULT_MINUTES = 180
# ----------------------------

# ---------- Data helpers ----------
def get_recent_bars_yahoo(ticker, minutes=DEFAULT_MINUTES):
    """Fetch recent intraday bars from yfinance; try 1m then 5m."""
    try:
        df = yf.download(tickers=ticker, period="7d", interval="1m", progress=False)
        if df is None or df.empty:
            raise ValueError
        df = df[["Close"]].rename(columns={"Close": "close"})
        df = df.iloc[-minutes:]
        df.index = pd.to_datetime(df.index)
        if df.empty:
            raise ValueError
        return df
    except Exception:
        try:
            df = yf.download(tickers=ticker, period="30d", interval="5m", progress=False)
            df = df[["Close"]].rename(columns={"Close": "close"})
            df = df.iloc[-int(minutes/5):]
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            return None

_CHAIN_CACHE = {"ts": 0.0, "data": None}
_CHAIN_TTL = 45  # seconds, tuneable

def fetch_option_chain_nifty(force_refresh: bool = False):
    """
    Fetch NSE NIFTY option-chain JSON with lightweight caching, retry and session warm-up.
    - Returns JSON dict on success, or None on failure (returns stale cached data if available).
    - force_refresh: if True, bypasses cache and attempts to fetch fresh data.
    """
    now = time.time()
    # Return cached copy if fresh and not forcing refresh
    if not force_refresh and (_CHAIN_CACHE["data"] is not None) and (now - _CHAIN_CACHE["ts"] < _CHAIN_TTL):
        return _CHAIN_CACHE["data"]

    s = requests.Session()
    # Headers tuned for NSE access; keep these as-is
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nseindia.com/option-chain",
        "Origin": "https://www.nseindia.com"
    }

    # Try a small number of retries with backoff
    backoff = [0.5, 1.0, 2.0]  # seconds
    last_exception = None
    for attempt, wait in enumerate(backoff, start=1):
        try:
            # Warm-up request required by NSE (establish cookies & headers)
            s.get("https://www.nseindia.com", headers=headers, timeout=5)
            time.sleep(0.3)  # small pause to be polite
            resp = s.get(NSE_OPTION_CHAIN_URL, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                # store in cache
                _CHAIN_CACHE["data"] = data
                _CHAIN_CACHE["ts"] = time.time()
                return data
            else:
                # non-200: keep trying (but save response text to logs if needed)
                last_exception = Exception(f"NSE returned status {resp.status_code}")
        except Exception as e:
            last_exception = e

        # wait before next attempt (polite)
        time.sleep(wait)

    # If all retries failed, return stale cache if available
    if _CHAIN_CACHE["data"] is not None:
        return _CHAIN_CACHE["data"]

    # final fallback: None (caller should handle gracefully)
    # Optional: you can log last_exception here for debugging

# ---------- Indicator & sentiment ----------
def compute_indicators(bars):
    """Return indicators dict or None if bars missing."""
    if bars is None or bars.empty:
        return None
    close = bars['close'].astype(float)
    last_close = float(close.iloc[-1])
    ema_fast = float(close.ewm(span=8, adjust=False).mean().iloc[-1])
    ema_slow = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    trend = "Bullish" if ema_fast > ema_slow else "Bearish"
    diff = abs(ema_fast - ema_slow)
    trend_strength = 1 if diff < 0.5 else 2 if diff < 1.5 else 3
    atr = float(close.diff().abs().rolling(14, min_periods=3).mean().iloc[-1])
    window = min(60, len(close))
    local_max = float(close.rolling(window, min_periods=3).max().iloc[-1])
    res_dist_pct = (local_max - last_close) / last_close * 100
    return {
        "last_close": last_close,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "trend": trend,
        "trend_strength": trend_strength,
        "atr": atr,
        "resistance": local_max,
        "resistance_dist_pct": res_dist_pct
    }

def compute_option_metrics(oc):
    """Compute PCR (put OI / call OI) and sums."""
    if oc is None:
        return {"pcr": 1.0, "oi_call": 1.0, "oi_put": 1.0}
    calls = puts = 0.0
    try:
        for row in oc.get("records", {}).get("data", []):
            ce = row.get("CE"); pe = row.get("PE")
            if ce: calls += float(ce.get("openInterest", 0) or 0)
            if pe: puts += float(pe.get("openInterest", 0) or 0)
    except Exception:
        pass
    return {"pcr": puts / (calls + 1e-9), "oi_call": calls, "oi_put": puts}

def market_mood(ind, opt):
    """Composite mood -1..1"""
    if ind is None:
        return 0.0
    pcr = opt.get("pcr", 1.0)
    pcr_s = np.clip((1 - pcr) * 2, -1, 1)
    tr_s = 0.5 if ind['trend'] == "Bullish" else -0.5
    vol_s = 0.2 if ind['atr'] < 6 else -0.2
    return float(np.clip((pcr_s + tr_s + vol_s) / 2, -1, 1))

def aggregate(ind, opt, mood):
    """Build factor table and net_score/verdict."""
    if ind is None:
        return {"table": [], "net_score": 0.0, "verdict": "No data"}
    table = []
    table.append({"Factor": "Price trend", "Bias": ind['trend'], "Strength": "✅" * ind['trend_strength']})
    if ind['resistance_dist_pct'] < 0.6:
        table.append({"Factor": "Resistance proximity", "Bias": "Neutral-to-bearish", "Strength": "⚠️"})
    else:
        table.append({"Factor": "Resistance proximity", "Bias": "Neutral", "Strength": ""})
    pcr = opt.get("pcr", 1.0)
    if mood > 0.6:
        sent_bias, sent_strength = "Bullish but overheated", "✅⚠️"
    elif pcr < 0.95:
        sent_bias, sent_strength = "Bullish", "✅"
    elif pcr > 1.05:
        sent_bias, sent_strength = "Bearish", "⚠️"
    else:
        sent_bias, sent_strength = "Neutral", ""
    table.append({"Factor": "Sentiment (MMI/PCR)", "Bias": sent_bias, "Strength": sent_strength})
    table.append({"Factor": "Global cues", "Bias": "Slightly positive", "Strength": "✅"})
    table.append({"Factor": "Volatility", "Bias": "Low (bullish but fragile)", "Strength": "✅⚠️" if ind['atr'] < 4 else "⚠️"})
    trend_num = 0.5 if ind['trend'] == "Bullish" else -0.5
    pcr_num = 1.0 - pcr
    global_num = 0.2
    vol_num = 0.2 if ind['atr'] < 4 else -0.2
    raw = trend_num + pcr_num + global_num + vol_num + (0.5 * mood)
    net_score = float(np.clip(raw, -3, 3))
    if net_score >= 1.2:
        verdict = "Bullish"
    elif net_score >= 0.2:
        verdict = "Mildly bullish, but with caution."
    elif net_score >= -0.5:
        verdict = "Neutral to cautious"
    else:
        verdict = "Bearish"
    return {"table": table, "net_score": round(net_score, 2), "verdict": verdict}

# ---------- Option probability helpers ----------
def days_to_next_expiry(oc):
    """Return days to next expiry from option chain JSON; fallback 7."""
    try:
        exps = oc.get("records", {}).get("expiryDates", [])
        if not exps:
            return 7.0
        now = datetime.utcnow()
        for e in exps:
            dt = datetime.fromisoformat(e)
            if dt > now:
                return (dt - now).total_seconds() / 86400.0
        dt = datetime.fromisoformat(exps[-1])
        return (dt - now).total_seconds() / 86400.0
    except Exception:
        return 7.0

def prob_ITM_bs(S, K, r, sigma, T):
    """Black-Scholes risk-neutral probability that S_T > K."""
    if sigma <= 0 or T <= 0:
        return 1.0 if S > K else 0.0
    d2 = (math.log(S / K) + (r - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return float(1.0 - norm.cdf(d2))

def realized_vol_from_spot(bars):
    """Estimate annualized realized vol from bars (log returns)."""
    if bars is None or bars.empty:
        return None
    close = bars['close'].astype(float)
    logr = np.log(close / close.shift(1)).dropna()
    if logr.empty:
        return None
    # approximate: assume sample frequency is minutes; annualize by sqrt(samples_per_day * trading_days)
    # find average seconds per sample
    delta_seconds = (close.index[-1] - close.index[0]).total_seconds() if len(close) > 1 else 60.0
    seconds_per_sample = delta_seconds / max(1, len(close))
    samples_per_day = max(1, 24 * 3600 / seconds_per_sample)
    annual_factor = np.sqrt(samples_per_day * 252.0)
    sigma = float(logr.std() * annual_factor)
    return sigma

def estimate_option_probability(strike, opt_type, spot_symbol, oc, bars_for_vol=None, r=0.06):
    """
    Estimate:
      - prob_itm: probability S_T > K (using IV if available else historical)
      - prob_pop: probability S_T > K + premium (if premium available)
      - returns dict with spot, dte, sigma, premium (if known)
    """
    # spot
    spot = None
    try:
        df_spot = yf.download(tickers=spot_symbol, period="1d", interval="1m", progress=False)
        spot = float(df_spot['Close'].iloc[-1])
    except Exception:
        if bars_for_vol is not None:
            spot = float(bars_for_vol['close'].iloc[-1])
    if spot is None:
        return {"error": "no spot data"}

    # DTE
    dte_days = days_to_next_expiry(oc) if oc else 7.0
    T = max(0.0001, dte_days / 365.0)

    # find option row in chain
    iv = None
    premium = None
    opt_row = None
    if oc:
        for row in oc.get("records", {}).get("data", []):
            try:
                if float(row.get("strikePrice", 0)) == float(strike):
                    block = row.get(opt_type)
                    if block:
                        opt_row = block
                        iv_val = block.get("impliedVolatility") or block.get("impliedVolatilityValue") or block.get("iv")
                        if iv_val is not None:
                            iv_float = float(iv_val)
                            iv = iv_float / 100.0 if iv_float > 1 else iv_float
                        lp = block.get("lastPrice") or block.get("lastTradedPrice") or block.get("last")
                        try:
                            premium = float(lp) if lp is not None else None
                        except:
                            premium = None
                    break
            except Exception:
                continue

    # sigma selection
    method = "implied" if iv is not None else "historical"
    if iv is not None:
        sigma = iv
    else:
        sigma = realized_vol_from_spot(bars_for_vol) or 0.6

    prob_itm = prob_ITM_bs(S=spot, K=float(strike), r=float(r), sigma=float(sigma), T=float(T))
    prob_pop = None
    if premium is not None:
        breakeven = float(strike) + float(premium) if opt_type == 'CE' else float(strike) - float(premium)
        prob_pop = prob_ITM_bs(S=spot, K=breakeven, r=float(r), sigma=float(sigma), T=float(T))

    return {
        "spot": round(float(spot), 2),
        "dte_days": round(float(dte_days), 2),
        "sigma": round(float(sigma), 4),
        "prob_itm": round(prob_itm, 4),
        "prob_pop": round(prob_pop, 4) if prob_pop is not None else None,
        "premium": round(float(premium), 2) if premium is not None else None,
        "method": method,
        "opt_row": opt_row
    }

# ---------- Smart suggestion engine ----------
def generate_option_suggestions(
    focus="NIFTY",
    side_preference=None,
    num_strikes_each_side=8,
    top_k=5,
    min_pop_threshold=0.30
):
    oc = fetch_option_chain_nifty()
    if oc is None:
        return {"error": "option-chain unavailable (NSE blocked). Try again or use broker API."}
    # try to get spot from chain
    try:
        spot = float(oc.get("records", {}).get("underlyingValue", None) or 0.0)
        if spot == 0.0:
            raise ValueError
    except Exception:
        bars_tmp = get_recent_bars_yahoo("^NSEI", minutes=60)
        spot = float(bars_tmp['close'].iloc[-1]) if (bars_tmp is not None and not bars_tmp.empty) else None
    if spot is None:
        return {"error": "spot unavailable"}

    strikes = sorted({float(row.get("strikePrice")) for row in oc.get("records", {}).get("data", []) if row.get("strikePrice") is not None})
    if not strikes:
        return {"error": "no strikes found"}

    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot))
    # sentiment direction
    bars_all = get_recent_bars_yahoo("^NSEI", minutes=180)
    if bars_all is None:
        market_direction = "NEUTRAL"
    else:
        ind_main = compute_indicators(bars_all)
        opt_main = compute_option_metrics(oc)
        agg_main = aggregate(ind_main, opt_main, market_mood(ind_main, opt_main))
        market_direction = "BULL" if agg_main["net_score"] >= 0.2 else ("BEAR" if agg_main["net_score"] <= -0.2 else "NEUTRAL")

    if side_preference is None:
        if market_direction == "BULL":
            want = ["CE"]
        elif market_direction == "BEAR":
            want = ["PE"]
        else:
            want = ["CE", "PE"]
    else:
        want = ["CE"] if side_preference == "BUY_CALL" else ["PE"]

    candidates = []
    start = max(0, atm_idx - num_strikes_each_side)
    end = min(len(strikes) - 1, atm_idx + num_strikes_each_side)
    examine = strikes[start:end+1]
    dte_days = days_to_next_expiry(oc) or 7.0

    for K in examine:
        for typ in want:
            # find row
            row_block = None
            for row in oc.get("records", {}).get("data", []):
                try:
                    if float(row.get("strikePrice", 0)) == float(K):
                        row_block = row.get(typ)
                        break
                except Exception:
                    continue
            if not row_block:
                continue
            # premium & iv
            premium = None
            iv = None
            try:
                premium = row_block.get("lastPrice") or row_block.get("lastTradedPrice") or row_block.get("last")
                premium = float(premium) if premium is not None else None
            except:
                premium = None
            iv_raw = row_block.get("impliedVolatility") or row_block.get("impliedVolatilityValue") or row_block.get("iv")
            try:
                iv = float(iv_raw) / 100.0 if iv_raw is not None and float(iv_raw) > 1 else (float(iv_raw) if iv_raw is not None else None)
            except:
                iv = None

            # sigma fallback
            sigma = iv
            if sigma is None:
                bars_vol = get_recent_bars_yahoo("^NSEI", minutes=7*24*60)
                sigma = realized_vol_from_spot(bars_vol) or 0.6

            # compute probabilities
            T = max(0.0001, dte_days / 365.0)
            prob_itm = prob_ITM_bs(S=spot, K=K, r=0.06, sigma=sigma, T=T)
            pop = None
            if premium is not None:
                breakeven = K + premium if typ == "CE" else K - premium
                pop = prob_ITM_bs(S=spot, K=breakeven, r=0.06, sigma=sigma, T=T)

            if premium is None:
                continue

            entry = premium
            sl_price = round(entry * (1 - 0.12), 2)
            tgt1 = round(entry * (1 + 0.20), 2)
            tgt2 = round(entry * (1 + 0.35), 2)
            tgt3 = round(entry * (1 + 0.50), 2)
            rr1 = (tgt1 - entry) / max(1e-9, (entry - sl_price))
            reason = f"Trend={market_direction}; spot={spot:.2f}; K={K}; IV={'{:.2%}'.format(sigma) if sigma else 'NA'}"
            candidates.append({
                "strike": int(K),
                "type": typ,
                "spot": round(spot, 2),
                "entry": round(entry, 2),
                "premium": round(entry, 2),
                "iv": round(sigma, 4) if sigma else None,
                "dte_days": round(dte_days, 2),
                "prob_itm": round(prob_itm, 4),
                "pop": round(pop, 4) if pop is not None else None,
                "stop_loss": sl_price,
                "targets": [tgt1, tgt2, tgt3],
                "rr1": round(rr1, 2),
                "reason": reason
            })

    # filter & ranking
    filtered = []
    for c in candidates:
        if c["entry"] is None:
            continue
        if c.get("pop") is None or c.get("pop") < min_pop_threshold:
            continue
        if c["entry"] <= 0.2:
            continue
        if c["entry"] > (0.5 * c["spot"]):
            continue
        filtered.append(c)

    # sort by pop desc then prob_itm
    filtered = sorted(filtered, key=lambda x: (x.get("pop", 0), x.get("prob_itm", 0)), reverse=True)
    return {"market_direction": market_direction, "candidates": filtered[:top_k], "all_candidates_count": len(candidates)}

# ---------- Routes ----------
@app.route('/sentiment')
def sentiment_route():
    bars = get_recent_bars_yahoo("^NSEI")
    ind = compute_indicators(bars) if bars is not None else None
    oc = fetch_option_chain_nifty()
    opt = compute_option_metrics(oc)
    mood = market_mood(ind, opt) if ind is not None else 0.0
    agg = aggregate(ind, opt, mood) if ind is not None else {"table": [], "net_score": 0.0, "verdict": "No data"}

    suggestions = []
    for name, tick in TICKER_MAP.items():
        b = get_recent_bars_yahoo(tick)
        ind2 = compute_indicators(b)
        action = "NO DATA" if ind2 is None else ("BUY" if ind2['trend'] == "Bullish" and opt.get('pcr', 1.0) <= 1.0 else "NO ACTION")
        suggestions.append({"symbol": name, "action": action, "trend": ind2['trend'] if ind2 is not None else "NA"})

    out = pd.Series({
        "updated_at": datetime.utcnow().isoformat(),
        "indicators": ind,
        "option_metrics": opt,
        "market_mood": mood,
        "result": agg,
        "suggestions": suggestions
    }).to_json()

    return render_template_string("""
    <html><body>
      <h2>Sentiment for Tomorrow — NIFTY 50</h2>
      <a href="/sentiment"><button>Refresh</button></a>
      <h3>Result: {{agg.verdict}} (Net score: {{agg.net_score}})</h3>

      <table border=1 cellpadding=6>
        <tr><th>Factor</th><th>Bias</th><th>Strength</th></tr>
        {% for r in agg.table %}
        <tr><td>{{r.Factor}}</td><td>{{r.Bias}}</td><td>{{r.Strength}}</td></tr>
        {% endfor %}
      </table>

      <h3>Spot Suggestions (Basic)</h3>
      <table border=1 cellpadding=6>
        <tr><th>Index</th><th>Action</th><th>Trend</th></tr>
        {% for s in suggestions %}
        <tr><td>{{s.symbol}}</td><td>{{s.action}}</td><td>{{s.trend}}</td></tr>
        {% endfor %}
      </table>

      <h3>Option Tools</h3>
      <p>Estimate individual option: /option?strike=25600&type=CE&symbol=^NSEI</p>
      <p>Smart suggestions: /suggest (try /suggest?min_pop=0.35&k=5)</p>

      <h4>Raw JSON</h4><pre>{{out}}</pre>
    </body></html>
    """, agg=agg, suggestions=suggestions, out=out)

@app.route('/option')
def option_route():
    strike = request.args.get('strike', None)
    opt_type = request.args.get('type', 'CE').upper()
    symbol = request.args.get('symbol', '^NSEI')
    if strike is None:
        return "<p>Please provide ?strike=XXXXX (e.g. /option?strike=25600&type=CE)</p>"
    oc = fetch_option_chain_nifty()
    bars = get_recent_bars_yahoo(symbol, minutes=360)
    res = estimate_option_probability(strike, opt_type, symbol, oc, bars_for_vol=bars, r=0.06)
    if 'error' in res:
        return f"<p>Error: {res['error']}</p>"
    prob_pct = int(res['prob_itm'] * 100)
    pop_pct = int(res['prob_pop'] * 100) if res.get('prob_pop') is not None else None
    method = res.get('method')
    premium = res.get('premium')
    return f"""
    <html><body>
      <h2>Option Probability — {symbol} {strike}{opt_type}</h2>
      <p>Spot: {res['spot']}</p>
      <p>DTE: {res['dte_days']} days</p>
      <p>Vol (sigma): {res['sigma']}</p>
      <p>P(ITM): <b>{prob_pct}%</b> (method={method})</p>
      <p>POP (breakeven): <b>{pop_pct}%</b></p>
      <p>Premium (last): {premium}</p>
      <p><a href="/sentiment">Back</a></p>
    </body></html>
    """

@app.route('/suggest')
def suggest_route():
    side = request.args.get("side", None)
    min_pop = float(request.args.get("min_pop", 0.30))
    k = int(request.args.get("k", 5))
    prefs = None
    if side == "BUY_CALL": prefs = "BUY_CALL"
    if side == "BUY_PUT": prefs = "BUY_PUT"
    res = generate_option_suggestions(side_preference=prefs, min_pop_threshold=min_pop, top_k=k)
    if "error" in res:
        return f"<p>Error: {res['error']}</p><p>Try /sentiment to verify chain access.</p>"
    html = "<h2>Smart Option Suggestions (based on sentiment)</h2>"
    html += f"<p>Market Direction: <b>{res['market_direction']}</b></p>"
    if not res['candidates']:
        html += "<p>No candidates passed filters (try lowering min_pop or expand strike range).</p>"
    else:
        html += "<table border=1 cellpadding=6><tr><th>Index</th><th>Type</th><th>Strike</th><th>Entry</th><th>POP</th><th>P(ITM)</th><th>SL</th><th>T1/T2/T3</th><th>Reason</th></tr>"
        for c in res['candidates']:
            pop_display = f"{int(c['pop']*100)}%" if c.get('pop') else 'N/A'
            html += f"<tr><td>NIFTY</td><td>{c['type']}</td><td>{c['strike']}</td><td>{c['entry']}</td>"
            html += f"<td>{pop_display}</td><td>{int(c['prob_itm']*100)}%</td>"
            html += f"<td>{c['stop_loss']}</td><td>{c['targets'][0]}/{c['targets'][1]}/{c['targets'][2]}</td>"
            html += f"<td>{c['reason']}</td></tr>"
        html += "</table>"
    html += "<p><a href='/sentiment'>Back</a></p>"
    return html

@app.route('/')
def home():
    return """
    <html><body>
    <h1>Mini Sentiment & Option Helper</h1>
    <a href='/sentiment'><button>Sentiment</button></a>
    <a href='/suggest'><button>Smart Suggestions</button></a>
    </body></html>
    """

if __name__ == '__main__':
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

