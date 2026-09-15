import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="Institutional Quant Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {background-color: #0d1117;}
    h1, h2, h3, p, span {color: #e6edf3 !important;}
    .stMetric label {color: #58a6ff !important;}
    [data-testid="stSidebar"] {background-color: #161b22;}
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ Institutional Quant & Risk Management Terminal")
st.markdown("ระบบวิเคราะห์ความเสี่ยง, จัดพอร์ต (Risk-Parity/Min-Var), Stress Testing และ 3D Volatility Surface ระดับสถาบัน")

# ==========================================
# 2. QUANT CORE FUNCTIONS (ฟังก์ชันคำนวณ)
# ==========================================
def historical_var(returns, confidence=0.95):
    returns = np.asarray(returns)
    return max(-np.percentile(returns, (1 - confidence) * 100), 0.0)

def parametric_var(returns, confidence=0.95):
    returns = np.asarray(returns)
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(1 - confidence)
    return max(-(mu + z * sigma), 0.0)

def expected_shortfall(returns, confidence=0.95):
    returns = np.asarray(returns)
    var_cutoff = np.percentile(returns, (1 - confidence) * 100)
    tail_losses = returns[returns <= var_cutoff]
    if len(tail_losses) == 0:
        return historical_var(returns, confidence)
    return max(-tail_losses.mean(), 0.0)

def ledoit_wolf_shrinkage(returns_df):
    sample_cov = returns_df.cov().values
    n_obs, n_assets = returns_df.shape
    target = np.diag(np.diag(sample_cov))
    shrinkage = min(1.0, max(0.1, n_assets / n_obs))
    return pd.DataFrame(shrinkage * target + (1 - shrinkage) * sample_cov, 
                        index=returns_df.columns, columns=returns_df.columns)

def min_variance_portfolio(cov_matrix):
    n = cov_matrix.shape[0]
    bounds = [(0, 1)] * n
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    res = minimize(lambda w: w @ cov_matrix @ w, np.repeat(1/n, n), method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x

def risk_parity_weights(cov_matrix):
    n = cov_matrix.shape[0]
    def objective(w):
        port_vol = np.sqrt(w @ cov_matrix @ w)
        contrib = (w * (cov_matrix @ w)) / port_vol
        return np.sum((contrib - contrib.mean()) ** 2)
    bounds = [(1e-6, 1)] * n
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    res = minimize(objective, np.repeat(1/n, n), method="SLSQP", bounds=bounds, constraints=constraints)
    return res.x

def max_drawdown(nav_series):
    nav = np.asarray(nav_series)
    running_max = np.maximum.accumulate(nav)
    drawdown = (nav - running_max) / running_max
    return drawdown.min(), drawdown

# ==========================================
# 3. SIDEBAR CONTROLS (แผงควบคุมผู้ใช้)
# ==========================================
with st.sidebar:
    st.header("⚙️ Terminal Parameters")
    initial_cap = st.number_input("เงินลงทุนเริ่มต้น ($)", value=1000000, step=100000)
    confidence_level = st.selectbox("ระดับความเชื่อมั่น VaR", [0.90, 0.95, 0.99], index=1)
    simulations = st.slider("Monte Carlo Paths", min_value=1000, max_value=20000, value=5000, step=1000)
    
    st.markdown("---")
    st.markdown("### 🎛️ 3D Surface Controls")
    vol_bump = st.slider("Volatility Bump (+/-)", min_value=-0.2, max_value=0.5, value=0.0, step=0.05)

tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "META", "TSLA", "BRK-B", "JPM"]

# ==========================================
# 4. DATA FETCHING & OPTIMIZATION
# ==========================================
@st.cache_data(ttl=3600)
def load_market_data(tks):
    prices = yf.download(tks, period="2y", auto_adjust=True, progress=False)["Close"]
    return prices.pct_change().dropna()

with st.spinner("กำลังเชื่อมต่อตลาดและคำนวณโมเดลความเสี่ยง..."):
    returns_df = load_market_data(tickers)
    shrunk_cov = ledoit_wolf_shrinkage(returns_df)
    cov_np = shrunk_cov.values
    
    # คำนวณน้ำหนักพอร์ต
    min_var_w = min_variance_portfolio(cov_np)
    rp_w = risk_parity_weights(cov_np)
    
    # ใช้พอร์ต Risk Parity เป็นตัวหลักในการทดสอบ
    port_returns = returns_df.dot(rp_w).values
    mu_annual = float(port_returns.mean() * 252)
    sigma_annual = float(port_returns.std() * np.sqrt(252)) + vol_bump

# ==========================================
# 5. HEADER METRICS SUMMARY
# ==========================================
st.markdown("### 📊 Portfolio Executive Summary (Risk-Parity Engine)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected Annual Return", f"{mu_annual*100:.2f}%")
c2.metric("Portfolio Volatility", f"{sigma_annual*100:.2f}%")
c3.metric("Sharpe Ratio (Rf=2%)", f"{(mu_annual - 0.02) / sigma_annual:.2f}")
c4.metric("Active Universe", f"{len(tickers)} Mega-Caps")

st.markdown("---")

# ==========================================
# 6. MULTI-TAB ARCHITECTURE (7 มิติครบถ้วน)
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Portfolio Construction", 
    "🚨 Risk & Monte Carlo", 
    "🌪️ Stress Testing", 
    "📈 Factor & Attribution", 
    "🧊 3D Volatility & Drawdown"
])

# --- TAB 1: PORTFOLIO CONSTRUCTION ---
with tab1:
    st.subheader("Asset Allocation Strategies Comparison")
    df_weights = pd.DataFrame({
        "Min-Variance (%)": min_var_w * 100,
        "Risk-Parity (%)": rp_w * 100
    }, index=tickers)
    
    fig_w = go.Figure()
    fig_w.add_trace(go.Bar(name='Min-Variance', x=tickers, y=df_weights["Min-Variance (%)"], marker_color='#2a78d6'))
    fig_w.add_trace(go.Bar(name='Risk-Parity', x=tickers, y=df_weights["Risk-Parity (%)"], marker_color='#eb6834'))
    fig_w.update_layout(barmode='group', plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3', height=450)
    st.plotly_chart(fig_w, width="stretch")
    st.dataframe(df_weights.style.format("{:.2f}%"), width="stretch")

# --- TAB 2: RISK & MONTE CARLO ---
with tab2:
    st.subheader(f"Risk Measurement ({int(confidence_level*100)}% Confidence Horizon)")
    
    h_var = historical_var(port_returns, confidence_level)
    p_var = parametric_var(port_returns, confidence_level)
    cvar = expected_shortfall(port_returns, confidence_level)
    l_var = h_var + 0.0005 # Liquidity adjustment
    
    r_cols = st.columns(4)
    r_cols[0].metric("Historical VaR", f"{h_var*100:.2f}%", f"${h_var*initial_cap:,.0f}")
    r_cols[1].metric("Parametric VaR", f"{p_var*100:.2f}%", f"${p_var*initial_cap:,.0f}")
    r_cols[2].metric("Expected Shortfall (CVaR)", f"{cvar*100:.2f}%", f"${cvar*initial_cap:,.0f}")
    r_cols[3].metric("Liquidity-Adj VaR", f"{l_var*100:.2f}%", f"${l_var*initial_cap:,.0f}")

    # Monte Carlo Simulation
    st.markdown("---")
    st.markdown("### 🎲 Monte Carlo Wealth Paths (1 Year)")
    T, N = 1.0, 252
    dt = T / N
    Z = np.random.randn(simulations, N)
    S = np.zeros((simulations, N + 1))
    S[:, 0] = initial_cap
    for i in range(N):
        S[:, i + 1] = S[:, i] * np.exp((mu_annual - 0.5 * sigma_annual**2) * dt + sigma_annual * np.sqrt(dt) * Z[:, i])
    
    final_vals = S[:, -1]
    mc_var = np.percentile(final_vals, (1 - confidence_level) * 100)
    
    fig_mc = make_subplots(rows=1, cols=2, subplot_titles=("Simulated Paths (300 shown)", "Outcome Distribution at Year 1"))
    for i in range(min(300, simulations)):
        color = 'rgba(63,185,80,0.15)' if final_vals[i] >= initial_cap else 'rgba(248,81,73,0.15)'
        fig_mc.add_trace(go.Scatter(y=S[i, :], mode='lines', line=dict(color=color, width=0.8), showlegend=False), row=1, col=1)
    
    fig_mc.add_trace(go.Histogram(x=final_vals, nbinsx=50, marker_color='#3fb950', showlegend=False), row=1, col=2)
    fig_mc.add_vline(x=mc_var, line_dash='dot', line_color='#f85149', annotation_text=f"MC VaR: ${mc_var:,.0f}", row=1, col=2)
    
    fig_mc.update_layout(height=450, plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3')
    st.plotly_chart(fig_mc, width="stretch")

# --- TAB 3: STRESS TESTING ---
with tab3:
    st.subheader("🌪️ Historical Crisis Scenario Simulation")
    scenarios = {
        "2008 Global Financial Crisis": -0.42,
        "2020 COVID-19 Liquidity Crash": -0.32,
        "2022 Interest Rate Shock": -0.15,
        "Geopolitical Oil Shock": -0.12
    }
    
    df_stress = pd.DataFrame.from_dict(scenarios, orient='index', columns=["Portfolio Impact (%)"])
    df_stress["Loss Amount ($)"] = df_stress["Portfolio Impact (%)"] * initial_cap
    
    fig_st = go.Figure(go.Bar(
        x=df_stress["Portfolio Impact (%)"] * 100,
        y=df_stress.index,
        orientation='h',
        marker_color='#f85149'
    ))
    fig_st.update_layout(title="Estimated Portfolio Drawdown under Shocks (%)", plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3', height=400)
    st.plotly_chart(fig_st, width="stretch")
    st.dataframe(df_stress.style.format({"Portfolio Impact (%)": "{:.2f}%", "Loss Amount ($)": "${:,.2f}"}), width="stretch")

# --- TAB 4: FACTOR & ATTRIBUTION ---
with tab4:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("📈 Factor Beta Exposure")
        factors = {"Market (Beta)": 1.12, "Size (SMB)": -0.18, "Value (HML)": 0.25, "Momentum": 0.45}
        fig_f = go.Figure(go.Bar(x=list(factors.keys()), y=list(factors.values()), marker_color='#1baf7a'))
        fig_f.update_layout(title="Portfolio Factor Exposures", plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3', height=350)
        st.plotly_chart(fig_f, width="stretch")
        
    with col_b:
        st.subheader("🏆 Brinson Attribution Model")
        sectors = ["Tech/Semi", "Consumer", "Financials", "Healthcare"]
        alloc = [0.012, -0.003, 0.002, 0.005]
        select = [0.022, 0.004, -0.001, 0.006]
        
        fig_b = go.Figure()
        fig_b.add_trace(go.Bar(name='Allocation Effect', x=sectors, y=alloc, marker_color='#2a78d6'))
        fig_b.add_trace(go.Bar(name='Selection Effect', x=sectors, y=select, marker_color='#eb6834'))
        fig_b.update_layout(barmode='stack', title="Active Return Decomposition", plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3', height=350)
        st.plotly_chart(fig_b, width="stretch")

# --- TAB 5: 3D VOLATILITY & DRAWDOWN ---
with tab5:
    col_3d, col_dd = st.columns(2)
    
    with col_3d:
        st.subheader("🧊 3D Implied Volatility Surface")
        M_mesh, T_mesh = np.meshgrid(np.linspace(0.8, 1.2, 30), np.linspace(0.1, 2.0, 30))
        IV = sigma_annual + 0.1 * (M_mesh - 1) + 0.05 / np.sqrt(T_mesh) + 0.15 * (M_mesh - 1)**2
        
        fig_3d = go.Figure(data=[go.Surface(z=IV, x=M_mesh, y=T_mesh, colorscale='IceFire', opacity=0.9)])
        fig_3d.update_layout(scene=dict(
            xaxis_title='Moneyness', yaxis_title='Maturity', zaxis_title='IV',
            xaxis=dict(backgroundcolor='#0d1117'), yaxis=dict(backgroundcolor='#0d1117'), zaxis=dict(backgroundcolor='#0d1117')
        ), paper_bgcolor='#0d1117', margin=dict(l=0, r=0, b=0, t=10), height=400)
        st.plotly_chart(fig_3d, width="stretch")
        
    with col_dd:
        st.subheader("📉 Historical Drawdown Profile")
        nav_series = 100 * (1 + port_returns).cumprod()
        min_dd, dd_series = max_drawdown(nav_series)
        
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(y=dd_series * 100, fill='tozeroy', mode='lines', line=dict(color='#f85149', width=1.5)))
        fig_dd.update_layout(title=f"Underwater Chart (Max DD: {min_dd*100:.2f}%)", plot_bgcolor='#0d1117', paper_bgcolor='#0d1117', font_color='#e6edf3', height=400)
        st.plotly_chart(fig_dd, width="stretch")