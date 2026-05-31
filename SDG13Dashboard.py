import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# 1. PAGE CONFIG & THEME
#    Principles: Aesthetic-Usability, White Space, Consistency
# ==========================================================
st.set_page_config(
    page_title="SDG 13: Climate & Health",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Design tokens (one place to keep colors consistent) ----
NAVY_TEXT   = "#93C5FD"
SUBTLE_TEXT = "#94A3B8"
BORDER      = "#1E293B"
INSIGHT_BG  = "#0F2A43"
INSIGHT_BD  = "#3B82F6"
POLICY_BG   = "#0F2E1E"
POLICY_BD   = "#22C55E"
MATH_BG     = "#111827"
NEUTRAL_BAR = "#64748B"
HIGHLIGHT   = "#EF4444"

PLOTLY_TEMPLATE = "plotly_dark"
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
TRANSPARENT = "rgba(0,0,0,0)"

def style_fig(fig, height=380):
    """Consistent dark styling + responsive margins for every chart."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin={"r": 10, "t": 10, "l": 10, "b": 10},
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        font={"color": "#E2E8F0", "size": 13},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig

st.markdown(f"""
    <style>
    [data-testid="collapsedControl"] {{display: none;}}
    .title {{font-size: 2.4rem; font-weight: 800; color: {NAVY_TEXT};
             margin-bottom: 0; line-height: 1.1;}}
    .subtitle {{font-size: 1.1rem; color: {SUBTLE_TEXT}; margin-bottom: 1.2rem;}}
    .insight-box, .policy-box, .math-box {{
        color: #E5E7EB; padding: 1.1rem 1.25rem; border-radius: 8px;
        margin-bottom: 1rem; font-size: 0.97rem; line-height: 1.55;
    }}
    .insight-box {{background: {INSIGHT_BG}; border-left: 5px solid {INSIGHT_BD};}}
    .policy-box  {{background: {POLICY_BG};  border-left: 5px solid {POLICY_BD};}}
    .math-box    {{background: {MATH_BG};    border: 1px solid {BORDER};}}
    .insight-box strong, .policy-box strong {{color: #FFFFFF;}}
    .math-box code {{background:#1F2937; color:#FBBF24; padding:1px 5px; border-radius:4px;}}
    .takeaway {{color:{SUBTLE_TEXT}; font-size:0.85rem; font-style:italic;
                margin-top:-0.4rem; margin-bottom:0.8rem;}}
    button[data-baseweb="tab"] {{font-size: 1rem;}}
    div[data-testid="stHorizontalBlock"] {{flex-wrap: wrap;}}

    .footer {{
        color: #64748B;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid #1E293B;
        line-height: 1.6;
    }}
    .footer a {{
        color: #60A5FA;
        text-decoration: none;
        font-weight: 500;
    }}
    .footer a:hover {{
        text-decoration: underline;
    }}
    </style>
""", unsafe_allow_html=True)



# ==========================================================
# 2. DATA LOADING
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("final_sdg13_dashboard_data.csv")
    if "Log_GDP_per_Capita" in df.columns and "GDP_per_Capita" not in df.columns:
        df["GDP_per_Capita"] = np.expm1(df["Log_GDP_per_Capita"])
    return df

df = load_data()

ASEAN = {
    "BRN": "Brunei", "KHM": "Cambodia", "IDN": "Indonesia", "LAO": "Laos",
    "MYS": "Malaysia", "MMR": "Myanmar", "PHL": "Philippines",
    "SGP": "Singapore", "THA": "Thailand", "VNM": "Vietnam",
}

# Real model results pulled from the notebook (Huber RLM, n=4,959)
MODEL = {
    "n": 4959,
    "rmse": 0.0866, "mae": 0.0702, "r2": 0.758,
    "coefs": [
        ("Log_CO2_Emissions",        "Carbon emissions (log)",       -0.0584, 0.0016, "<0.001", -0.0616, -0.0553, "down"),
        ("Log_GDP_per_Capita",       "Wealth / GDP per person (log)",-0.0435, 0.0016, "<0.001", -0.0467, -0.0404, "down"),
        ("Temperature anomaly",      "Temperature anomaly (°C)",     +0.0097, 0.0026, "<0.001", +0.0046, +0.0148, "up"),
        ("GDPadjustedUrbanization",  "Urban density (wealth-adj.)",  -0.0005, 0.0001, "<0.001", -0.0006, -0.0003, "down"),
    ],
    "vif_before": [("Urbanization vs Wealth", "> 15")],
    "bp_p": "< 0.001",
}

# ==========================================================
# 3. HEADER & GLOBAL CONTROL
#    Principles: Visual Hierarchy, Hick's Law (one global input)
# ==========================================================
st.markdown('<div class="title">🌍 SDG 13: The Climate–Health Paradox</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">How wealth, carbon, and rising temperatures shape a country\'s health risk.</div>', unsafe_allow_html=True)

min_year, max_year = int(df["Year"].min()), int(df["Year"].max())
selected_year = st.slider("🗓️ Pick a year", min_value=min_year, max_value=max_year, value=max_year)
df_year = df[df["Year"] == selected_year].copy()

st.caption(f"Showing {len(df_year)} countries for {selected_year}.")

# ==========================================================
# 4. TABS the story in four steps (Miller's Law: small chunks)
# ==========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. What's happening?",
    "🔬 2. Why is it happening?",
    "🎯 3. What can we do?",
    "🧮 4. How we know (The method)",
])

# ----------------------------------------------------------
# TAB 1 WHAT'S HAPPENING  (Focus + Context)
# ----------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg. temperature anomaly", f"{df_year['Temperature anomaly'].mean():.2f} °C",
              help="How much warmer than the historical baseline, on average.")
    c2.metric("Avg. health risk", f"{df_year['Health_Vulnerability'].mean():.3f}",
              help="0 = low risk, 1 = high risk. Higher means more exposed to health harm.")
    c3.metric("Avg. GDP per person", f"${df_year['GDP_per_Capita'].mean():,.0f}")
    most_vuln = df_year.loc[df_year["Health_Vulnerability"].idxmax()]
    c4.metric(f"Most at-risk country ({most_vuln['Country Code']})",
              f"{most_vuln['Health_Vulnerability']:.3f}")

    st.write("")
    col_map, col_trend = st.columns([3, 2])

    with col_map:
        st.markdown(f"**Where health risk is highest ({selected_year})**")
        vmin = float(df["Health_Vulnerability"].min())
        vmax = float(df["Health_Vulnerability"].max())
        fig_map = px.choropleth(
            df_year, locations="Country Code", color="Health_Vulnerability",
            hover_name="Country Code",
            hover_data={"GDP_per_Capita": ":$,.0f", "Temperature anomaly": ":.2f"},
            color_continuous_scale="Reds", range_color=[vmin, vmax],
        )
        fig_map.update_geos(bgcolor=TRANSPARENT, lakecolor=TRANSPARENT,
                            landcolor="#1E293B", showframe=False)
        style_fig(fig_map, 380)
        st.plotly_chart(fig_map, use_container_width=True, config=PLOT_CONFIG)
        st.markdown('<div class="takeaway">Darker red = higher health risk. '
                    'Risk clusters in lower-income regions.</div>', unsafe_allow_html=True)

    with col_trend:
        st.markdown("**The global trend over time**")
        g = df.groupby("Year")["Health_Vulnerability"].mean().reset_index()
        fig_ts = px.line(g, x="Year", y="Health_Vulnerability", markers=True)
        fig_ts.update_traces(line_color="#60A5FA")
        fig_ts.add_vline(x=selected_year, line_width=2, line_dash="dash", line_color=HIGHLIGHT)
        # Data-driven tight bounds: pad the true min/max so small, real
        # variations are visible without flattening or exaggerating them.
        y_lo, y_hi = g["Health_Vulnerability"].min(), g["Health_Vulnerability"].max()
        pad = (y_hi - y_lo) * 0.18
        fig_ts.update_yaxes(range=[y_lo - pad, y_hi + pad], dtick=0.01, tickformat=".2f")
        style_fig(fig_ts, 380)
        st.plotly_chart(fig_ts, use_container_width=True, config=PLOT_CONFIG)
        st.markdown('<div class="takeaway">Note the narrow scale (about 0.46–0.50): the global '
                    'average rose until ~2006, fell for a decade, then ticked up recently. '
                    'The shifts are real but small.</div>', unsafe_allow_html=True)

# ----------------------------------------------------------
# TAB 2 WHY  (Progressive Disclosure for the math)
# ----------------------------------------------------------
with tab2:
    col_text, col_scatter = st.columns([1, 2])

    with col_text:
        st.markdown("""
        <div class="insight-box">
        <strong>📈 The surprising pattern</strong><br><br>
        Richer, higher-polluting countries tend to have <em>lower</em> health risk.
        They can afford strong hospitals, clean water, and cooling so their people
        are better protected.<br><br>
        But the temperature rising caused by those emissions raises health risk
        <em>everywhere</em> and it hits poorer countries hardest.
        That is the paradox: the biggest emitters are the best shielded.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 Show the quick version of the method"):
            st.markdown("""
            <div class="math-box">
            <strong>In plain terms:</strong><br>
            We checked which factors move health risk up or down, and made sure the
            result was trustworthy.<br><br>
            • Wealthier &amp; higher-emission → risk goes <strong>down</strong><br>
            • Hotter than normal → risk goes <strong>up</strong><br><br>
            Full statistical detail lives in <strong>Tab 4</strong>.
            </div>
            """, unsafe_allow_html=True)

    with col_scatter:
        st.markdown(f"**Wealth vs. health risk ({selected_year})**")
        fig_scatter = px.scatter(
            df_year, x="GDP_per_Capita", y="Health_Vulnerability",
            size="CO2_Emissions", color="Temperature anomaly",
            hover_name="Country Code", log_x=True, color_continuous_scale="RdBu_r",
            labels={"GDP_per_Capita": "GDP per person (log scale)",
                    "Health_Vulnerability": "Health risk",
                    "Temperature anomaly": "Temp. anomaly (°C)",
                    "CO2_Emissions": "CO₂"},
        )
        style_fig(fig_scatter, 420)
        st.plotly_chart(fig_scatter, use_container_width=True, config=PLOT_CONFIG)
        st.markdown('<div class="takeaway">Each dot is a country. As wealth rises '
                    '(right), health risk falls. Bubble size = CO₂ emissions; '
                    'color = temperature anomaly.</div>', unsafe_allow_html=True)

# ----------------------------------------------------------
# TAB 3 WHAT CAN WE DO  (Consistency: red = the Philippines)
# ----------------------------------------------------------
with tab3:
    st.markdown("""
    <div class="policy-box">
    <strong>💡 What this means for the Philippines</strong><br><br>
    Our model shows that planned, denser cities slightly lower health risk
    because it is easier to deliver hospitals, vaccines, cooling centers, and
    disaster relief to people who live closer together.<br><br>
    <strong>The takeaway:</strong> invest in well-planned, climate-ready urban areas
    so that health services can reach people quickly when extreme weather hits.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Health risk across ASEAN ({selected_year})**")

    present = [c for c in ASEAN if c in df_year["Country Code"].values]
    missing = [ASEAN[c] for c in ASEAN if c not in df_year["Country Code"].values]

    df_asean = (df_year[df_year["Country Code"].isin(present)]
                .sort_values("Health_Vulnerability", ascending=False))
    color_map = {c: (HIGHLIGHT if c == "PHL" else NEUTRAL_BAR) for c in df_asean["Country Code"]}

    fig_bar = px.bar(
        df_asean, x="Country Code", y="Health_Vulnerability",
        color="Country Code", color_discrete_map=color_map,
        hover_data={"GDP_per_Capita": ":$,.0f", "Urbanization_Rate": ":.1f"},
        labels={"Health_Vulnerability": "Health risk"},
    )
    fig_bar.update_layout(showlegend=False)
    style_fig(fig_bar, 400)
    st.plotly_chart(fig_bar, use_container_width=True, config=PLOT_CONFIG)
    st.markdown('<div class="takeaway">The Philippines is shown in red; '
                'its ASEAN neighbours are grey for comparison.</div>', unsafe_allow_html=True)

    if missing:
        st.caption(f"Note: {', '.join(missing)} not available in the dataset for {selected_year}, "
                   f"so {'it is' if len(missing)==1 else 'they are'} excluded from this ranking.")

# ----------------------------------------------------------
# TAB 4 HOW WE KNOW (the method)
# ----------------------------------------------------------
with tab4:
    st.markdown("""
    <div class="insight-box">
    <strong>🧮 How the analysis was built</strong><br><br>
    This tab shows the statistical work behind the story. We took messy country
    data, cleaned it, fixed two technical problems, and fit a model that is
    resistant to outliers so the conclusions are reliable, not lucky.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**How well the model fits**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Observations", f"{MODEL['n']:,}", help="Country-year rows used to train the model.")
    m2.metric("Fit (R²)", f"{MODEL['r2']:.2f}", help="About 76% of the variation in health risk is explained.")
    m3.metric("Typical error (RMSE)", f"{MODEL['rmse']:.3f}", help="Average size of the model's miss, in risk units.")
    m4.metric("Avg. error (MAE)", f"{MODEL['mae']:.3f}", help="Average absolute miss, in risk units.")
    st.caption("Health risk runs 0–1, so an average miss near 0.07 is small.")

    st.write("")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("**What drives health risk (and which way)**")
        names   = [c[1] for c in MODEL["coefs"]]
        coefs   = [c[2] for c in MODEL["coefs"]]
        lo      = [c[2] - c[5] for c in MODEL["coefs"]]
        hi      = [c[6] - c[2] for c in MODEL["coefs"]]
        bar_col = [HIGHLIGHT if c[2] > 0 else "#3B82F6" for c in MODEL["coefs"]]

        fig_co = go.Figure()
        fig_co.add_trace(go.Bar(
            y=names, x=coefs, orientation="h",
            marker_color=bar_col,
            error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo,
                         color="#94A3B8", thickness=1.5),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig_co.add_vline(x=0, line_color="#94A3B8", line_width=1)
        fig_co.update_layout(xaxis_title="Effect on health risk (← lowers · raises →)")
        style_fig(fig_co, 360)
        st.plotly_chart(fig_co, use_container_width=True, config=PLOT_CONFIG)
        st.markdown('<div class="takeaway">Blue bars push risk down; red pushes it up. '
                    'Lines show the confidence range. Only temperature raises risk.</div>',
                    unsafe_allow_html=True)

    with col_right:
        st.markdown("**How the factors relate**")
        cols = ["Health_Vulnerability", "Log_GDP_per_Capita", "Log_CO2_Emissions",
                "GDPadjustedUrbanization", "Temperature anomaly"]
        short = ["Health risk", "Wealth", "CO₂", "Urban (adj.)", "Temp."]
        corr = df[cols].corr()
        fig_hm = px.imshow(corr, x=short, y=short, color_continuous_scale="RdBu_r",
                           zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
        style_fig(fig_hm, 360)
        st.plotly_chart(fig_hm, use_container_width=True, config=PLOT_CONFIG)
        st.markdown('<div class="takeaway">Red = move together, blue = move oppositely. '
                    'Wealth and CO₂ both track with lower risk.</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("**The full model results**")
    table = pd.DataFrame([{
        "Factor": c[1],
        "Effect": f"{c[2]:+.4f}",
        "Direction": "Raises risk ▲" if c[7] == "up" else "Lowers risk ▼",
        "Std. error": f"{c[3]:.4f}",
        "p-value": c[4],
        "95% range": f"[{c[5]:+.4f}, {c[6]:+.4f}]",
    } for c in MODEL["coefs"]])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("Model: Robust Linear Regression (Huber M-estimator). "
               "Every factor is statistically significant (p < 0.001).")

    with st.expander("🔧 The two problems we had to fix first (and how)"):
        st.markdown(f"""
        <div class="math-box">
        <strong>1. Overlapping factors (multicollinearity).</strong>
        Wealth and urbanization were so closely linked that the model couldn't tell
        them apart (VIF {MODEL['vif_before'][0][1]}). We mathematically removed the
        wealth portion from urbanization, which created a clean, independent
        <code>GDPadjustedUrbanization</code> factor. Afterwards all factors were well-behaved.
        <br><br>
        <strong>2. Uneven error spread (heteroskedasticity).</strong>
        A Breusch–Pagan test showed the model's errors were far larger for some
        countries than others (p {MODEL['bp_p']}). That breaks ordinary regression.
        So we switched to <strong>robust regression</strong>, which downweights
        extreme cases and keeps the conclusions trustworthy.
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("**⚠️ What this analysis can't tell us**")
    st.markdown("""
    <div class="math-box">
    <strong>Correlation, not proof of cause.</strong> The model shows factors that
    <em>move together</em> with health risk. It does not prove one causes the other.
    Wealthier countries having lower risk does not mean emissions <em>make</em> people
    healthier wealth, better healthcare, and many other things travel together.<br><br>
    <strong>Other causes we didn't measure.</strong> Governance, conflict, disease
    history, and geography all affect health risk but aren't in this model. Some of
    what we credit to wealth or temperature may really belong to these.<br><br>
    <strong>A snapshot in time.</strong> The relationships reflect 1995–2023. They may
    not hold for the future or for any single country's unique path.<br><br>
    <strong>Data gaps.</strong> Some countries (e.g. Singapore) are missing, and minor
    gaps were filled by interpolation so results are estimates, not exact counts.
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# 5. DASHBOARD FOOTER
#    Principles: Negative Space, Consistency, Transparency
# ==========================================================
st.markdown("""
    <div class="footer">
        🎓 <strong>Project Purpose:</strong>Final Project for CIS 220.<br>
        👤 <strong>Designed & Built by:</strong> Erden Jhed Teope | 📚 <strong>Literature & Data Sources:</strong> 
        <a href="https://gain.nd.edu/our-work/country-index/methodology/" target="_blank">ND-GAIN Index</a> • 
        <a href="https://www.ipcc.ch/report/ar6/wg2/" target="_blank">IPCC Report (2022)</a> • 
        <a href="https://unhabitat.org/wcr/" target="_blank">UN-Habitat (2022)</a> • 
        <a href="https://www.who.int/news-room/fact-sheets/detail/climate-change-and-health" target="_blank">WHO Fact Sheet (2023)</a>
    </div>
""", unsafe_allow_html=True)
