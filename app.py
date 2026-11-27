import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.tools.tools import add_constant
import streamlit as st

# ============================================================
# 1. CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Observatorio TIC & Productividad · Pasto 2019–2024",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta "papel + tinta + tres acentos"
INK = "#14213D"
PAPER = "#F7F3EA"
AMBER = "#E8890C"
TEAL = "#0F9D8A"
CORAL = "#E4572E"
INDIGO = "#3A3F9E"
SLATE = "#64748B"
PALETA = [INDIGO, TEAL, AMBER, CORAL, SLATE, "#8B5CF6"]


def html(s: str) -> str:
    """Quita la indentación para que Markdown no trate el HTML como bloque de código."""
    return "\n".join(line.strip() for line in s.strip().splitlines())


def mostrar(s: str):
    st.markdown(html(s), unsafe_allow_html=True)


st.markdown(
    html(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,700;9..144,900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background: #F7F3EA; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3, h4 { font-family: 'Fraunces', serif !important; color: #14213D; letter-spacing: -0.01em; }
h3 { font-weight: 700 !important; }
code { font-family: 'JetBrains Mono', monospace !important; background: #EFE8D8 !important; color: #14213D !important; border-radius: 6px; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background: #14213D; }
section[data-testid="stSidebar"] * { color: #EDE7D6 !important; }
section[data-testid="stSidebar"] h3 { color: #F4C26B !important; font-size: 1.05rem; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
section[data-testid="stSidebar"] [data-baseweb="select"] > div { background: rgba(255,255,255,.07); border-color: rgba(255,255,255,.2); }
section[data-testid="stSidebar"] [data-baseweb="tag"] { background: #E8890C !important; }
section[data-testid="stSidebar"] [data-baseweb="tag"] * { color: #14213D !important; font-weight: 700; }

/* ---------- Hero ---------- */
.hero {
  position: relative; overflow: hidden; border-radius: 28px; padding: 3.2rem 3rem 2.8rem 3rem;
  background:
    radial-gradient(circle at 12% 20%, rgba(232,137,12,.55) 0%, transparent 38%),
    radial-gradient(circle at 88% 15%, rgba(15,157,138,.55) 0%, transparent 40%),
    radial-gradient(circle at 70% 95%, rgba(58,63,158,.8) 0%, transparent 45%),
    #14213D;
  color: #F7F3EA; box-shadow: 0 24px 60px rgba(20,33,61,.28); margin-bottom: 1.6rem;
}
.hero::after {
  content: ""; position: absolute; inset: 0; opacity: .12; pointer-events: none;
  background-image: linear-gradient(rgba(255,255,255,.6) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.6) 1px, transparent 1px);
  background-size: 44px 44px;
}
.hero > * { position: relative; z-index: 2; }
.hero-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: .78rem; letter-spacing: .22em; text-transform: uppercase; color: #F4C26B; margin-bottom: 1rem; }
.hero-title { font-family: 'Fraunces', serif; font-weight: 900; font-size: 3.5rem; line-height: 1.02; margin: 0 0 1.1rem 0; max-width: 880px; }
.hero-title em { font-style: italic; color: #F4C26B; font-weight: 700; }
.hero-sub { font-size: 1.1rem; line-height: 1.6; color: #D9D6CB; max-width: 780px; margin: 0 0 1.6rem 0; }
.chip { display: inline-block; padding: .38rem .9rem; margin: 0 .5rem .5rem 0; border: 1px solid rgba(247,243,234,.35); border-radius: 999px; font-size: .8rem; font-weight: 600; backdrop-filter: blur(4px); background: rgba(255,255,255,.07); }

/* ---------- Bienvenida ---------- */
.section-kicker { font-family: 'JetBrains Mono', monospace; font-size: .74rem; letter-spacing: .2em; text-transform: uppercase; color: #E8890C; font-weight: 600; margin-bottom: .2rem; }
.lead { font-size: 1.12rem; line-height: 1.75; color: #1E293B; text-align: justify; text-justify: inter-word; hyphens: auto; -webkit-hyphens: auto; }
.lead strong { color: #14213D; }
.dropcap::first-letter { font-family: 'Fraunces', serif; font-weight: 900; font-size: 4.2rem; float: left; line-height: .85; padding: .35rem .6rem 0 0; color: #E8890C; }

.stat-wall { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; }
.stat { background: #14213D; color: #F7F3EA; border-radius: 18px; padding: 1.1rem 1.2rem; }
.stat:nth-child(2) { background: #0F9D8A; }
.stat:nth-child(3) { background: #E8890C; color: #14213D; }
.stat:nth-child(4) { background: #3A3F9E; }
.stat .n { font-family: 'Fraunces', serif; font-weight: 900; font-size: 2.1rem; line-height: 1; }
.stat .l { font-size: .8rem; margin-top: .4rem; opacity: .9; font-weight: 600; }

.pipeline { display: flex; flex-wrap: wrap; gap: .5rem; align-items: stretch; margin: 1.2rem 0 .4rem 0; }
.step { flex: 1 1 150px; background: #FFFFFF; border: 1.5px solid #14213D; border-radius: 16px; padding: .9rem 1rem; position: relative; box-shadow: 4px 4px 0 #14213D; }
.step .k { font-family: 'JetBrains Mono', monospace; font-size: .7rem; color: #E8890C; font-weight: 600; }
.step .t { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.05rem; color: #14213D; margin: .15rem 0; }
.step .d { font-size: .8rem; color: #475569; line-height: 1.4; }

.qcard { background: #FFFFFF; border-radius: 20px; padding: 1.4rem 1.4rem 1.2rem; border: 1px solid #E7DFCB; height: 100%; position: relative; overflow: hidden; }
.qcard::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 6px; background: var(--c); }
.qcard .q { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.15rem; color: #14213D; margin-bottom: .4rem; }
.qcard .a { font-size: .9rem; line-height: 1.55; color: #334155; }

.nav-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin-top: .6rem; }
.nav-item { background: #EFE8D8; border-radius: 14px; padding: .9rem 1rem; }
.nav-item b { font-family: 'Fraunces', serif; color: #14213D; font-size: 1rem; }
.nav-item span { display: block; font-size: .82rem; color: #475569; margin-top: .2rem; line-height: 1.4; }

/* ---------- KPI ---------- */
div[data-testid="stMetric"] {
  background: #FFFFFF; border: 1px solid #E7DFCB; border-top: 5px solid #E8890C;
  padding: 1rem 1.2rem; border-radius: 16px; box-shadow: 0 8px 22px rgba(20,33,61,.06);
}
div[data-testid="stMetric"] label { color: #64748B !important; font-weight: 700 !important; font-size: .78rem !important; text-transform: uppercase; letter-spacing: .05em; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-family: 'Fraunces', serif !important; color: #14213D !important; font-weight: 900 !important; }

/* ---------- Cajas de lectura ---------- */
.insight-box {
  background: #FFFFFF; border: 1.5px solid #14213D; box-shadow: 5px 5px 0 #0F9D8A;
  padding: 1rem 1.3rem; border-radius: 14px; margin: .6rem 0 1.4rem 0; color: #1E293B; font-size: .94rem; line-height: 1.6;
}
.insight-box strong { color: #14213D; }

/* ---------- Pestañas ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: #14213D; padding: 7px; border-radius: 16px; }
.stTabs [data-baseweb="tab"] { border-radius: 11px; padding: 9px 18px; font-weight: 700; color: #D9D6CB; background: transparent; }
.stTabs [aria-selected="true"] { background: #E8890C !important; color: #14213D !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Gráficos y tablas ---------- */
div[data-testid="stPlotlyChart"] { background: #FFFFFF; border: 1px solid #E7DFCB; border-radius: 18px; padding: .4rem; box-shadow: 0 8px 22px rgba(20,33,61,.05); }
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid #E7DFCB; }
.stButton > button, .stDownloadButton > button { background: #14213D; color: #F7F3EA; border-radius: 12px; border: none; font-weight: 700; padding: .6rem 1.2rem; }
.stButton > button:hover, .stDownloadButton > button:hover { background: #E8890C; color: #14213D; }

.footer { text-align: center; color: #64748B; font-size: .8rem; padding: 2rem 0 0 0; border-top: 1px solid #E0D6BD; margin-top: 2.5rem; }
@media (max-width: 900px) { .hero-title { font-size: 2.2rem; } .nav-grid { grid-template-columns: 1fr; } .hero { padding: 2rem 1.4rem; } }
</style>
"""
    ),
    unsafe_allow_html=True,
)

# Plantilla Plotly coherente con el diseño
pio.templates["pasto"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color=INK, size=13),
        colorway=PALETA,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#EDE6D3", zerolinecolor="#D9CFB4", linecolor="#D9CFB4"),
        yaxis=dict(gridcolor="#EDE6D3", zerolinecolor="#D9CFB4", linecolor="#D9CFB4"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
)
pio.templates.default = "pasto"


def estilo(fig, h=420):
    fig.update_layout(height=h)
    return fig


# ============================================================
# 2. CARGA DE DATOS
# ============================================================
DATA_DIR = Path("data")


@st.cache_data
def cargar_datos():
    df = pd.read_parquet(DATA_DIR / "EMICRON_Pasto_Maestra_2019_2024.parquet")
    cargas = pd.read_csv(DATA_DIR / "cargas_acp_imtic.csv")
    with open(DATA_DIR / "resultados_modelos_pasto.json", "r", encoding="utf-8") as f:
        modelos = json.load(f)
    return df, cargas, modelos


df, cargas, modelos_json = cargar_datos()
df_res_oficial = pd.DataFrame(modelos_json["coeficientes"])

DESC_TIC = {
    "P2532": "Usa Internet en las operaciones del micronegocio",
    "P1559": "Tiene presencia en redes sociales o página web",
    "P2524": "Usa computador de escritorio, portátil o tableta",
    "P1006_1": "Uso TIC: Comprar insumos o materias primas",
    "P1006_2": "Uso TIC: Vender productos o servicios",
    "P1006_3": "Uso TIC: Realizar trámites gubernamentales",
    "P1006_4": "Uso TIC: Banca electrónica y servicios financieros",
    "P1006_5": "Uso TIC: Publicidad y mercadeo digital",
    "P1006_6": "Uso TIC: Atención y servicio al cliente",
    "P1006_7": "Uso TIC: Capacitación del personal o propietario",
    "P1006_8": "Uso TIC: Investigación de mercados y precios",
    "P1006_9": "Uso TIC: Gestión administrativa o contable",
    "P1006_10": "Uso TIC: Comunicación interna / coordinación",
    "P1006_11": "Uso TIC: Plataformas digitales de domicilio/entrega",
    "P1006_12": "Uso TIC: Medios de pago digitales (billeteras/QR)",
    "P1006_13": "Uso TIC: Otros usos operativos",
}
cargas["descripcion"] = cargas["variable"].map(DESC_TIC).fillna(cargas["variable"])

df["Genero"] = np.where(df["mujer"] == 1, "Mujer", "Hombre")
df["Tipo_Unidad"] = np.where(df["patron"] == 1, "Empleador (Patrón)", "Cuenta Propia")
df["Tiene_TIC"] = np.where(df["IMTIC"] > 0, "Con adopción TIC (IMTIC > 0)", "Sin adopción TIC (IMTIC = 0)")


# ============================================================
# 3. ESTIMADOR ECONOMÉTRICO (5 MODELOS DE LA TESIS)
# ============================================================
@st.cache_data
def estimar_modelos_tesis(df_sub, incluir_fe_sector=True, incluir_fe_anio=True):
    vars_req = ["ln_prod_laboral", "IMTIC", "edad", "mujer", "antiguedad", "patron", "empleo", "sector", "anio_estudio"]
    d = df_sub.dropna(subset=vars_req).copy()
    d = d[np.isfinite(d["ln_prod_laboral"])].copy()

    if len(d) < 60:
        return None, None, None, len(d)

    cols_cat = []
    if incluir_fe_sector and d["sector"].nunique() > 1:
        cols_cat.append("sector")
    if incluir_fe_anio and d["anio_estudio"].nunique() > 1:
        cols_cat.append("anio_estudio")

    cols_prev = set(d.columns)
    if cols_cat:
        d = pd.get_dummies(d, columns=cols_cat, drop_first=True, dtype=int)
    dummies = [c for c in d.columns if c not in cols_prev]

    controles = ["IMTIC", "edad", "mujer", "antiguedad", "patron", "empleo"]
    X_cols = controles + dummies
    y = d["ln_prod_laboral"].astype(float)
    X = add_constant(d[X_cols].astype(float))

    m_ols = sm.OLS(y, X).fit(cov_type="HC1")
    m_q10 = QuantReg(y, X).fit(q=0.10, max_iter=2000)
    m_q50 = QuantReg(y, X).fit(q=0.50, max_iter=2000)
    m_q90 = QuantReg(y, X).fit(q=0.90, max_iter=2000)
    m_huber = RLM(y, X, M=sm.robust.norms.HuberT()).fit()

    modelos_dict = {
        "OLS_HC1": m_ols,
        "Quantile_0.10": m_q10,
        "Quantile_0.50": m_q50,
        "Quantile_0.90": m_q90,
        "Huber_M": m_huber,
    }

    filas = []
    for nombre, mod in modelos_dict.items():
        ci = mod.conf_int()
        filas.append(
            pd.DataFrame(
                {
                    "variable": mod.params.index,
                    "coef": mod.params.values,
                    "std_err": mod.bse.values,
                    "pvalue": mod.pvalues.values,
                    "ci_low": ci[0].values,
                    "ci_high": ci[1].values,
                    "modelo": nombre,
                }
            )
        )

    metricas = {
        "r2": float(m_ols.rsquared),
        "r2_adj": float(m_ols.rsquared_adj),
        "pr2_q10": float(m_q10.prsquared),
        "pr2_q50": float(m_q50.prsquared),
        "pr2_q90": float(m_q90.prsquared),
    }

    diag_df = pd.DataFrame(
        {
            "fitted": m_ols.fittedvalues.values,
            "resid": m_ols.resid.values,
            "IMTIC_nivel": d["IMTIC_nivel"].values,
        }
    )
    return pd.concat(filas, ignore_index=True), metricas, diag_df, len(d)


# ============================================================
# 4. BARRA LATERAL
# ============================================================
with st.sidebar:
    st.markdown("### 🧭 Ficha metodológica")
    st.caption(
        "**Ámbito:** San Juan de Pasto (cabecera municipal, `COD_DEPTO=52`, `AREA=52`, `CLASE=1`)\n\n"
        "**Población:** micronegocios ($L \\le 9$)\n\n"
        "**Fuente:** microdatos EMICRON · DANE (2019–2024)"
    )
    st.divider()
    st.markdown("### 🎛️ Filtros de consulta")

    anios_disp = sorted(df["anio"].dropna().unique().astype(int))
    anios_sel = st.multiselect("Corte anual (2019–2024)", anios_disp, default=anios_disp)

    sectores_disp = sorted(df["sector_nombre"].dropna().unique())
    sectores_sel = st.multiselect("Gran sector (GRUPOS4)", sectores_disp, default=sectores_disp)

    tipos_disp = ["Cuenta Propia", "Empleador (Patrón)"]
    tipos_sel = st.multiselect("Categoría ocupacional", tipos_disp, default=tipos_disp)

    genero_disp = ["Hombre", "Mujer"]
    genero_sel = st.multiselect("Sexo del propietario(a)", genero_disp, default=genero_disp)

    solo_submuestra_ayudantes = st.checkbox("Restringir a unidades con ayudantes (submuestra M4)", value=False)
    st.divider()
    st.caption("Los filtros afectan los paneles 1, 2, 5 y el recálculo en vivo del panel 3.")

df_f = df[
    df["anio"].isin(anios_sel)
    & df["sector_nombre"].isin(sectores_sel)
    & df["Tipo_Unidad"].isin(tipos_sel)
    & df["Genero"].isin(genero_sel)
].copy()

if solo_submuestra_ayudantes and "m4_total_ayudantes" in df_f.columns:
    df_f = df_f[df_f["m4_total_ayudantes"] > 0].copy()

# ============================================================
# 5. HERO + BIENVENIDA Y CONTEXTUALIZACIÓN
# ============================================================
mostrar(
    """
<div class="hero">
  <div class="hero-eyebrow">Programa de Economía · Universidad de Nariño</div>
  <div class="hero-title">Observatorio <em>Micronegocios</em> San Juan de Pasto 2019-2024</div>
  <p class="hero-sub">
    Un observatorio interactivo que mide cómo la adopción de tecnologías de la información y las comunicaciones
    se relaciona con la productividad laboral de los micronegocios de San Juan de Pasto entre 2019 y 2024.
  </p>
  <span class="chip">📍 San Juan de Pasto</span>
  <span class="chip">🏪 Micronegocios (L ≤ 9)</span>
  <span class="chip">📅 2019–2024</span>
  <span class="chip">🧮 IMTIC · MCO · Cuantiles · Huber</span>
</div>
"""
)

# --- Bienvenida: narrativa + cifras del universo completo ---
n_total = len(df)
anio_min, anio_max = int(df["anio"].min()), int(df["anio"].max())
n_sect = df["sector_nombre"].nunique()
prod_med_total = df["prod_laboral"].median()
adop_total = (df["IMTIC"] > 0).mean() * 100

bienv_l, bienv_r = st.columns([1.35, 1], gap="large")
with bienv_l:
    mostrar('<div class="section-kicker">Bienvenida</div>')
    st.markdown("### Una puerta de entrada a los datos")
    mostrar(
        """
<p class="lead dropcap">
Bienvenido(a) al Observatorio. En Pasto, como en el resto del país, la economía cotidiana descansa sobre miles de
<strong>micronegocios</strong>: tiendas, talleres, puestos de comida, peluquerías y servicios que emplean a nueve
personas o menos. Son la principal fuente de ingreso de muchos hogares, pero también unidades con
<strong>productividad baja y muy desigual</strong>. Una pregunta recorre toda esta investigación:
<strong>¿la adopción de herramientas digitales marca una diferencia en lo que produce cada trabajador?</strong>
</p>
<p class="lead">
Para responderla se reconstruyó un panel de seis años de la <strong>Encuesta de Micronegocios (EMICRON)</strong> del DANE,
se sintetizaron <strong>16 indicadores de uso de TIC</strong> en un único índice (el <strong>IMTIC</strong>) y se estimaron
cinco modelos complementarios para observar el efecto promedio y también cómo cambia en los extremos de la distribución.
</p>
"""
    )
with bienv_r:
    mostrar(
        f"""
<div class="stat-wall">
  <div class="stat"><div class="n">{n_total:,}</div><div class="l">micronegocios observados en la base maestra</div></div>
  <div class="stat"><div class="n">{anio_min}–{anio_max}</div><div class="l">ventana temporal del estudio</div></div>
  <div class="stat"><div class="n">${prod_med_total:,.0f}</div><div class="l">productividad laboral mediana por mes</div></div>
  <div class="stat"><div class="n">{adop_total:.0f}%</div><div class="l">de unidades con algún uso de TIC (IMTIC &gt; 0)</div></div>
</div>
"""
    )

# --- Ruta metodológica ---
mostrar('<div class="section-kicker" style="margin-top:1.6rem">Ruta de la investigación</div>')
mostrar(
    """
<div class="pipeline">
  <div class="step"><div class="k">01</div><div class="t">EMICRON</div><div class="d">Microdatos DANE de la cabecera de Pasto, 2019–2024.</div></div>
  <div class="step"><div class="k">02</div><div class="t">ACP</div><div class="d">16 variables binarias de TIC se resumen en el primer componente principal.</div></div>
  <div class="step"><div class="k">03</div><div class="t">IMTIC</div><div class="d">Índice normalizado de 0 a 1 de adopción tecnológica.</div></div>
  <div class="step"><div class="k">04</div><div class="t">Modelos</div><div class="d">MCO con errores HC1, regresión cuantílica (10, 50, 90) y Huber.</div></div>
  <div class="step"><div class="k">05</div><div class="t">Simulador</div><div class="d">Escenarios de cierre de brecha con los coeficientes oficiales.</div></div>
</div>
"""
)

# --- Preguntas guía ---
st.markdown("")
q1, q2, q3 = st.columns(3, gap="medium")
with q1:
    mostrar(
        f"""
<div class="qcard" style="--c:{AMBER}">
  <div class="q">¿Qué es la productividad laboral aquí?</div>
  <div class="a">Es el valor agregado homologado mensual dividido entre el personal ocupado. Se analiza en logaritmos
  para que los coeficientes se lean como variaciones porcentuales aproximadas.</div>
</div>
"""
    )
with q2:
    mostrar(
        f"""
<div class="qcard" style="--c:{TEAL}">
  <div class="q">¿Qué es el IMTIC?</div>
  <div class="a">Un índice de 0 a 1 que combina acceso (internet, computador, redes) y usos operativos (ventas, pagos,
  banca, mercadeo, trámites). Cero significa que no hay uso de TIC; uno, adopción plena.</div>
</div>
"""
    )
with q3:
    mostrar(
        f"""
<div class="qcard" style="--c:{INDIGO}">
  <div class="q">¿Por qué cinco modelos?</div>
  <div class="a">La productividad es asimétrica y heterocedástica. MCO con HC1 da el efecto promedio; los cuantiles
  muestran si el efecto difiere entre negocios de subsistencia y consolidados; Huber resiste valores extremos.</div>
</div>
"""
    )

with st.expander("🗺️ Cómo recorrer este observatorio", expanded=False):
    mostrar(
        """
<div class="nav-grid">
  <div class="nav-item"><b>1 · Panorama</b><span>Estructura productiva, sectores, género y escala (cuenta propia vs. empleadores).</span></div>
  <div class="nav-item"><b>2 · Índice IMTIC</b><span>Cargas del ACP, evolución de la brecha digital y adopción por sector.</span></div>
  <div class="nav-item"><b>3 · Modelos</b><span>Tabla 4.5 oficial o recálculo en vivo con tus filtros, con el gradiente cuantílico.</span></div>
  <div class="nav-item"><b>4 · Diagnóstico</b><span>Residuales, heterocedasticidad y auditoría de cobertura muestral.</span></div>
  <div class="nav-item"><b>5 · Laboratorio</b><span>Construye tus propios gráficos con las variables de la base maestra.</span></div>
  <div class="nav-item"><b>6 · Simulador</b><span>Proyecta la productividad de un perfil de negocio según su nivel de IMTIC.</span></div>
</div>
"""
    )

with st.expander("📖 Glosario rápido", expanded=False):
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            "- **Micronegocio:** unidad económica con hasta 9 personas ocupadas.\n"
            "- **EMICRON:** Encuesta de Micronegocios del DANE.\n"
            "- **ACP:** Análisis de Componentes Principales, técnica que resume varias variables en pocas dimensiones.\n"
            "- **IMTIC:** Índice Multidimensional de adopción de TIC, entre 0 y 1."
        )
    with g2:
        st.markdown(
            "- **HC1:** errores estándar robustos a heterocedasticidad (White).\n"
            "- **Regresión cuantílica:** estima el efecto en distintos puntos de la distribución (τ = 0.10, 0.50, 0.90).\n"
            "- **Huber:** M-estimador que reduce el peso de observaciones atípicas.\n"
            "- **Cuenta propia / Patrón:** trabaja solo vs. contrata personal."
        )

st.markdown("---")

# ============================================================
# 6. KPI Y ESTADO DE FILTROS
# ============================================================
if df_f.empty:
    st.warning("⚠️ No hay observaciones con la combinación de filtros actual. Ajusta la selección en el panel izquierdo.")
    st.stop()

df_f_val = df_f.dropna(subset=["ln_prod_laboral", "IMTIC"]).copy()

mostrar('<div class="section-kicker">Tablero de control</div>')
st.markdown("### Indicadores de la selección actual")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Micronegocios", f"{len(df_f):,}")
k2.metric("Muestra econométrica", f"{len(df_f_val):,}")
k3.metric("Productividad mediana", f"${df_f['prod_laboral'].median():,.0f}/mes")
k4.metric("IMTIC promedio", f"{df_f['IMTIC'].mean():.3f}")
k5.metric("Adopción TIC (> 0)", f"{(df_f['IMTIC'] > 0).mean() * 100:.1f}%")
st.markdown("")

# ============================================================
# 7. PESTAÑAS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "① Panorama",
        "② Índice IMTIC",
        "③ Modelos",
        "④ Diagnóstico",
        "⑤ Laboratorio",
        "⑥ Simulador",
    ]
)

# ------------------------------------------------------------
# TAB 1
# ------------------------------------------------------------
with tab1:
    st.markdown("### Estructura productiva y evolución de la productividad (2019–2024)")
    mostrar(
        """
<div class="insight-box"><strong>Lectura:</strong> la productividad laboral (valor agregado homologado por trabajador ocupado)
presenta una marcada asimetría positiva y brechas persistentes según el nivel de adopción tecnológica, el sector,
el sitio de operación y el sexo del propietario.</div>
"""
    )

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader("ln(Productividad) por año y nivel de IMTIC")
        fig_box = px.box(
            df_f_val, x="anio", y="ln_prod_laboral", color="IMTIC_nivel",
            color_discrete_sequence=["#94A3B8", INDIGO, TEAL],
            labels={"anio": "Año de estudio", "ln_prod_laboral": "ln(Valor agregado / personal ocupado)", "IMTIC_nivel": "Nivel TIC"},
        )
        fig_box.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(estilo(fig_box, 440), use_container_width=True)

    with c2:
        st.subheader("Mapa de la estructura: sector y categoría")
        fig_tm = px.treemap(
            df_f.assign(n=1), path=[px.Constant("Pasto"), "sector_nombre", "Tipo_Unidad"], values="n",
            color_discrete_sequence=PALETA,
        )
        fig_tm.update_traces(root_color="#F7F3EA", textinfo="label+percent parent")
        st.plotly_chart(estilo(fig_tm, 440), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Productividad mediana por sitio de funcionamiento y sexo")
        ubic_df = (
            df_f.dropna(subset=["prod_laboral"]).groupby(["ubicacion_nombre", "Genero"], as_index=False)["prod_laboral"].median()
        )
        fig_ubic = px.bar(
            ubic_df, x="ubicacion_nombre", y="prod_laboral", color="Genero", barmode="group",
            color_discrete_map={"Hombre": INDIGO, "Mujer": AMBER},
            labels={"ubicacion_nombre": "Sitio de operación", "prod_laboral": "Productividad mediana (COP/mes)"},
        )
        fig_ubic.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(estilo(fig_ubic, 410), use_container_width=True)

    with c4:
        st.subheader("Cuenta propia vs. empleadores: trayectoria")
        tipo_df = df_f.dropna(subset=["prod_laboral"]).groupby(["anio", "Tipo_Unidad"], as_index=False)["prod_laboral"].median()
        fig_tipo = px.line(
            tipo_df, x="anio", y="prod_laboral", color="Tipo_Unidad", markers=True,
            color_discrete_map={"Cuenta Propia": SLATE, "Empleador (Patrón)": TEAL},
            labels={"anio": "Año", "prod_laboral": "Productividad mediana (COP/mes)", "Tipo_Unidad": "Categoría"},
        )
        fig_tipo.update_traces(line=dict(width=4), marker=dict(size=10))
        st.plotly_chart(estilo(fig_tipo, 410), use_container_width=True)

    st.subheader("Productividad mediana según nivel de adopción TIC")
    brecha = df_f.dropna(subset=["prod_laboral"]).groupby("IMTIC_nivel", as_index=False)["prod_laboral"].agg(["median", "count"]).reset_index()
    fig_br = px.bar(
        brecha, x="IMTIC_nivel", y="median", text=brecha["median"].round(0).map("${:,.0f}".format),
        color="IMTIC_nivel", color_discrete_sequence=["#94A3B8", INDIGO, TEAL],
        labels={"IMTIC_nivel": "Nivel de adopción TIC", "median": "Productividad mediana (COP/mes)"},
    )
    fig_br.update_traces(textposition="outside")
    fig_br.update_layout(showlegend=False)
    st.plotly_chart(estilo(fig_br, 360), use_container_width=True)

# ------------------------------------------------------------
# TAB 2
# ------------------------------------------------------------
with tab2:
    st.markdown("### Construcción del Índice Multidimensional de Adopción de TIC (IMTIC)")
    mostrar(
        """
<div class="insight-box"><strong>Metodología ACP:</strong> el IMTIC sintetiza, mediante el primer componente principal (PC1),
las 16 variables binarias del módulo de TIC de la EMICRON, normalizadas al intervalo [0, 1]. Esto evita la colinealidad entre
herramientas digitales y captura tanto el acceso básico como los usos operativos.</div>
"""
    )

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.subheader("Cargas factoriales del PC1")
        cs = cargas.sort_values("carga_pc1", ascending=True)
        fig_cargas = px.bar(
            cs, x="carga_pc1", y="descripcion", orientation="h", text=cs["carga_pc1"].round(3),
            color="carga_pc1", color_continuous_scale=[[0, "#F4D9A8"], [0.5, AMBER], [1, CORAL]],
            labels={"carga_pc1": "Carga factorial (PC1)", "descripcion": "Indicador TIC (EMICRON)"},
        )
        fig_cargas.update_traces(textposition="outside")
        fig_cargas.update_layout(coloraxis_showscale=False, xaxis_range=[0, cargas["carga_pc1"].max() * 1.18])
        st.plotly_chart(estilo(fig_cargas, 600), use_container_width=True)

    with c2:
        st.subheader("Evolución de la brecha digital")
        niv_anio = pd.crosstab(df_f["anio"], df_f["IMTIC_nivel"], normalize="index") * 100
        niv_anio = niv_anio.reset_index().melt(id_vars="anio", var_name="Nivel", value_name="Porcentaje")
        fig_niv = px.bar(
            niv_anio, x="anio", y="Porcentaje", color="Nivel", barmode="stack",
            color_discrete_map={
                "Bajo (Sin uso TIC)": "#D5CDB8",
                "Medio (Adopción básica)": INDIGO,
                "Alto (Adopción avanzada)": TEAL,
            },
            labels={"anio": "Año", "Porcentaje": "Participación (%)"},
        )
        fig_niv.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(estilo(fig_niv, 340), use_container_width=True)

        st.subheader("IMTIC promedio por sector")
        imtic_sec = df_f.groupby("sector_nombre", as_index=False)["IMTIC"].mean().sort_values("IMTIC", ascending=True)
        fig_isec = px.bar(
            imtic_sec, x="IMTIC", y="sector_nombre", orientation="h", text=imtic_sec["IMTIC"].round(3),
            color_discrete_sequence=[INDIGO], labels={"IMTIC": "IMTIC promedio (0–1)", "sector_nombre": "Sector"},
        )
        fig_isec.update_traces(textposition="outside")
        st.plotly_chart(estilo(fig_isec, 230), use_container_width=True)

    st.subheader("Mapa de calor: IMTIC promedio por sector y año")
    heat = df_f.pivot_table(index="sector_nombre", columns="anio", values="IMTIC", aggfunc="mean")
    fig_heat = px.imshow(
        heat, text_auto=".2f", aspect="auto",
        color_continuous_scale=[[0, "#F7F3EA"], [0.5, "#F4C26B"], [1, CORAL]],
        labels=dict(x="Año", y="Sector", color="IMTIC"),
    )
    st.plotly_chart(estilo(fig_heat, 340), use_container_width=True)

# ------------------------------------------------------------
# TAB 3
# ------------------------------------------------------------
with tab3:
    st.markdown("### Estimación econométrica: MCO (HC1), regresión cuantílica y M-estimador de Huber")

    op1, op2, op3 = st.columns([1.2, 1, 1])
    with op1:
        modo_estimacion = st.radio(
            "Origen de estimación:",
            ["Resultados oficiales de la tesis (N = 14.217)", "Recalcular en vivo sobre filtros actuales"],
            horizontal=False,
        )
    es_oficial = modo_estimacion.startswith("Resultados")
    with op2:
        fe_sec = st.checkbox("Efectos fijos sectoriales (GRUPOS4)", value=True, disabled=es_oficial)
    with op3:
        fe_anio = st.checkbox("Efectos fijos temporales (2019–2024)", value=True, disabled=es_oficial)

    if es_oficial:
        res_act = df_res_oficial.copy()
        n_act = modelos_json["n_obs"]
        met_act = {
            "r2": modelos_json["r2"]["OLS_HC1"],
            "r2_adj": modelos_json["r2"]["OLS_HC1_adj"],
            "pr2_q10": modelos_json["r2"]["Quantile_0.10"],
            "pr2_q50": modelos_json["r2"]["Quantile_0.50"],
            "pr2_q90": modelos_json["r2"]["Quantile_0.90"],
        }
        _, _, diag_act, _ = estimar_modelos_tesis(df, True, True)
    else:
        res_act, met_act, diag_act, n_act = estimar_modelos_tesis(df_f, fe_sec, fe_anio)

    if res_act is None:
        st.error("La selección de filtros tiene menos de 60 observaciones válidas. Amplía los filtros para estimar.")
    else:
        mostrar(
            f"""
<div class="insight-box"><strong>Especificación estimada (N = {n_act:,}):</strong>
<code>ln(Prod_Laboral) = β₀ + β₁·IMTIC + β₂·edad + β₃·mujer + β₄·antiguedad + β₅·patron + β₆·empleo + δ_sector + γ_año + ε</code><br>
<strong>R² MCO (HC1):</strong> <code>{met_act["r2"]:.4f}</code> &nbsp;|&nbsp;
<strong>R² ajustado:</strong> <code>{met_act["r2_adj"]:.4f}</code> &nbsp;|&nbsp;
<strong>Pseudo-R² (Q0.10 / Q0.50 / Q0.90):</strong>
<code>{met_act["pr2_q10"]:.4f} / {met_act["pr2_q50"]:.4f} / {met_act["pr2_q90"]:.4f}</code></div>
"""
        )

        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.subheader("Efecto marginal del IMTIC por método (IC 95%)")
            sub_imtic = res_act[res_act["variable"] == "IMTIC"].copy()
            nombres_map = {
                "OLS_HC1": "MCO (errores robustos HC1)",
                "Quantile_0.10": "Cuantílica (τ = 0.10)",
                "Quantile_0.50": "Cuantílica (τ = 0.50)",
                "Quantile_0.90": "Cuantílica (τ = 0.90)",
                "Huber_M": "M-estimador de Huber",
            }
            sub_imtic["Modelo"] = sub_imtic["modelo"].map(nombres_map)

            fig_coef = go.Figure()
            fig_coef.add_trace(
                go.Scatter(
                    x=sub_imtic["coef"], y=sub_imtic["Modelo"], mode="markers+text",
                    text=[f"{v:.4f}" for v in sub_imtic["coef"]], textposition="top center",
                    marker=dict(size=14, color=[INK, SLATE, INDIGO, TEAL, AMBER][: len(sub_imtic)], line=dict(width=2, color="white")),
                    error_x=dict(
                        type="data", symmetric=False,
                        array=sub_imtic["ci_high"] - sub_imtic["coef"],
                        arrayminus=sub_imtic["coef"] - sub_imtic["ci_low"],
                        thickness=2.5, width=7, color="#334155",
                    ),
                )
            )
            fig_coef.add_vline(x=0, line_dash="dash", line_color=CORAL)
            fig_coef.update_layout(xaxis_title="Coeficiente β_IMTIC sobre ln(Productividad)", yaxis=dict(autorange="reversed"))
            st.plotly_chart(estilo(fig_coef, 430), use_container_width=True)

        with c2:
            st.subheader("Dispersión empírica: IMTIC vs. ln(Productividad)")
            df_scatter = df_f_val.sample(min(len(df_f_val), 3500), random_state=42)
            fig_reg = px.scatter(
                df_scatter, x="IMTIC", y="ln_prod_laboral", opacity=0.25, trendline="ols",
                trendline_color_override=CORAL, color_discrete_sequence=[INDIGO],
                labels={"IMTIC": "Índice IMTIC (0–1)", "ln_prod_laboral": "ln(Productividad)"},
            )
            st.plotly_chart(estilo(fig_reg, 430), use_container_width=True)

        # Gradiente cuantílico
        sub_q = sub_imtic[sub_imtic["modelo"].str.startswith("Quantile")].copy()
        if not sub_q.empty:
            st.subheader("Gradiente cuantílico: ¿el efecto del IMTIC cambia a lo largo de la distribución?")
            sub_q["tau"] = sub_q["modelo"].str.replace("Quantile_", "", regex=False).astype(float)
            sub_q = sub_q.sort_values("tau")
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(
                x=list(sub_q["tau"]) + list(sub_q["tau"])[::-1],
                y=list(sub_q["ci_high"]) + list(sub_q["ci_low"])[::-1],
                fill="toself", fillcolor="rgba(15,157,138,.18)", line=dict(width=0), hoverinfo="skip", name="IC 95%",
            ))
            fig_g.add_trace(go.Scatter(
                x=sub_q["tau"], y=sub_q["coef"], mode="lines+markers", name="β_IMTIC",
                line=dict(color=TEAL, width=4), marker=dict(size=12),
            ))
            ols_b = sub_imtic[sub_imtic["modelo"] == "OLS_HC1"]["coef"]
            if not ols_b.empty:
                fig_g.add_hline(y=float(ols_b.values[0]), line_dash="dot", line_color=AMBER,
                                annotation_text="Efecto promedio MCO", annotation_position="top left")
            fig_g.update_layout(xaxis_title="Cuantil τ de la productividad", yaxis_title="β_IMTIC", xaxis=dict(tickvals=[0.1, 0.5, 0.9]))
            st.plotly_chart(estilo(fig_g, 360), use_container_width=True)

        st.subheader("Tabla 4.5 · Resumen comparativo de los modelos estimados")

        def estrellas(p):
            if pd.isna(p):
                return ""
            if p < 0.01:
                return "***"
            if p < 0.05:
                return "**"
            if p < 0.10:
                return "*"
            return ""

        orden_mod = ["OLS_HC1", "Quantile_0.10", "Quantile_0.50", "Quantile_0.90", "Huber_M"]
        etq_mod = {"OLS_HC1": "MCO (HC1)", "Quantile_0.10": "Q 0.10", "Quantile_0.50": "Q 0.50", "Quantile_0.90": "Q 0.90", "Huber_M": "Huber"}
        vars_tab = [("const", "Intercept"), ("IMTIC", "IMTIC"), ("edad", "edad"), ("mujer", "mujer"),
                    ("antiguedad", "antiguedad"), ("patron", "patron"), ("empleo", "empleo")]

        filas_t = []
        for v_cod, v_nom in vars_tab:
            f_c = {"Variable": v_nom}
            f_s = {"Variable": ""}
            for m in orden_mod:
                sub = res_act[(res_act["modelo"] == m) & (res_act["variable"] == v_cod)]
                if not sub.empty:
                    f_c[etq_mod[m]] = f"{sub['coef'].values[0]:.4f}{estrellas(sub['pvalue'].values[0])}"
                    f_s[etq_mod[m]] = f"({sub['std_err'].values[0]:.4f})"
                else:
                    f_c[etq_mod[m]] = ""
                    f_s[etq_mod[m]] = ""
            filas_t.extend([f_c, f_s])

        st.dataframe(pd.DataFrame(filas_t), use_container_width=True, hide_index=True)
        st.caption("Significancia: *** p < 0.01, ** p < 0.05, * p < 0.10. Errores estándar entre paréntesis. Incluye efectos fijos sectoriales y temporales.")

        st.download_button(
            "⬇ Descargar Tabla 4.5 completa (.csv)",
            data=res_act.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"Tabla_4_5_Pasto_N{n_act}.csv",
            mime="text/csv",
        )

# ------------------------------------------------------------
# TAB 4
# ------------------------------------------------------------
with tab4:
    st.markdown("### Diagnóstico de supuestos clásicos (justificación de HC1, cuantiles y Huber)")
    mostrar(
        """
<div class="insight-box"><strong>Sustento econométrico (Sección 4.4 de la tesis):</strong> el gráfico de residuales frente a
valores ajustados evidencia heterocedasticidad (dispersión no constante), mientras que las desviaciones en las colas justifican
los errores estándar robustos de White (HC1), la regresión cuantílica y el M-estimador de Huber.</div>
"""
    )

    if diag_act is not None:
        diag_sample = diag_act.sample(min(len(diag_act), 4000), random_state=42)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Residuales vs. valores ajustados (MCO)")
            fig_res = px.scatter(
                diag_sample, x="fitted", y="resid", opacity=0.28, color_discrete_sequence=[INDIGO],
                labels={"fitted": "Valores ajustados ln(Productividad)", "resid": "Residuales MCO"},
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color=CORAL)
            st.plotly_chart(estilo(fig_res, 410), use_container_width=True)

        with c2:
            st.subheader("Distribución de residuales (asimetría y colas)")
            fig_hist_r = px.histogram(
                diag_act, x="resid", nbins=60, marginal="box", color_discrete_sequence=[TEAL],
                labels={"resid": "Residuales MCO"},
            )
            st.plotly_chart(estilo(fig_hist_r, 410), use_container_width=True)

    st.divider()
    st.subheader("Auditoría de cobertura muestral (unipersonales + empleadores, 2019–2024)")
    aud_anio = df.groupby("anio", as_index=False).agg(
        Total_Pasto_Cabecera=("anio", "count"),
        Unipersonales_L1=("m4_total_ayudantes", lambda s: (s == 0).sum()),
        Con_Ayudantes_M4=("m4_total_ayudantes", lambda s: (s > 0).sum()),
        Muestra_Econometrica_Valida=("ln_prod_laboral", lambda s: np.isfinite(s).sum()),
    )
    st.dataframe(aud_anio, use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# TAB 5
# ------------------------------------------------------------
with tab5:
    st.markdown("### Laboratorio de gráficos de la base maestra")
    st.write("Elige variables de la investigación, genera gráficos a tu medida y expórtalos en PNG desde el ícono de cámara.")

    VARS_NUM = {
        "ln_prod_laboral": "ln(Productividad Laboral)",
        "prod_laboral": "Productividad Laboral (COP/mes)",
        "IMTIC": "Índice IMTIC (0–1)",
        "VALOR_AGREGADO_HOMOLOGADO": "Valor Agregado Mensual Homologado (COP)",
        "VENTAS_MES_ANTERIOR": "Ventas Mes Anterior (COP)",
        "COSTOS_MES_ANTERIOR": "Costos Mes Anterior (COP)",
        "GASTOS_MES": "Gastos Operativos Mes (COP)",
        "empleo": "Personal Ocupado Total (L)",
        "edad": "Edad del Propietario (años)",
        "antiguedad": "Antigüedad del Negocio (años)",
    }
    VARS_CAT = {
        "anio": "Año de Encuesta",
        "sector_nombre": "Gran Sector (GRUPOS4)",
        "IMTIC_nivel": "Nivel de Adopción TIC",
        "ubicacion_nombre": "Sitio de Funcionamiento",
        "Genero": "Sexo del Propietario",
        "Tipo_Unidad": "Categoría (Cuenta Propia vs. Patrón)",
    }

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        tipo_g = st.selectbox(
            "Tipo de visualización",
            ["Diagrama de Cajas (Boxplot)", "Dispersión (Scatter + Tendencia OLS)", "Barras Agrupadas (Mediana)",
             "Diagrama de Violín", "Histograma de Distribución"],
        )
    with g2:
        todas_v = {**VARS_CAT, **VARS_NUM}
        x_v = st.selectbox("Eje X", list(todas_v.keys()), index=1, format_func=lambda k: todas_v[k])
    with g3:
        y_v = st.selectbox("Eje Y (numérica)", list(VARS_NUM.keys()), index=0, format_func=lambda k: VARS_NUM[k])
    with g4:
        c_v = st.selectbox(
            "Agrupar por color", ["Ninguno"] + list(VARS_CAT.keys()), index=3,
            format_func=lambda k: "Sin agrupación" if k == "Ninguno" else VARS_CAT[k],
        )

    recortar = st.checkbox("Recortar percentiles extremos (1% – 99%) en el eje Y", value=False)
    df_p = df_f.copy()
    if recortar:
        q01, q99 = df_p[y_v].quantile([0.01, 0.99])
        df_p = df_p[df_p[y_v].between(q01, q99)]

    color_param = None if c_v == "Ninguno" else c_v

    if tipo_g == "Diagrama de Cajas (Boxplot)":
        fig_m = px.box(df_p.dropna(subset=[x_v, y_v]), x=x_v, y=y_v, color=color_param, labels=todas_v, color_discrete_sequence=PALETA)
    elif tipo_g == "Dispersión (Scatter + Tendencia OLS)":
        df_p_s = df_p.dropna(subset=[x_v, y_v])
        df_p_s = df_p_s.sample(min(len(df_p_s), 3500), random_state=42)
        fig_m = px.scatter(df_p_s, x=x_v, y=y_v, color=color_param, trendline="ols", opacity=0.35, labels=todas_v, color_discrete_sequence=PALETA)
    elif tipo_g == "Barras Agrupadas (Mediana)":
        grp_cols = [x_v] + ([color_param] if color_param and color_param != x_v else [])
        df_b = df_p.groupby(grp_cols, as_index=False)[y_v].median()
        fig_m = px.bar(df_b, x=x_v, y=y_v, color=color_param, barmode="group", labels=todas_v, color_discrete_sequence=PALETA)
    elif tipo_g == "Diagrama de Violín":
        fig_m = px.violin(df_p.dropna(subset=[x_v, y_v]), x=x_v, y=y_v, color=color_param, box=True, labels=todas_v, color_discrete_sequence=PALETA)
    else:
        fig_m = px.histogram(df_p.dropna(subset=[y_v]), x=y_v, color=color_param, marginal="box", barmode="overlay", opacity=0.65, labels=todas_v, color_discrete_sequence=PALETA)

    fig_m.update_layout(title=f"{tipo_g}: {VARS_NUM[y_v]} según {todas_v[x_v]}")
    st.plotly_chart(estilo(fig_m, 540), use_container_width=True)
    st.caption("📷 Usa el ícono de cámara en la esquina superior derecha del gráfico para exportarlo en alta resolución.")

# ------------------------------------------------------------
# TAB 6
# ------------------------------------------------------------
with tab6:
    st.markdown("### Simulador prospectivo: adopción TIC y cierre de brechas")
    st.write("Evalúa el cambio esperado en la productividad mensual de un micronegocio con los parámetros oficiales de la Tabla 4.5.")

    s1, s2, s3 = st.columns(3)
    with s1:
        sim_imtic = st.slider("Nivel IMTIC objetivo (0 = sin TIC, 1 = adopción plena)", 0.0, 1.0, 0.40, 0.05)
        sim_mod = st.selectbox(
            "Estimador de referencia (Tabla 4.5)",
            ["OLS_HC1", "Quantile_0.10", "Quantile_0.50", "Quantile_0.90", "Huber_M"],
            format_func=lambda m: {
                "OLS_HC1": "MCO HC1 (efecto promedio)",
                "Quantile_0.10": "Cuantil τ = 0.10 (subsistencia)",
                "Quantile_0.50": "Cuantil τ = 0.50 (mediana)",
                "Quantile_0.90": "Cuantil τ = 0.90 (consolidados)",
                "Huber_M": "M-estimador robusto de Huber",
            }[m],
        )
    with s2:
        sim_edad = st.number_input("Edad del propietario (años)", 18, 85, 42)
        sim_antig = st.number_input("Antigüedad del micronegocio (años)", 0.0, 50.0, 5.0, 0.5)
    with s3:
        sim_mujer = st.radio("Sexo del propietario(a)", [0, 1], format_func=lambda x: "Mujer" if x == 1 else "Hombre", horizontal=True)
        sim_patron = st.radio("Categoría", [0, 1], format_func=lambda x: "Empleador (Patrón)" if x == 1 else "Cuenta Propia", horizontal=True)
        sim_empleo = st.slider("Personal ocupado total (L)", 1, 9, 2 if sim_patron == 1 else 1)


    def est_ln_modelo(modelo, val_tic):
        c = df_res_oficial[df_res_oficial["modelo"] == modelo].set_index("variable")["coef"]
        return (
            c.get("const", 0)
            + c.get("IMTIC", 0) * val_tic
            + c.get("edad", 0) * sim_edad
            + c.get("mujer", 0) * sim_mujer
            + c.get("antiguedad", 0) * sim_antig
            + c.get("patron", 0) * sim_patron
            + c.get("empleo", 0) * sim_empleo
        )


    y_sin_tic = np.exp(est_ln_modelo(sim_mod, 0.0))
    y_con_tic = np.exp(est_ln_modelo(sim_mod, sim_imtic))
    var_pct = ((y_con_tic / y_sin_tic) - 1) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Productividad base (IMTIC = 0.00)", f"${y_sin_tic:,.0f} / mes")
    m2.metric(f"Proyectada (IMTIC = {sim_imtic:.2f})", f"${y_con_tic:,.0f} / mes", f"{var_pct:+.1f}%")
    m3.metric("Diferencial mensual por ocupado", f"{(y_con_tic - y_sin_tic):+,.0f} COP")

    grid_tic = np.linspace(0, 1, 50)
    curva_df = pd.DataFrame({"IMTIC": grid_tic, "Productividad_Estimada": [np.exp(est_ln_modelo(sim_mod, t)) for t in grid_tic]})
    fig_sim = px.area(
        curva_df, x="IMTIC", y="Productividad_Estimada", color_discrete_sequence=[TEAL],
        labels={"IMTIC": "Índice IMTIC (0–1)", "Productividad_Estimada": "Productividad mensual estimada (COP)"},
    )
    fig_sim.add_vline(x=sim_imtic, line_dash="dash", line_color=AMBER, annotation_text=f"IMTIC = {sim_imtic:.2f}")
    fig_sim.update_layout(title="Curva de respuesta de la productividad ante incrementos del IMTIC")
    st.plotly_chart(estilo(fig_sim, 360), use_container_width=True)

    st.subheader("El mismo perfil según cada estimador")
    comp = []
    for m, etq in [("OLS_HC1", "MCO HC1"), ("Quantile_0.10", "Q 0.10"), ("Quantile_0.50", "Q 0.50"),
                   ("Quantile_0.90", "Q 0.90"), ("Huber_M", "Huber")]:
        b = np.exp(est_ln_modelo(m, 0.0))
        p = np.exp(est_ln_modelo(m, sim_imtic))
        comp.append({"Estimador": etq, "Incremento (%)": (p / b - 1) * 100})
    comp_df = pd.DataFrame(comp)
    fig_comp = px.bar(
        comp_df, x="Estimador", y="Incremento (%)", text=comp_df["Incremento (%)"].map("{:+.1f}%".format),
        color="Estimador", color_discrete_sequence=[INK, SLATE, INDIGO, TEAL, AMBER],
    )
    fig_comp.update_traces(textposition="outside")
    fig_comp.update_layout(showlegend=False)
    st.plotly_chart(estilo(fig_comp, 340), use_container_width=True)
    st.caption("Los resultados son proyecciones condicionales del modelo para el perfil elegido (sin efectos fijos), no una predicción individual.")

# ============================================================
# 8. PIE
# ============================================================
mostrar(
    """
<div class="footer">Observatorio TIC &amp; Productividad · San Juan de Pasto 2019–2024 ·
Programa de Economía, Universidad de Nariño · Fuente: microdatos EMICRON, DANE</div>
"""
)
