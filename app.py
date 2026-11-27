import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.tools.tools import add_constant
import streamlit as st

# ------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO VISUAL (CSS)
# ------------------------------------------------------------
st.set_page_config(
    page_title="Observatorio TIC & Productividad · San Juan de Pasto (2019–2024)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Fondo y tipografía general */
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
    }
    /* Banner institucional superior */
    .hero-banner {
        background: linear-gradient(135deg, #0F2C59 0%, #1E4E8C 60%, #2B6CB0 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        color: #FFFFFF;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(15, 44, 89, 0.16);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.16);
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #E2E8F0;
        margin: 0;
    }
    /* Tarjetas KPI */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #1E4E8C;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetric"] label {
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #0F2C59 !important;
        font-weight: 800 !important;
    }
    /* Cajas de interpretación económica */
    .insight-box {
        background-color: #F8FAFC;
        border-left: 4px solid #0D9488;
        padding: 1rem 1.25rem;
        border-radius: 0 10px 10px 0;
        margin: 0.8rem 0 1.2rem 0;
        color: #1E293B;
        font-size: 0.94rem;
    }
    /* Estilo de pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F1F5F9;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        color: #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0F2C59 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 2. CARGA DE DATOS PROCESADOS (DATA/)
# ------------------------------------------------------------
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

# Diccionario oficial de las 16 variables TIC de la investigación
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
    "P1006_13": "Uso TIC: Otros usos operativos"
}
cargas["descripcion"] = cargas["variable"].map(DESC_TIC).fillna(cargas["variable"])

# Variables derivadas estrictamente de los controles de la tesis
df["Genero"] = np.where(df["mujer"] == 1, "Mujer", "Hombre")
df["Tipo_Unidad"] = np.where(df["patron"] == 1, "Empleador (Patrón)", "Cuenta Propia")
df["Tiene_TIC"] = np.where(df["IMTIC"] > 0, "Con adopción TIC (IMTIC > 0)", "Sin adopción TIC (IMTIC = 0)")

PALETA_AZUL = ["#0F2C59", "#1E4E8C", "#2B6CB0", "#0D9488", "#D97706"]

# ------------------------------------------------------------
# 3. ESTIMADOR ECONOMÉTRICO EN VIVO (SOLO LOS 5 MODELOS DE LA TESIS)
# ------------------------------------------------------------
@st.cache_data
def estimar_modelos_tesis(df_sub, incluir_fe_sector=True, incluir_fe_anio=True):
    vars_req = [
        "ln_prod_laboral", "IMTIC", "edad", "mujer",
        "antiguedad", "patron", "empleo", "sector", "anio_estudio"
    ]
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

    # Exactamente los 5 modelos de la Tabla 4.5 de la tesis
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
        "Huber_M": m_huber
    }

    filas = []
    for nombre, mod in modelos_dict.items():
        ci = mod.conf_int()
        filas.append(pd.DataFrame({
            "variable": mod.params.index,
            "coef": mod.params.values,
            "std_err": mod.bse.values,
            "pvalue": mod.pvalues.values,
            "ci_low": ci[0].values,
            "ci_high": ci[1].values,
            "modelo": nombre
        }))

    metricas = {
        "r2": float(m_ols.rsquared),
        "r2_adj": float(m_ols.rsquared_adj),
        "pr2_q10": float(m_q10.prsquared),
        "pr2_q50": float(m_q50.prsquared),
        "pr2_q90": float(m_q90.prsquared)
    }

    diag_df = pd.DataFrame({
        "fitted": m_ols.fittedvalues.values,
        "resid": m_ols.resid.values,
        "IMTIC_nivel": d["IMTIC_nivel"].values
    })

    return pd.concat(filas, ignore_index=True), metricas, diag_df, len(d)

# ------------------------------------------------------------
# 4. BARRA LATERAL · FILTROS DE LA INVESTIGACIÓN
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏛️ Ficha Metodológica")
    st.caption(
        "**Ámbito:** San Juan de Pasto (Cabecera Municipal, `COD_DEPTO=52`, `AREA=52`, `CLASE=1`)\n\n"
        "**Población:** Micronegocios ($L \\le 9$)\n\n"
        "**Fuente:** Microdatos EMICRON · DANE (2019–2024)"
    )
    st.divider()
    st.markdown("### 🎛️ Filtros de Consulta")

    anios_disp = sorted(df["anio"].dropna().unique().astype(int))
    anios_sel = st.multiselect("Corte anual (2019–2024)", anios_disp, default=anios_disp)

    sectores_disp = sorted(df["sector_nombre"].dropna().unique())
    sectores_sel = st.multiselect("Gran sector (GRUPOS4)", sectores_disp, default=sectores_disp)

    tipos_disp = ["Cuenta Propia", "Empleador (Patrón)"]
    tipos_sel = st.multiselect("Categoría ocupacional", tipos_disp, default=tipos_disp)

    genero_disp = ["Hombre", "Mujer"]
    genero_sel = st.multiselect("Sexo del propietario(a)", genero_disp, default=genero_disp)

    solo_submuestra_ayudantes = st.checkbox(
        "Restringir a unidades con ayudantes (trazabilidad submuestra M4)",
        value=False
    )

df_f = df[
    df["anio"].isin(anios_sel) &
    df["sector_nombre"].isin(sectores_sel) &
    df["Tipo_Unidad"].isin(tipos_sel) &
    df["Genero"].isin(genero_sel)
].copy()

if solo_submuestra_ayudantes and "m4_total_ayudantes" in df_f.columns:
    df_f = df_f[df_f["m4_total_ayudantes"] > 0].copy()

# ------------------------------------------------------------
# 5. ENCABEZADO HERO Y TARJETAS KPI
# ------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">Programa de Economía · Universidad de Nariño</div>
    <div class="hero-title">Observatorio de Adopción de TIC y Productividad Laboral</div>
    <p class="hero-subtitle">
        Análisis microeconométrico de los micronegocios (L ≤ 9) en San Juan de Pasto, 2019–2024 ·
        Índice Multidimensional IMTIC (ACP), MCO (HC1), Regresión Cuantílica y Estimador Robusto de Huber
    </p>
</div>
""", unsafe_allow_html=True)

if df_f.empty:
    st.warning("⚠️ No hay observaciones con la combinación de filtros actual. Ajusta la selección en el panel izquierdo.")
    st.stop()

df_f_val = df_f.dropna(subset=["ln_prod_laboral", "IMTIC"]).copy()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Micronegocios en muestra", f"{len(df_f):,}")
k2.metric("Muestra econométrica válida", f"{len(df_f_val):,}")
k3.metric("Productividad Mediana", f"${df_f['prod_laboral'].median():,.0f}/mes")
k4.metric("IMTIC Promedio (0–1)", f"{df_f['IMTIC'].mean():.3f}")
k5.metric("Adopción TIC (> 0)", f"{(df_f['IMTIC'] > 0).mean() * 100:.1f}%")

st.markdown("")

# ------------------------------------------------------------
# 6. PESTAÑAS PRINCIPALES DEL OBSERVATORIO
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Panorama Territorial",
    "2. Índice IMTIC (ACP 16 Variables)",
    "3. Modelos Econométricos (Tabla 4.5)",
    "4. Diagnóstico y Validación",
    "5. Constructor Manual de Gráficos",
    "6. Simulador de Brechas"
])

# ============================================================
# PESTAÑA 1: PANORAMA TERRITORIAL Y ESTRUCTURA PRODUCTIVA
# ============================================================
with tab1:
    st.markdown("### Estructura Productiva y Evolución de la Productividad Laboral (2019–2024)")
    st.markdown(
        '<div class="insight-box">'
        '<strong>Hallazgo estructural:</strong> La productividad laboral (Valor Agregado Homologado por trabajador ocupado) '
        'presenta una marcada asimetría positiva y brechas persistentes según el nivel de adopción tecnológica (IMTIC), '
        'el sector económico, el sitio de operación y el sexo del propietario.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader("Trayectoria de ln(Productividad Laboral) por Año y Nivel IMTIC")
        fig_box = px.box(
            df_f_val,
            x="anio", y="ln_prod_laboral", color="IMTIC_nivel",
            color_discrete_sequence=["#94A3B8", "#2B6CB0", "#0D9488"],
            labels={
                "anio": "Año de estudio",
                "ln_prod_laboral": "ln(Valor Agregado / Personal Ocupado)",
                "IMTIC_nivel": "Nivel de Adopción TIC"
            }
        )
        fig_box.update_layout(height=430, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig_box, use_container_width=True)

    with c2:
        st.subheader("Composición por Gran Sector (GRUPOS4)")
        sec_df = (
            df_f.groupby("sector_nombre", as_index=False)
            .agg(Unidades=("anio", "count"), Prod_Mediana=("prod_laboral", "median"), IMTIC_Prom=("IMTIC", "mean"))
        )
        fig_sec = px.pie(
            sec_df, names="sector_nombre", values="Unidades", hole=0.48,
            color_discrete_sequence=PALETA_AZUL
        )
        fig_sec.update_traces(textposition="inside", textinfo="percent+label")
        fig_sec.update_layout(height=430, showlegend=False)
        st.plotly_chart(fig_sec, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Productividad Mediana según Sitio de Funcionamiento y Género")
        ubic_df = (
            df_f.dropna(subset=["prod_laboral"])
            .groupby(["ubicacion_nombre", "Genero"], as_index=False)["prod_laboral"]
            .median()
        )
        fig_ubic = px.bar(
            ubic_df, x="ubicacion_nombre", y="prod_laboral", color="Genero", barmode="group",
            color_discrete_map={"Hombre": "#1E4E8C", "Mujer": "#D97706"},
            labels={"ubicacion_nombre": "Sitio de operación", "prod_laboral": "Productividad mediana (COP/mes)"}
        )
        fig_ubic.update_layout(height=400, xaxis_tickangle=-25)
        st.plotly_chart(fig_ubic, use_container_width=True)

    with c4:
        st.subheader("Cuenta Propia vs. Empleadores (Patrón): Escala y Productividad")
        tipo_df = (
            df_f.dropna(subset=["prod_laboral"])
            .groupby(["anio", "Tipo_Unidad"], as_index=False)["prod_laboral"]
            .median()
        )
        fig_tipo = px.line(
            tipo_df, x="anio", y="prod_laboral", color="Tipo_Unidad", markers=True,
            color_discrete_map={"Cuenta Propia": "#64748B", "Empleador (Patrón)": "#0D9488"},
            labels={"anio": "Año", "prod_laboral": "Productividad mediana (COP/mes)", "Tipo_Unidad": "Categoría"}
        )
        fig_tipo.update_layout(height=400)
        st.plotly_chart(fig_tipo, use_container_width=True)

# ============================================================
# PESTAÑA 2: ÍNDICE MULTIDIMENSIONAL IMTIC (ACP)
# ============================================================
with tab2:
    st.markdown("### Construcción del Índice Multidimensional de Adopción de TIC (IMTIC)")
    st.markdown(
        '<div class="insight-box">'
        '<strong>Metodología ACP:</strong> El IMTIC sintetiza mediante Análisis de Componentes Principales (PC1) '
        'las 16 variables binarias del Módulo de TIC de la EMICRON, normalizadas en el intervalo [0, 1]. '
        'Esto evita la colinealidad entre herramientas digitales y captura tanto el acceso básico como los usos operativos.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.subheader("Cargas Factoriales del Primer Componente Principal (PC1)")
        fig_cargas = px.bar(
            cargas.sort_values("carga_pc1", ascending=True),
            x="carga_pc1", y="descripcion", orientation="h",
            text=cargas.sort_values("carga_pc1", ascending=True)["carga_pc1"].round(3),
            color="carga_pc1", color_continuous_scale="Blues",
            labels={"carga_pc1": "Carga factorial (PC1)", "descripcion": "Indicador TIC (EMICRON)"}
        )
        fig_cargas.update_traces(textposition="outside")
        fig_cargas.update_layout(showlegend=False, height=560, xaxis_range=[0, cargas["carga_pc1"].max() * 1.15])
        st.plotly_chart(fig_cargas, use_container_width=True)

    with c2:
        st.subheader("Evolución de la Brecha Digital (2019–2024)")
        niv_anio = pd.crosstab(df_f["anio"], df_f["IMTIC_nivel"], normalize="index") * 100
        niv_anio = niv_anio.reset_index().melt(id_vars="anio", var_name="Nivel", value_name="Porcentaje")
        fig_niv = px.bar(
            niv_anio, x="anio", y="Porcentaje", color="Nivel", barmode="stack",
            color_discrete_map={
                "Bajo (Sin uso TIC)": "#CBD5E1",
                "Medio (Adopción básica)": "#2B6CB0",
                "Alto (Adopción avanzada)": "#0D9488"
            },
            labels={"anio": "Año", "Porcentaje": "Participación (%)"}
        )
        fig_niv.update_layout(height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_niv, use_container_width=True)

        st.subheader("IMTIC Promedio por Sector Económico")
        imtic_sec = df_f.groupby("sector_nombre", as_index=False)["IMTIC"].mean().sort_values("IMTIC", ascending=True)
        fig_isec = px.bar(
            imtic_sec, x="IMTIC", y="sector_nombre", orientation="h",
            text=imtic_sec["IMTIC"].round(3), color_discrete_sequence=["#1E4E8C"],
            labels={"IMTIC": "IMTIC promedio (0–1)", "sector_nombre": "Sector"}
        )
        fig_isec.update_traces(textposition="outside")
        fig_isec.update_layout(height=220)
        st.plotly_chart(fig_isec, use_container_width=True)

# ============================================================
# PESTAÑA 3: MODELOS ECONOMÉTRICOS (TABLA 4.5 DE LA TESIS)
# ============================================================
with tab3:
    st.markdown("### Estimación Econométrica: MCO (HC1), Regresión Cuantílica y M-Estimador de Huber")

    op1, op2, op3 = st.columns([1.2, 1, 1])
    with op1:
        modo_estimacion = st.radio(
            "Origen de estimación:",
            ["Resultados oficiales de la tesis (N = 14.217)", "Recalcular en vivo sobre filtros actuales"],
            horizontal=False
        )
    with op2:
        fe_sec = st.checkbox("Efectos fijos sectoriales (GRUPOS4)", value=True, disabled=(modo_estimacion.startswith("Resultados")))
    with op3:
        fe_anio = st.checkbox("Efectos fijos temporales (2019–2024)", value=True, disabled=(modo_estimacion.startswith("Resultados")))

    if modo_estimacion.startswith("Resultados"):
        res_act = df_res_oficial.copy()
        n_act = modelos_json["n_obs"]
        met_act = {
            "r2": modelos_json["r2"]["OLS_HC1"],
            "r2_adj": modelos_json["r2"]["OLS_HC1_adj"],
            "pr2_q10": modelos_json["r2"]["Quantile_0.10"],
            "pr2_q50": modelos_json["r2"]["Quantile_0.50"],
            "pr2_q90": modelos_json["r2"]["Quantile_0.90"]
        }
        _, _, diag_act, _ = estimar_modelos_tesis(df, True, True)
    else:
        res_act, met_act, diag_act, n_act = estimar_modelos_tesis(df_f, fe_sec, fe_anio)

    if res_act is None:
        st.error("La selección de filtros tiene menos de 60 observaciones válidas. Amplía los filtros para estimar.")
    else:
        st.markdown(
            f'<div class="insight-box">'
            f'<strong>Especificación estimada (N = {n_act:,}):</strong> '
            f'<code>ln(Prod_Laboral) = β₀ + β₁·IMTIC + β₂·edad + β₃·mujer + β₄·antiguedad + β₅·patron + β₆·empleo + δ_sector + γ_año + ε</code><br>'
            f'<strong>R² MCO (HC1):</strong> <code>{met_act["r2"]:.4f}</code> &nbsp;|&nbsp; '
            f'<strong>R² Ajustado:</strong> <code>{met_act["r2_adj"]:.4f}</code> &nbsp;|&nbsp; '
            f'<strong>Pseudo-R² (Q0.10 / Q0.50 / Q0.90):</strong> '
            f'<code>{met_act["pr2_q10"]:.4f} / {met_act["pr2_q50"]:.4f} / {met_act["pr2_q90"]:.4f}</code>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Gráfico 1: Gradiente cuantílico y robustez del IMTIC (Fig. 2 de la tesis)
        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.subheader("Efecto Marginal del IMTIC por Método y Cuantil (IC 95%)")
            sub_imtic = res_act[res_act["variable"] == "IMTIC"].copy()
            nombres_map = {
                "OLS_HC1": "MCO (Errores Robustos HC1)",
                "Quantile_0.10": "Regresión Cuantílica (τ = 0.10)",
                "Quantile_0.50": "Regresión Cuantílica (τ = 0.50)",
                "Quantile_0.90": "Regresión Cuantílica (τ = 0.90)",
                "Huber_M": "M-Estimador Robusto de Huber"
            }
            sub_imtic["Modelo"] = sub_imtic["modelo"].map(nombres_map)

            fig_coef = go.Figure()
            fig_coef.add_trace(go.Scatter(
                x=sub_imtic["coef"], y=sub_imtic["Modelo"],
                mode="markers+text",
                text=[f"{v:.4f}" for v in sub_imtic["coef"]],
                textposition="top center",
                marker=dict(size=13, color=["#0F2C59", "#64748B", "#2B6CB0", "#0D9488", "#D97706"]),
                error_x=dict(
                    type="data", symmetric=False,
                    array=sub_imtic["ci_high"] - sub_imtic["coef"],
                    arrayminus=sub_imtic["coef"] - sub_imtic["ci_low"],
                    thickness=2.5, width=7, color="#334155"
                )
            ))
            fig_coef.add_vline(x=0, line_dash="dash", line_color="crimson")
            fig_coef.update_layout(
                xaxis_title="Coeficiente β_IMTIC sobre ln(Productividad Laboral)",
                yaxis=dict(autorange="reversed"),
                height=420
            )
            st.plotly_chart(fig_coef, use_container_width=True)

        with c2:
            st.subheader("Dispersión Empírica: IMTIC vs. ln(Productividad)")
            df_scatter = df_f_val.sample(min(len(df_f_val), 3500), random_state=42)
            fig_reg = px.scatter(
                df_scatter, x="IMTIC", y="ln_prod_laboral",
                opacity=0.22, trendline="ols",
                trendline_color_override="#DC2626",
                color_discrete_sequence=["#1E4E8C"],
                labels={"IMTIC": "Índice IMTIC (0–1)", "ln_prod_laboral": "ln(Productividad Laboral)"}
            )
            fig_reg.update_layout(height=420)
            st.plotly_chart(fig_reg, use_container_width=True)

        # Tabla 4.5 completa
        st.subheader("Tabla 4.5 · Resumen Comparativo de los Modelos Estimados")
        def estrellas(p):
            if pd.isna(p): return ""
            if p < 0.01: return "***"
            if p < 0.05: return "**"
            if p < 0.10: return "*"
            return ""

        orden_mod = ["OLS_HC1", "Quantile_0.10", "Quantile_0.50", "Quantile_0.90", "Huber_M"]
        etq_mod = {
            "OLS_HC1": "MCO (HC1)",
            "Quantile_0.10": "Q 0.10",
            "Quantile_0.50": "Q 0.50",
            "Quantile_0.90": "Q 0.90",
            "Huber_M": "Huber"
        }
        vars_tab = [
            ("const", "Intercept"), ("IMTIC", "IMTIC"), ("edad", "edad"),
            ("mujer", "mujer"), ("antiguedad", "antiguedad"),
            ("patron", "patron"), ("empleo", "empleo")
        ]

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

        tabla_visual = pd.DataFrame(filas_t)
        st.dataframe(tabla_visual, use_container_width=True, hide_index=True)
        st.caption("Nota de significancia: *** p < 0.01, ** p < 0.05, * p < 0.10. Errores estándar entre paréntesis. Incluye efectos fijos sectoriales y temporales.")

        st.download_button(
            "⬇ Descargar Tabla 4.5 completa (.csv)",
            data=res_act.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"Tabla_4_5_Pasto_N{n_act}.csv",
            mime="text/csv"
        )

# ============================================================
# PESTAÑA 4: DIAGNÓSTICO DE RESIDUALES Y VALIDACIÓN MUESTRAL
# ============================================================
with tab4:
    st.markdown("### Diagnóstico de Supuestos Clásicos (Justificación de HC1, Cuantiles y Huber)")
    st.markdown(
        '<div class="insight-box">'
        '<strong>Sustento econométrico (Sección 4.4 de la tesis):</strong> El gráfico de residuales frente a valores ajustados '
        'evidencia heterocedasticidad (dispersión no constante), mientras que las desviaciones en las colas justifican '
        'el uso de errores estándar robustos de White (HC1), la regresión cuantílica y el M-estimador de Huber.'
        '</div>',
        unsafe_allow_html=True
    )

    if diag_act is not None:
        diag_sample = diag_act.sample(min(len(diag_act), 4000), random_state=42)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Residuales vs. Valores Ajustados (MCO)")
            fig_res = px.scatter(
                diag_sample, x="fitted", y="resid", opacity=0.25,
                color_discrete_sequence=["#1E4E8C"],
                labels={"fitted": "Valores ajustados ln(Productividad)", "resid": "Residuales MCO"}
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="#DC2626")
            fig_res.update_layout(height=400)
            st.plotly_chart(fig_res, use_container_width=True)

        with c2:
            st.subheader("Distribución de Residuales MCO (Asimetría y Colas)")
            fig_hist_r = px.histogram(
                diag_act, x="resid", nbins=60, marginal="box",
                color_discrete_sequence=["#0D9488"],
                labels={"resid": "Residuales MCO"}
            )
            fig_hist_r.update_layout(height=400)
            st.plotly_chart(fig_hist_r, use_container_width=True)

    st.divider()
    st.subheader("Auditoría de Cobertura Muestral (Unipersonales + Empleadores, 2019–2024)")
    aud_anio = (
        df.groupby("anio", as_index=False)
        .agg(
            Total_Pasto_Cabecera=("anio", "count"),
            Unipersonales_L1=("m4_total_ayudantes", lambda s: (s == 0).sum()),
            Con_Ayudantes_M4=("m4_total_ayudantes", lambda s: (s > 0).sum()),
            Muestra_Econometrica_Valida=("ln_prod_laboral", lambda s: np.isfinite(s).sum())
        )
    )
    st.dataframe(aud_anio, use_container_width=True, hide_index=True)

# ============================================================
# PESTAÑA 5: CONSTRUCTOR MANUAL DE GRÁFICOS
# ============================================================
with tab5:
    st.markdown("### Constructor Manual de Gráficos de la Base Maestra")
    st.write("Selecciona libremente variables de la investigación para generar gráficos personalizados y descargarlos en PNG:")

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
        "antiguedad": "Antigüedad del Negocio (años)"
    }
    VARS_CAT = {
        "anio": "Año de Encuesta",
        "sector_nombre": "Gran Sector (GRUPOS4)",
        "IMTIC_nivel": "Nivel de Adopción TIC",
        "ubicacion_nombre": "Sitio de Funcionamiento",
        "Genero": "Sexo del Propietario",
        "Tipo_Unidad": "Categoría (Cuenta Propia vs. Patrón)"
    }

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        tipo_g = st.selectbox(
            "Tipo de visualización",
            ["Diagrama de Cajas (Boxplot)", "Dispersión (Scatter + Tendencia OLS)",
             "Barras Agrupadas (Mediana)", "Diagrama de Violín", "Histograma de Distribución"]
        )
    with g2:
        todas_v = {**VARS_CAT, **VARS_NUM}
        x_v = st.selectbox("Eje X", list(todas_v.keys()), index=1, format_func=lambda k: todas_v[k])
    with g3:
        y_v = st.selectbox("Eje Y (numérica)", list(VARS_NUM.keys()), index=0, format_func=lambda k: VARS_NUM[k])
    with g4:
        c_v = st.selectbox(
            "Agrupar por color",
            ["Ninguno"] + list(VARS_CAT.keys()),
            index=3,
            format_func=lambda k: "Sin agrupación" if k == "Ninguno" else VARS_CAT[k]
        )

    recortar = st.checkbox("Recortar percentiles extremos (1% - 99%) en Eje Y para visualizar mejor valores monetarios", value=False)
    df_p = df_f.copy()
    if recortar:
        q01, q99 = df_p[y_v].quantile([0.01, 0.99])
        df_p = df_p[df_p[y_v].between(q01, q99)]

    color_param = None if c_v == "Ninguno" else c_v

    if tipo_g == "Diagrama de Cajas (Boxplot)":
        fig_m = px.box(df_p.dropna(subset=[x_v, y_v]), x=x_v, y=y_v, color=color_param, labels=todas_v, color_discrete_sequence=PALETA_AZUL)
    elif tipo_g == "Dispersión (Scatter + Tendencia OLS)":
        df_p_s = df_p.dropna(subset=[x_v, y_v]).sample(min(len(df_p), 3500), random_state=42)
        fig_m = px.scatter(df_p_s, x=x_v, y=y_v, color=color_param, trendline="ols", opacity=0.3, labels=todas_v, color_discrete_sequence=PALETA_AZUL)
    elif tipo_g == "Barras Agrupadas (Mediana)":
        grp_cols = [x_v] + ([color_param] if color_param and color_param != x_v else [])
        df_b = df_p.groupby(grp_cols, as_index=False)[y_v].median()
        fig_m = px.bar(df_b, x=x_v, y=y_v, color=color_param, barmode="group", labels=todas_v, color_discrete_sequence=PALETA_AZUL)
    elif tipo_g == "Diagrama de Violín":
        fig_m = px.violin(df_p.dropna(subset=[x_v, y_v]), x=x_v, y=y_v, color=color_param, box=True, labels=todas_v, color_discrete_sequence=PALETA_AZUL)
    else:
        fig_m = px.histogram(df_p.dropna(subset=[y_v]), x=y_v, color=color_param, marginal="box", barmode="overlay", opacity=0.65, labels=todas_v, color_discrete_sequence=PALETA_AZUL)

    fig_m.update_layout(height=520, title=f"{tipo_g}: {VARS_NUM[y_v]} según {todas_v[x_v]}")
    st.plotly_chart(fig_m, use_container_width=True)
    st.caption("📷 Para exportar esta gráfica en alta resolución para tu tesis o sustentación, haz clic en el ícono de cámara en la esquina superior derecha del gráfico.")

# ============================================================
# PESTAÑA 6: SIMULADOR PROSPECTIVO DE BRECHAS Y PRODUCTIVIDAD
# ============================================================
with tab6:
    st.markdown("### Simulador Prospectivo de Adopción TIC y Cierre de Brechas en Pasto")
    st.write("Evalúa el cambio esperado en la productividad laboral mensual de un micronegocio utilizando los parámetros oficiales estimados en la Tabla 4.5:")

    s1, s2, s3 = st.columns(3)
    with s1:
        sim_imtic = st.slider("Nivel IMTIC objetivo (0.00 = sin TIC, 1.00 = adopción plena)", 0.0, 1.0, 0.40, 0.05)
        sim_mod = st.selectbox(
            "Estimador de referencia (Tabla 4.5)",
            ["OLS_HC1", "Quantile_0.10", "Quantile_0.50", "Quantile_0.90", "Huber_M"],
            format_func=lambda m: {
                "OLS_HC1": "MCO HC1 (Efecto promedio)",
                "Quantile_0.10": "Cuantil τ = 0.10 (Subsistencia)",
                "Quantile_0.50": "Cuantil τ = 0.50 (Mediana)",
                "Quantile_0.90": "Cuantil τ = 0.90 (Consolidados)",
                "Huber_M": "M-Estimador Robusto de Huber"
            }[m]
        )
    with s2:
        sim_edad = st.number_input("Edad del propietario (años)", 18, 85, 42)
        sim_antig = st.number_input("Antigüedad del micronegocio (años)", 0.0, 50.0, 5.0, 0.5)
    with s3:
        sim_mujer = st.radio("Sexo del propietario(a)", [0, 1], format_func=lambda x: "Mujer" if x == 1 else "Hombre", horizontal=True)
        sim_patron = st.radio("Categoría", [0, 1], format_func=lambda x: "Empleador (Patrón)" if x == 1 else "Cuenta Propia", horizontal=True)
        sim_empleo = st.slider("Personal ocupado total (L)", 1, 9, 2 if sim_patron == 1 else 1)

    c_mod = df_res_oficial[df_res_oficial["modelo"] == sim_mod].set_index("variable")["coef"]

    def est_ln(val_tic):
        return (
            c_mod.get("const", 0)
            + c_mod.get("IMTIC", 0) * val_tic
            + c_mod.get("edad", 0) * sim_edad
            + c_mod.get("mujer", 0) * sim_mujer
            + c_mod.get("antiguedad", 0) * sim_antig
            + c_mod.get("patron", 0) * sim_patron
            + c_mod.get("empleo", 0) * sim_empleo
        )

    y_sin_tic = np.exp(est_ln(0.0))
    y_con_tic = np.exp(est_ln(sim_imtic))
    var_pct = ((y_con_tic / y_sin_tic) - 1) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Productividad base (IMTIC = 0.00)", f"${y_sin_tic:,.0f} / mes")
    m2.metric(f"Productividad proyectada (IMTIC = {sim_imtic:.2f})", f"${y_con_tic:,.0f} / mes", f"+{var_pct:.1f}%")
    m3.metric("Diferencial mensual por ocupado", f"+${(y_con_tic - y_sin_tic):,.0f}")

    # Curva completa de trayectoria IMTIC 0 -> 1 para el perfil configurado
    grid_tic = np.linspace(0, 1, 50)
    curva_df = pd.DataFrame({
        "IMTIC": grid_tic,
        "Productividad_Estimada": [np.exp(est_ln(t)) for t in grid_tic]
    })
    fig_sim = px.area(
        curva_df, x="IMTIC", y="Productividad_Estimada",
        color_discrete_sequence=["#1E4E8C"],
        labels={"IMTIC": "Índice IMTIC (0–1)", "Productividad_Estimada": "Productividad mensual estimada (COP)"}
    )
    fig_sim.add_vline(x=sim_imtic, line_dash="dash", line_color="#D97706", annotation_text=f"IMTIC = {sim_imtic:.2f}")
    fig_sim.update_layout(height=340, title="Curva de Respuesta de Productividad ante Incrementos en el IMTIC")
    st.plotly_chart(fig_sim, use_container_width=True)
