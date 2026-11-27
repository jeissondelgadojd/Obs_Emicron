import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.tools.tools import add_constant
import streamlit as st

# ------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="Observatorio TIC · Micronegocios Pasto (2019–2024)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = Path("data")

@st.cache_data
def cargar_datos():
    df = pd.read_parquet(DATA_DIR / "EMICRON_Pasto_Maestra_2019_2024.parquet")
    cargas = pd.read_csv(DATA_DIR / "cargas_acp_imtic.csv")
    with open(DATA_DIR / "resultados_modelos_pasto.json", "r", encoding="utf-8") as f:
        modelos = json.load(f)
    return df, cargas, modelos

df, cargas, modelos_json = cargar_datos()

DESC_TIC = {
    "P2532": "Usa Internet en las operaciones del negocio",
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

# Etiquetas legibles para el constructor manual de gráficos
VARS_NUMERICAS = {
    "ln_prod_laboral": "ln(Productividad Laboral)",
    "prod_laboral": "Productividad Laboral (COP/mes)",
    "IMTIC": "Índice IMTIC (0–1)",
    "VALOR_AGREGADO_HOMOLOGADO": "Valor Agregado Mensual (COP)",
    "VENTAS_MES_ANTERIOR": "Ventas Mes Anterior (COP)",
    "COSTOS_MES_ANTERIOR": "Costos Mes Anterior (COP)",
    "GASTOS_MES": "Gastos Operativos Mes (COP)",
    "empleo": "Personal Ocupado Total (L)",
    "edad": "Edad del Propietario (años)",
    "antiguedad": "Antigüedad del Negocio (años)",
    "m4_total_ayudantes": "Número de Ayudantes"
}

VARS_CATEGORICAS = {
    "anio": "Año de Encuesta",
    "sector_nombre": "Sector Económico (GRUPOS4)",
    "IMTIC_nivel": "Nivel de Adopción TIC",
    "ubicacion_nombre": "Sitio de Funcionamiento",
    "Genero": "Género del Propietario",
    "Tipo_Unidad": "Tipo de Micronegocio (Cuenta Propia vs Patrón)"
}

df["Genero"] = np.where(df["mujer"] == 1, "Mujer", "Hombre")
df["Tipo_Unidad"] = np.where(df["patron"] == 1, "Empleador (Patrón)", "Cuenta Propia")

# ------------------------------------------------------------
# FUNCIÓN CACHEADA DE ESTIMACIÓN ECONOMÉTRICA EN VIVO
# ------------------------------------------------------------
@st.cache_data
def estimar_modelos_en_vivo(df_sub, cuantil_extra=0.75, incluir_fe_sector=True, incluir_fe_anio=True):
    vars_req = ["ln_prod_laboral", "IMTIC", "edad", "mujer", "antiguedad", "patron", "empleo", "sector", "anio_estudio"]
    d = df_sub.dropna(subset=vars_req).copy()
    d = d[np.isfinite(d["ln_prod_laboral"])].copy()

    if len(d) < 50:
        return None, None, len(d)

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

    # 1. MCO HC1
    m_ols = sm.OLS(y, X).fit(cov_type="HC1")
    # 2. MCO Ponderado por F_EXP (WLS)
    w = d["F_EXP"].fillna(1.0).astype(float)
    m_wls = sm.WLS(y, X, weights=w).fit(cov_type="HC1")
    # 3. Regresión Cuantílica (0.10, 0.50, 0.90 + cuantil personalizado)
    m_q10 = QuantReg(y, X).fit(q=0.10, max_iter=1500)
    m_q50 = QuantReg(y, X).fit(q=0.50, max_iter=1500)
    m_q90 = QuantReg(y, X).fit(q=0.90, max_iter=1500)
    m_qext = QuantReg(y, X).fit(q=cuantil_extra, max_iter=1500)
    # 4. M-Estimador de Huber
    m_huber = RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    # 5. M-Estimador Bicuadrado de Tukey
    m_tukey = RLM(y, X, M=sm.robust.norms.TukeyBiweight()).fit()

    modelos_dict = {
        "MCO (HC1)": m_ols,
        "MCO Ponderado (F_EXP)": m_wls,
        "Cuantil τ = 0.10": m_q10,
        "Cuantil τ = 0.50": m_q50,
        f"Cuantil τ = {cuantil_extra:.2f}": m_qext,
        "Cuantil τ = 0.90": m_q90,
        "Huber RLM": m_huber,
        "Tukey Biweight RLM": m_tukey
    }

    filas = []
    for nombre, mod in modelos_dict.items():
        ci = mod.conf_int()
        tmp = pd.DataFrame({
            "variable": mod.params.index,
            "coef": mod.params.values,
            "std_err": mod.bse.values,
            "pvalue": mod.pvalues.values,
            "ci_low": ci[0].values,
            "ci_high": ci[1].values,
            "modelo": nombre
        })
        filas.append(tmp)

    metricas = {
        "R2_OLS": float(m_ols.rsquared),
        "R2_OLS_adj": float(m_ols.rsquared_adj),
        "R2_WLS": float(m_wls.rsquared),
        "PseudoR2_Q50": float(m_q50.prsquared)
    }
    return pd.concat(filas, ignore_index=True), metricas, len(d)

# ------------------------------------------------------------
# BARRA LATERAL · FILTROS GLOBALES
# ------------------------------------------------------------
st.sidebar.title("Filtros Territoriales")
st.sidebar.markdown("**San Juan de Pasto (Cabecera Municipal)**\nMicrodatos EMICRON · DANE (2019–2024)")

anios_disp = sorted(df["anio"].dropna().unique().astype(int))
anios_sel = st.sidebar.multiselect("Años de estudio", anios_disp, default=anios_disp)

sectores_disp = sorted(df["sector_nombre"].dropna().unique())
sectores_sel = st.sidebar.multiselect("Sector económico", sectores_disp, default=sectores_disp)

tipos_disp = ["Cuenta Propia", "Empleador (Patrón)"]
tipos_sel = st.sidebar.multiselect("Tipo de micronegocio", tipos_disp, default=tipos_disp)

usar_fexp = st.sidebar.toggle("Ponderar KPIs con Factor de Expansión (F_EXP)", value=True)

df_f = df[
    df["anio"].isin(anios_sel) &
    df["sector_nombre"].isin(sectores_sel) &
    df["Tipo_Unidad"].isin(tipos_sel)
].copy()

# ------------------------------------------------------------
# ENCABEZADO Y KPIs
# ------------------------------------------------------------
st.title("📊 Observatorio de Adopción de TIC y Productividad Laboral")
st.caption("Micronegocios (L ≤ 9) de San Juan de Pasto, Nariño (2019–2024) · Universidad de Nariño")

if df_f.empty:
    st.warning("Selecciona al menos una opción en cada filtro de la barra lateral.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
if usar_fexp and "F_EXP" in df_f.columns:
    total_unidades = df_f["F_EXP"].sum()
    etiqueta_n = "Micronegocios expandidos (F_EXP)"
else:
    total_unidades = len(df_f)
    etiqueta_n = "Muestra encuestada (n)"

col1.metric(etiqueta_n, f"{total_unidades:,.0f}")
col2.metric("Productividad Laboral Mediana", f"${df_f['prod_laboral'].median():,.0f} / mes")
col3.metric("Índice IMTIC Promedio (0–1)", f"{df_f['IMTIC'].mean():.3f}")
col4.metric("Unidades con algún uso TIC", f"{(df_f['IMTIC'] > 0).mean() * 100:.1f}%")

st.divider()

# ------------------------------------------------------------
# 5 PESTAÑAS INTERACTIVAS
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Panorama Territorial",
    "2. Índice IMTIC (ACP)",
    "3. Econometría en Vivo (Todos los Métodos)",
    "4. Constructor Manual de Gráficos",
    "5. Simulador de Productividad"
])

# === PESTAÑA 1: PANORAMA TERRITORIAL ===
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Productividad Laboral (ln) por Año y Nivel TIC")
        fig_box = px.box(
            df_f.dropna(subset=["ln_prod_laboral"]),
            x="anio", y="ln_prod_laboral", color="IMTIC_nivel",
            labels={"anio": "Año", "ln_prod_laboral": "ln(Productividad Laboral)", "IMTIC_nivel": "Nivel TIC"}
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with c2:
        st.subheader("Composición Sectorial")
        if usar_fexp:
            sec_df = df_f.groupby("sector_nombre", as_index=False)["F_EXP"].sum().rename(columns={"F_EXP": "Unidades"})
        else:
            sec_df = df_f.groupby("sector_nombre", as_index=False).size().rename(columns={"size": "Unidades"})
        fig_sec = px.pie(sec_df, names="sector_nombre", values="Unidades", hole=0.45)
        st.plotly_chart(fig_sec, use_container_width=True)

# === PESTAÑA 2: ÍNDICE IMTIC ===
with tab2:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Cargas Factoriales del Primer Componente Principal (ACP)")
        fig_cargas = px.bar(
            cargas.sort_values("carga_pc1", ascending=True),
            x="carga_pc1", y="descripcion", orientation="h",
            labels={"carga_pc1": "Carga factorial (PC1)", "descripcion": "Indicador TIC"},
            color="carga_pc1", color_continuous_scale="Blues"
        )
        fig_cargas.update_layout(showlegend=False, height=520)
        st.plotly_chart(fig_cargas, use_container_width=True)

    with c2:
        st.subheader("Evolución de Niveles de Adopción TIC (%)")
        niv_anio = pd.crosstab(df_f["anio"], df_f["IMTIC_nivel"], normalize="index") * 100
        niv_anio = niv_anio.reset_index().melt(id_vars="anio", var_name="Nivel", value_name="Porcentaje")
        fig_niv = px.bar(niv_anio, x="anio", y="Porcentaje", color="Nivel", barmode="stack")
        st.plotly_chart(fig_niv, use_container_width=True)

# === PESTAÑA 3: ECONOMETRÍA EN VIVO (TODOS LOS MÉTODOS) ===
with tab3:
    st.subheader("Estimación Econométrica Dinámica (Recalcula según filtros seleccionados)")

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        tau_custom = st.slider("Cuantil adicional personalizado (τ)", 0.05, 0.95, 0.75, 0.05)
    with ctrl2:
        fe_sec = st.checkbox("Incluir Efectos Fijos Sectoriales (GRUPOS4)", value=True)
    with ctrl3:
        fe_anio = st.checkbox("Incluir Efectos Fijos Temporales (Año)", value=True)

    res_vivo, met_vivo, n_vivo = estimar_modelos_en_vivo(df_f, tau_custom, fe_sec, fe_anio)

    if res_vivo is None:
        st.error("Muestra insuficiente con los filtros actuales para estimar los modelos (se requieren al menos 50 observaciones).")
    else:
        st.info(
            f"**Muestra efectiva estimada:** `N = {n_vivo:,}` | "
            f"**R² MCO (HC1):** `{met_vivo['R2_OLS']:.4f}` | "
            f"**R² Ajustado:** `{met_vivo['R2_OLS_adj']:.4f}` | "
            f"**R² Ponderado (F_EXP):** `{met_vivo['R2_WLS']:.4f}` | "
            f"**Pseudo-R² (Q0.50):** `{met_vivo['PseudoR2_Q50']:.4f}`"
        )

        var_eval = st.selectbox(
            "Selecciona la variable cuyo efecto deseas comparar entre los 8 estimadores:",
            ["IMTIC", "mujer", "patron", "antiguedad", "edad", "empleo"],
            format_func=lambda v: {
                "IMTIC": "Índice Multidimensional TIC (IMTIC)",
                "mujer": "Brecha de Género (Mujer = 1)",
                "patron": "Condición de Empleador (Patrón = 1)",
                "antiguedad": "Antigüedad del Negocio (años)",
                "edad": "Edad del Propietario (años)",
                "empleo": "Tamaño de Personal Ocupado (L)"
            }[v]
        )

        sub_v = res_vivo[res_vivo["variable"] == var_eval].copy()
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(
            x=sub_v["coef"], y=sub_v["modelo"],
            mode="markers",
            marker=dict(size=12, color="#1F4E78"),
            error_x=dict(
                type="data", symmetric=False,
                array=sub_v["ci_high"] - sub_v["coef"],
                arrayminus=sub_v["coef"] - sub_v["ci_low"],
                thickness=2.5, width=6
            )
        ))
        fig_comp.add_vline(x=0, line_dash="dash", line_color="crimson")
        fig_comp.update_layout(
            title=f"Comparación de Coeficientes e IC 95% para '{var_eval}' (8 Especificaciones)",
            xaxis_title="Coeficiente estimado",
            yaxis=dict(autorange="reversed"),
            height=430
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.subheader("Matriz Completa de Coeficientes y Errores Estándar")
        vars_tab = ["const", "IMTIC", "edad", "mujer", "antiguedad", "patron", "empleo"]
        piv_coef = res_vivo[res_vivo["variable"].isin(vars_tab)].pivot(index="variable", columns="modelo", values="coef").reindex(vars_tab)
        st.dataframe(piv_coef.style.format("{:.4f}"), use_container_width=True)

        csv_descarga = res_vivo.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇ Descargar todos los resultados econométricos filtrados (.csv)",
            data=csv_descarga,
            file_name=f"modelos_pasto_N{n_vivo}.csv",
            mime="text/csv"
        )

# === PESTAÑA 4: CONSTRUCTOR MANUAL DE GRÁFICOS ===
with tab4:
    st.subheader("Constructor Manual e Interactivo de Gráficos")
    st.write("Configura libremente los ejes, el tipo de visualización, la agrupación y los filtros de valores atípicos:")

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        tipo_graf = st.selectbox(
            "Tipo de gráfico",
            ["Dispersión (Scatter + Tendencia OLS)", "Diagrama de Cajas (Boxplot)",
             "Diagrama de Violín", "Barras Agrupadas (Mediana / Media)", "Histograma / Distribución"]
        )
    with g2:
        todas_vars = {**VARS_NUMERICAS, **VARS_CATEGORICAS}
        eje_x = st.selectbox("Variable Eje X", list(todas_vars.keys()), index=2, format_func=lambda k: todas_vars[k])
    with g3:
        eje_y = st.selectbox("Variable Eje Y (numérica)", list(VARS_NUMERICAS.keys()), index=0, format_func=lambda k: VARS_NUMERICAS[k])
    with g4:
        color_var = st.selectbox(
            "Agrupar / Colorear por",
            ["Ninguno"] + list(VARS_CATEGORICAS.keys()),
            index=2,
            format_func=lambda k: "Sin agrupación" if k == "Ninguno" else VARS_CATEGORICAS[k]
        )

    opt1, opt2 = st.columns(2)
    with opt1:
        recortar_outliers = st.checkbox("Recortar extremos (percentiles 1% - 99%) para mejor visualización", value=False)
    with opt2:
        agg_func = st.radio("Estadístico (para gráficos de barras)", ["Mediana", "Media"], horizontal=True)

    df_plot = df_f.copy()
    if recortar_outliers and eje_y in VARS_NUMERICAS:
        p01, p99 = df_plot[eje_y].quantile([0.01, 0.99])
        df_plot = df_plot[df_plot[eje_y].between(p01, p99)]

    color_arg = None if color_var == "Ninguno" else color_var
    labels_dict = {**todas_vars}

    if tipo_graf == "Dispersión (Scatter + Tendencia OLS)":
        fig_manual = px.scatter(
            df_plot.dropna(subset=[eje_x, eje_y]),
            x=eje_x, y=eje_y, color=color_arg,
            trendline="ols", opacity=0.35, labels=labels_dict
        )
    elif tipo_graf == "Diagrama de Cajas (Boxplot)":
        fig_manual = px.box(
            df_plot.dropna(subset=[eje_x, eje_y]),
            x=eje_x, y=eje_y, color=color_arg, labels=labels_dict
        )
    elif tipo_graf == "Diagrama de Violín":
        fig_manual = px.violin(
            df_plot.dropna(subset=[eje_x, eje_y]),
            x=eje_x, y=eje_y, color=color_arg, box=True, labels=labels_dict
        )
    elif tipo_graf == "Barras Agrupadas (Mediana / Media)":
        cols_grp = [eje_x] + ([color_arg] if color_arg and color_arg != eje_x else [])
        if agg_func == "Mediana":
            df_bar = df_plot.groupby(cols_grp, as_index=False)[eje_y].median()
        else:
            df_bar = df_plot.groupby(cols_grp, as_index=False)[eje_y].mean()
        fig_manual = px.bar(
            df_bar, x=eje_x, y=eje_y, color=color_arg, barmode="group", labels=labels_dict
        )
    else:  # Histograma
        fig_manual = px.histogram(
            df_plot.dropna(subset=[eje_y]),
            x=eje_y, color=color_arg, marginal="box", barmode="overlay", opacity=0.65, labels=labels_dict
        )

    fig_manual.update_layout(height=540, title=f"{tipo_graf}: {todas_vars[eje_x]} vs. {VARS_NUMERICAS[eje_y]}")
    st.plotly_chart(fig_manual, use_container_width=True)
    st.caption("💡 Puedes descargar cualquier gráfico como imagen PNG de alta resolución usando el ícono de cámara 📷 en la esquina superior derecha del gráfico.")

# === PESTAÑA 5: SIMULADOR DE PRODUCTIVIDAD ===
with tab5:
    st.subheader("Simulador de Impacto de Adopción TIC en Productividad Laboral")
    res_base = pd.DataFrame(modelos_json["coeficientes"])

    c1, c2, c3 = st.columns(3)
    with c1:
        sim_imtic = st.slider("Nivel IMTIC propuesto (0 = nulo, 1 = máximo)", 0.0, 1.0, 0.35, 0.05)
        sim_modelo = st.selectbox(
            "Perfil productivo (estrato)",
            ["OLS_HC1", "Quantile_0.10", "Quantile_0.50", "Quantile_0.90", "Huber_M"],
            format_func=lambda x: {
                "OLS_HC1": "Promedio general (MCO HC1)",
                "Quantile_0.10": "Estrato bajo / subsistencia (τ = 0.10)",
                "Quantile_0.50": "Estrato intermedio (τ = 0.50)",
                "Quantile_0.90": "Estrato alto / consolidado (τ = 0.90)",
                "Huber_M": "Estimación robusta (Huber)"
            }[x]
        )
    with c2:
        sim_edad = st.number_input("Edad del propietario (años)", 18, 85, 42)
        sim_antig = st.number_input("Antigüedad del negocio (años)", 0.0, 50.0, 5.0, 0.5)
    with c3:
        sim_mujer = st.radio("Propietario(a)", [0, 1], format_func=lambda x: "Mujer" if x == 1 else "Hombre", horizontal=True)
        sim_patron = st.radio("Tipo de unidad", [0, 1], format_func=lambda x: "Empleador (Patrón)" if x == 1 else "Cuenta propia", horizontal=True)
        sim_empleo = st.slider("Personal ocupado total (L)", 1, 9, 2 if sim_patron == 1 else 1)

    coefs_mod = res_base[res_base["modelo"] == sim_modelo].set_index("variable")["coef"]

    def predecir_ln(val_imtic):
        return (
            coefs_mod.get("const", 0)
            + coefs_mod.get("IMTIC", 0) * val_imtic
            + coefs_mod.get("edad", 0) * sim_edad
            + coefs_mod.get("mujer", 0) * sim_mujer
            + coefs_mod.get("antiguedad", 0) * sim_antig
            + coefs_mod.get("patron", 0) * sim_patron
            + coefs_mod.get("empleo", 0) * sim_empleo
        )

    prod_base = np.exp(predecir_ln(0.0))
    prod_sim = np.exp(predecir_ln(sim_imtic))
    ganancia_pct = ((prod_sim / prod_base) - 1) * 100

    k1, k2, k3 = st.columns(3)
    k1.metric("Productividad sin TIC (IMTIC = 0)", f"${prod_base:,.0f} / mes")
    k2.metric(f"Productividad con IMTIC = {sim_imtic:.2f}", f"${prod_sim:,.0f} / mes", f"+{ganancia_pct:.1f}%")
    k3.metric("Ganancia mensual por trabajador", f"+${(prod_sim - prod_base):,.0f}")
