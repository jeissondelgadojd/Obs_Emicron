import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
df_res = pd.DataFrame(modelos_json["coeficientes"])

# Diccionario descriptivo de las 16 variables TIC de EMICRON
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

# ------------------------------------------------------------
# BARRA LATERAL · FILTROS GLOBALES
# ------------------------------------------------------------
st.sidebar.title("Filtros Territoriales")
st.sidebar.markdown("**San Juan de Pasto (Cabecera Municipal)**\nMicrodatos EMICRON · DANE (2019–2024)")

anios_disp = sorted(df["anio"].dropna().unique().astype(int))
anios_sel = st.sidebar.multiselect("Años de estudio", anios_disp, default=anios_disp)

sectores_disp = sorted(df["sector_nombre"].dropna().unique())
sectores_sel = st.sidebar.multiselect("Sector económico", sectores_disp, default=sectores_disp)

usar_fexp = st.sidebar.toggle("Ponderar con Factor de Expansión (F_EXP)", value=True)

df_f = df[df["anio"].isin(anios_sel) & df["sector_nombre"].isin(sectores_sel)].copy()

# ------------------------------------------------------------
# ENCABEZADO PRINCIPAL
# ------------------------------------------------------------
st.title("📊 Observatorio de Adopción de TIC y Productividad Laboral")
st.caption(
    "Micronegocios (L ≤ 9) de San Juan de Pasto, Nariño (2019–2024) · "
    "Base empírica de investigación económica (Universidad de Nariño)"
)

if df_f.empty:
    st.warning("Selecciona al menos un año y un sector en el panel izquierdo.")
    st.stop()

# KPIs superiores
col1, col2, col3, col4 = st.columns(4)
if usar_fexp and "F_EXP" in df_f.columns:
    total_unidades = df_f["F_EXP"].sum()
    etiqueta_n = "Micronegocios expandidos (F_EXP)"
else:
    total_unidades = len(df_f)
    etiqueta_n = "Muestra encuestada (n)"

prod_mediana = df_f["prod_laboral"].median()
imtic_prom = df_f["IMTIC"].mean()
pct_uso_tic = (df_f["IMTIC"] > 0).mean() * 100

col1.metric(etiqueta_n, f"{total_unidades:,.0f}")
col2.metric("Productividad Laboral Mediana", f"${prod_mediana:,.0f} / mes")
col3.metric("Índice IMTIC Promedio (0–1)", f"{imtic_prom:.3f}")
col4.metric("Unidades con algún uso TIC", f"{pct_uso_tic:.1f}%")

st.divider()

# ------------------------------------------------------------
# PESTAÑAS DEL OBSERVATORIO
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Panorama Territorial",
    "2. Brecha Digital e Índice IMTIC",
    "3. Modelos Econométricos",
    "4. Simulador de Productividad"
])

# === PESTAÑA 1: PANORAMA TERRITORIAL ===
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Evolución de Productividad Laboral (ln) por Año")
        fig_box = px.box(
            df_f.dropna(subset=["ln_prod_laboral"]),
            x="anio", y="ln_prod_laboral", color="IMTIC_nivel",
            labels={"anio": "Año", "ln_prod_laboral": "ln(Productividad Laboral)", "IMTIC_nivel": "Nivel TIC"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with c2:
        st.subheader("Distribución Sectorial de Micronegocios")
        if usar_fexp:
            sec_df = df_f.groupby("sector_nombre", as_index=False)["F_EXP"].sum().rename(columns={"F_EXP": "Unidades"})
        else:
            sec_df = df_f.groupby("sector_nombre", as_index=False).size().rename(columns={"size": "Unidades"})
        fig_sec = px.pie(sec_df, names="sector_nombre", values="Unidades", hole=0.45)
        st.plotly_chart(fig_sec, use_container_width=True)

    st.subheader("Productividad Mediana por Sitio de Funcionamiento y Género del Propietario")
    df_f["Genero"] = np.where(df_f["mujer"] == 1, "Mujer", "Hombre")
    ubic_df = (
        df_f.dropna(subset=["prod_laboral"])
        .groupby(["ubicacion_nombre", "Genero"], as_index=False)["prod_laboral"]
        .median()
    )
    fig_ubic = px.bar(
        ubic_df, x="ubicacion_nombre", y="prod_laboral", color="Genero", barmode="group",
        labels={"ubicacion_nombre": "Ubicación del micronegocio", "prod_laboral": "Productividad mediana (COP/mes)"}
    )
    st.plotly_chart(fig_ubic, use_container_width=True)

# === PESTAÑA 2: BRECHA DIGITAL E ÍNDICE IMTIC ===
with tab2:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Pesos Factoriales del Primer Componente Principal (ACP)")
        fig_cargas = px.bar(
            cargas.sort_values("carga_pc1", ascending=True),
            x="carga_pc1", y="descripcion", orientation="h",
            labels={"carga_pc1": "Carga factorial (PC1)", "descripcion": "Indicador TIC EMICRON"},
            color="carga_pc1", color_continuous_scale="Blues"
        )
        fig_cargas.update_layout(showlegend=False, height=520)
        st.plotly_chart(fig_cargas, use_container_width=True)

    with c2:
        st.subheader("Adopción TIC por Nivel y Año (%)")
        niv_anio = pd.crosstab(df_f["anio"], df_f["IMTIC_nivel"], normalize="index") * 100
        niv_anio = niv_anio.reset_index().melt(id_vars="anio", var_name="Nivel", value_name="Porcentaje")
        fig_niv = px.bar(
            niv_anio, x="anio", y="Porcentaje", color="Nivel", barmode="stack",
            labels={"anio": "Año", "Porcentaje": "Participación (%)"}
        )
        st.plotly_chart(fig_niv, use_container_width=True)

# === PESTAÑA 3: MODELOS ECONOMÉTRICOS ===
with tab3:
    st.subheader(f"Estimaciones Econométricas · Muestra Efectiva N = {modelos_json['n_obs']:,}")
    st.markdown(
        f"**R² MCO (HC1):** `{modelos_json['r2']['OLS_HC1']:.4f}` · "
        f"**R² Ajustado:** `{modelos_json['r2']['OLS_HC1_adj']:.4f}` · "
        "Incluye efectos fijos sectoriales (`GRUPOS4`) y temporales (`2019–2024`)."
    )

    sub_imtic = df_res[df_res["variable"] == "IMTIC"].copy()
    nombres_mod = {
        "OLS_HC1": "MCO (HC1)",
        "Quantile_0.10": "Cuantil τ = 0.10 (Subsistencia)",
        "Quantile_0.50": "Cuantil τ = 0.50 (Mediana)",
        "Quantile_0.90": "Cuantil τ = 0.90 (Consolidados)",
        "Huber_M": "M-Estimador de Huber"
    }
    sub_imtic["Modelo"] = sub_imtic["modelo"].map(nombres_mod)

    fig_coef = go.Figure()
    fig_coef.add_trace(go.Scatter(
        x=sub_imtic["coef"], y=sub_imtic["Modelo"],
        mode="markers",
        marker=dict(size=12, color="#1F4E78"),
        error_x=dict(
            type="data",
            symmetric=False,
            array=sub_imtic["ci_high"] - sub_imtic["coef"],
            arrayminus=sub_imtic["coef"] - sub_imtic["ci_low"],
            thickness=2.5, width=6
        ),
        name="Coeficiente IMTIC (IC 95%)"
    ))
    fig_coef.update_layout(
        title="Gradiente del Retorno Marginal de las TIC según Método y Cuantil",
        xaxis_title="Efecto estimado sobre ln(Productividad Laboral)",
        yaxis=dict(autorange="reversed"),
        height=400
    )
    st.plotly_chart(fig_coef, use_container_width=True)

    st.subheader("Tabla Comparativa Completa de Coeficientes")
    vars_principales = ["const", "IMTIC", "edad", "mujer", "antiguedad", "patron", "empleo"]
    tabla_piv = (
        df_res[df_res["variable"].isin(vars_principales)]
        .pivot(index="variable", columns="modelo", values="coef")
        .reindex(vars_principales)
    )
    st.dataframe(tabla_piv.style.format("{:.4f}"), use_container_width=True)

# === PESTAÑA 4: SIMULADOR DE PRODUCTIVIDAD ===
with tab4:
    st.subheader("Simulador de Impacto de Adopción TIC en Productividad Laboral")
    st.write("Ajusta el perfil del micronegocio para estimar su productividad mensual esperada según los parámetros estimados en Pasto:")

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

    coefs_mod = df_res[df_res["modelo"] == sim_modelo].set_index("variable")["coef"]

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
