import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine, text
import folium
from streamlit_folium import st_folium
import requests
import io
import os

# ================== Configuración ==================
st.set_page_config(page_title="Dashboard Logístico", layout="wide")
st.header("📦 Dashboard Predictivo de Entregas - ChivoFast")
st.markdown("Análisis y predicción de tiempos de entrega usando Inteligencia Artificial")

# ================== Base de Datos ==================
DATABASE_URL = st.secrets.get("DATABASE_URL", None)

if not DATABASE_URL:
    st.error("❌ No se encontró DATABASE_URL en los Secrets de Streamlit")
else:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
    engine = create_engine(DATABASE_URL, connect_args={"sslmode":"require"})
    try:
        with engine.connect() as conn:
            test = conn.execute(text("SELECT 1")).scalar()
            st.success(f"✅ Conexión PostgreSQL establecida (SELECT 1 = {test})")
    except Exception as e:
        st.error(f"❌ Error al conectar a la base de datos: {e}")

# ================== Cargar Datos ==================
@st.cache_data
def load_data():
    if not DATABASE_URL:
        return pd.DataFrame()
    df = pd.read_sql("SELECT * FROM entregas", engine)
    return df

df = load_data()

# ================== Exportar Excel ==================
def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="Entregas")
    writer.close()
    processed_data = output.getvalue()
    return processed_data

if not df.empty:
    st.download_button(
        label="📥 Exportar datos a Excel",
        data=to_excel(df),
        file_name="entregas.xlsx"
    )

# ================== KPIs ==================
if not df.empty:
    st.subheader("📌 Indicadores Clave (KPIs)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Promedio de Entrega (min)", round(df["tiempo_entrega"].mean(), 2))
    col2.metric("Retraso Promedio (min)", round(df["retraso"].mean(), 2))
    col3.metric("Total de Entregas", len(df))

# ================== Gráficos ==================
if not df.empty:
    st.subheader("📍 Distribución de Entregas por Zona")
    st.plotly_chart(px.histogram(df, x="zona", color="zona", title="Número de Entregas por Zona"))

    st.subheader("🚦 Impacto del Tráfico en Tiempo de Entrega")
    st.plotly_chart(px.box(df, x="trafico", y="tiempo_entrega", color="trafico"))

    st.subheader("🌦️ Impacto del Clima en Tiempo de Entrega")
    st.plotly_chart(px.box(df, x="clima", y="tiempo_entrega", color="clima"))

# ================== Mapa ==================
if not df.empty:
    st.subheader("🗺️ Mapa de Entregas en El Salvador")
    mapa = folium.Map(location=[13.7, -88.8], zoom_start=7)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[13.7 + (row.name%10)*0.01, -88.8 + (row.name%10)*0.01],
            radius=5,
            popup=f"Zona: {row['zona']}\nTiempo: {row['tiempo_entrega']} min",
            color="blue",
            fill=True
        ).add_to(mapa)
    st_folium(mapa, width=700, height=500)

# ================== Modelo de Predicción ==================
if not df.empty:
    st.subheader("🤖 Predicción de Tiempo de Entrega")
    df_ml = pd.get_dummies(df.drop(columns=["id_entrega","fecha"]), drop_first=True)
    X = df_ml.drop(columns=["tiempo_entrega"])
    y = df_ml["tiempo_entrega"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    st.write(f"MAE: {round(mae,2)} | RMSE: {round(rmse,2)} | R²: {round(r2,2)}")

    st.subheader("🔮 Estimar un nuevo pedido")
    zona = st.selectbox("Zona", df["zona"].unique())
    tipo_pedido = st.selectbox("Tipo de pedido", df["tipo_pedido"].unique())
    clima = st.selectbox("Clima", df["clima"].unique())
    trafico = st.selectbox("Tráfico", df["trafico"].unique())
    retraso = st.slider("Retraso estimado", 0, 30, 5)

    nuevo = pd.DataFrame([[zona, tipo_pedido, clima, trafico, retraso]],
                         columns=["zona","tipo_pedido","clima","trafico","retraso"])
    nuevo_ml = pd.get_dummies(nuevo)
    nuevo_ml = nuevo_ml.reindex(columns=X.columns, fill_value=0)
    prediccion = model.predict(nuevo_ml)[0]
    st.success(f"⏱️ Tiempo estimado de entrega: {round(prediccion,2)} minutos")
