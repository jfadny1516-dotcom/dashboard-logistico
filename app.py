# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine
import os
import folium
from streamlit_folium import st_folium
import requests
from io import BytesIO

# ============================================================
# Configuración
# ============================================================
st.set_page_config(page_title="Dashboard Predictivo de Entregas", layout="wide")

st.title("📦 Dashboard Predictivo de Entregas - ChivoFast")

# ==============================================
# DATABASE_URL
# ==============================================
DB_URL = os.getenv("DATABASE_URL")

if not DB_URL:
    st.error(
        "❌ No se encontró DATABASE_URL en los Secrets de Streamlit. "
        "Agrega tu conexión PostgreSQL en Secrets con la clave DATABASE_URL."
    )
    st.stop()

# Ajuste para SQLAlchemy
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DB_URL.startswith("postgresql://"):
    DB_URL = DB_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

# Crear motor SQLAlchemy
engine = create_engine(DB_URL, connect_args={"sslmode": "require"})

# ============================================================
# Función para cargar datos
# ============================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_sql("SELECT * FROM entregas", engine)
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {e}")
        return pd.DataFrame()

df = load_data()

# ============================================================
# Exportar a Excel
# ============================================================
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="Entregas")
    writer.save()
    processed_data = output.getvalue()
    return processed_data

# ============================================================
# Añadir nuevas entregas
# ============================================================
st.subheader("➕ Agregar nueva entrega")

zonas_validas = ["San Salvador", "San Miguel", "Santa Ana", "La Libertad"]

with st.form("agregar_entrega"):
    fecha = st.date_input("Fecha")
    zona = st.selectbox("Zona", zonas_validas)
    tipo_pedido = st.selectbox("Tipo de pedido", ["Supermercado", "Restaurante", "Tienda en línea", "Farmacia"])
    clima = st.selectbox("Clima", ["Soleado", "Lluvioso", "Nublado"])
    trafico = st.selectbox("Tráfico", ["Bajo", "Medio", "Alto"])
    tiempo_entrega = st.number_input("Tiempo de entrega (min)", min_value=1, max_value=300, value=30)
    retraso = st.number_input("Retraso (min)", min_value=0, max_value=60, value=5)
    submitted = st.form_submit_button("Agregar")

    if submitted:
        try:
            query = f"""
            INSERT INTO entregas (fecha, zona, tipo_pedido, clima, trafico, tiempo_entrega, retraso)
            VALUES ('{fecha}', '{zona}', '{tipo_pedido}', '{clima}', '{trafico}', {tiempo_entrega}, {retraso})
            """
            with engine.connect() as conn:
                conn.execute(query)
                st.success("✅ Entrega agregada correctamente")
            df = load_data()  # recargar datos
        except Exception as e:
            st.error(f"❌ Error al agregar entrega: {e}")

# ============================================================
# Cargar CSV
# ============================================================
st.subheader("📥 Cargar entregas desde CSV")

uploaded_file = st.file_uploader("Selecciona un archivo CSV", type="csv")

if uploaded_file:
    try:
        data_csv = pd.read_csv(uploaded_file)
        # Limpiar datos de zonas inválidas
        data_csv = data_csv[data_csv['zona'].isin(zonas_validas)]
        data_csv.to_sql("entregas", engine, if_exists="append", index=False)
        st.success("✅ CSV cargado correctamente")
        df = load_data()
    except Exception as e:
        st.error(f"❌ Error al procesar el CSV: {e}")

# ============================================================
# Exportar Excel
# ============================================================
st.subheader("📤 Exportar datos a Excel")
if st.button("Exportar base"):
    df_xlsx = to_excel(df)
    st.download_button(
        label="📥 Descargar Excel",
        data=df_xlsx,
        file_name="entregas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ============================================================
# Visualizaciones y KPIs
# ============================================================
if not df.empty:
    st.subheader("📌 KPIs")
    col1, col2, col3 = st.columns(3)
    col1.metric("Promedio de Entrega (min)", round(df["tiempo_entrega"].mean(), 2))
    col2.metric("Retraso Promedio (min)", round(df["retraso"].mean(), 2))
    col3.metric("Total de Entregas", len(df))

    st.subheader("📍 Distribución de Entregas por Zona")
    st.plotly_chart(px.histogram(df, x="zona", color="zona", title="Número de Entregas por Zona"))

    st.subheader("🚦 Impacto del Tráfico en Tiempo de Entrega")
    st.plotly_chart(px.box(df, x="trafico", y="tiempo_entrega", color="trafico"))

    st.subheader("🌦️ Impacto del Clima en Tiempo de Entrega")
    st.plotly_chart(px.box(df, x="clima", y="tiempo_entrega", color="clima"))

    # ============================================================
    # Modelo de predicción
    # ============================================================
    st.subheader("🤖 Predicción de Tiempo de Entrega")

    df_ml = pd.get_dummies(df.drop(columns=["id_entrega", "fecha"]), drop_first=True)
    X = df_ml.drop(columns=["tiempo_entrega"])
    y = df_ml["tiempo_entrega"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = (mean_squared_error(y_test, y_pred))**0.5
    r2 = r2_score(y_test, y_pred)

    st.write("📊 Resultados del Modelo:")
    st.write(f"MAE: {round(mae,2)} | RMSE: {round(rmse,2)} | R²: {round(r2,2)}")

    # ============================================================
    # Estimación de nuevo pedido
    # ============================================================
    st.subheader("🔮 Estimar un nuevo pedido")
    zona_pred = st.selectbox("Zona", df["zona"].unique())
    tipo_pedido_pred = st.selectbox("Tipo de pedido", df["tipo_pedido"].unique())
    clima_pred = st.selectbox("Clima", df["clima"].unique())
    trafico_pred = st.selectbox("Tráfico", df["trafico"].unique())
    retraso_pred = st.slider("Retraso estimado", 0, 30, 5)

    nuevo = pd.DataFrame([[zona_pred, tipo_pedido_pred, clima_pred, trafico_pred, retraso_pred]],
                         columns=["zona","tipo_pedido","clima","trafico","retraso"])
    nuevo_ml = pd.get_dummies(nuevo)
    nuevo_ml = nuevo_ml.reindex(columns=X.columns, fill_value=0)

    prediccion = model.predict(nuevo_ml)[0]
    st.success(f"⏱️ Tiempo estimado de entrega: {round(prediccion,2)} minutos")

# ============================================================
# Mapa con clima y tráfico
# ============================================================
st.subheader("🗺️ Mapa de Entregas con Clima y Tráfico")

if not df.empty:
    m = folium.Map(location=[13.7, -89.2], zoom_start=7)

    # API OpenWeatherMap
    WEATHER_API = "157cfb5a57724258093e18ea5efda645"

    # API OpenRouteService (tráfico)
    ORS_API = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjgzYzM4NmEzMjkzNzRkMjQ4NWQifQ=="

    # Coordenadas aproximadas de las zonas
    coords = {
        "San Salvador": [13.6929, -89.2182],
        "San Miguel": [13.4833, -88.1833],
        "Santa Ana": [13.9947, -89.5596],
        "La Libertad": [13.4667, -89.3036]
    }

    for _, row in df.iterrows():
        zona = row["zona"]
        lat, lon = coords[zona]

        # Clima
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API}&units=metric"
            data = requests.get(url).json()
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
        except:
            temp, desc = "N/A", "N/A"

        # Tráfico estimado (OpenRouteService)
        try:
            headers = {"Authorization": ORS_API, "Content-Type": "application/json"}
            body = {
                "coordinates": [[lon, lat], [lon+0.01, lat+0.01]],
            }
            ors_data = requests.post("https://api.openrouteservice.org/v2/directions/driving-car",
                                     json=body, headers=headers).json()
            duracion_min = ors_data['features'][0]['properties']['segments'][0]['duration']/60
            trafico_text = f"{duracion_min:.0f} min"
        except:
            trafico_text = "N/A"

        folium.Marker(
            [lat, lon],
            popup=f"Zona: {zona}<br>Clima: {desc} {temp}°C<br>Tiempo estimado (tráfico): {trafico_text}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    st_data = st_folium(m, width=700, height=500)
