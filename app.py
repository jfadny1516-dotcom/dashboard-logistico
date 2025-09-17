import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine, text
import os
import io
import folium
from streamlit_folium import st_folium
import requests

# ============================================================
# Configuración de Streamlit
# ============================================================
st.set_page_config(page_title="Dashboard Predictivo de Entregas", layout="wide")
st.header("📦 Dashboard Predictivo de Entregas - ChivoFast")
st.markdown("Análisis y predicción de tiempos de entrega usando Inteligencia Artificial")

# ============================================================
# Conexión a PostgreSQL
# ============================================================
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    st.error("❌ No se encontró DATABASE_URL en los Secrets de Streamlit")
else:
    db_url = DATABASE_URL
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    try:
        engine = create_engine(db_url, connect_args={"sslmode": "require"})
        with engine.connect() as conn:
            test = conn.execute(text("SELECT 1")).scalar()
            st.success(f"✅ Conexión a PostgreSQL establecida (SELECT 1 = {test})")
    except Exception as e:
        st.error("❌ Error al conectar a la base de datos:")
        st.text(str(e))

# ============================================================
# Función para cargar datos
# ============================================================
@st.cache_data
def load_data():
    if not DATABASE_URL:
        return pd.DataFrame()
    df = pd.read_sql("SELECT * FROM entregas", engine)
    return df

df = load_data()

# ============================================================
# Mostrar datos
# ============================================================
if not df.empty:
    st.subheader("📋 Datos cargados")
    st.dataframe(df)

    # ============================================================
    # KPIs
    # ============================================================
    st.subheader("📌 Indicadores Clave (KPIs)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Promedio de Entrega (min)", round(df["tiempo_entrega"].mean(), 2))
    col2.metric("Retraso Promedio (min)", round(df["retraso"].mean(), 2))
    col3.metric("Total de Entregas", len(df))

    # ============================================================
    # Gráficos
    # ============================================================
    st.subheader("📍 Distribución de Entregas por Zona")
    st.plotly_chart(px.histogram(df, x="zona", color="zona", title="Número de Entregas por Zona"))

    st.subheader("🚦 Impacto del Tráfico en Tiempo de Entrega")
    st.plotly_chart(px.box(df, x="trafico", y="tiempo_entrega", color="trafico"))

    st.subheader("🌦️ Impacto del Clima en Tiempo de Entrega")
    st.plotly_chart(px.box(df, x="clima", y="tiempo_entrega", color="clima"))

    # ============================================================
    # Predicción con RandomForest
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
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    st.write("📊 Resultados del Modelo:")
    st.write(f"MAE: {round(mae,2)} | RMSE: {round(rmse,2)} | R²: {round(r2,2)}")

    # ============================================================
    # Estimación de nuevo pedido
    # ============================================================
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

    # ============================================================
    # Exportar datos a Excel
    # ============================================================
    def to_excel(df):
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine="xlsxwriter")
        df.to_excel(writer, index=False, sheet_name="Entregas")
        writer.save()
        processed_data = output.getvalue()
        return processed_data

    st.download_button(
        label="📥 Descargar datos en Excel",
        data=to_excel(df),
        file_name="entregas.xlsx"
    )

    # ============================================================
    # Mapa de El Salvador con clima y tráfico
    # ============================================================
    st.subheader("🗺️ Mapa de El Salvador (Clima y Tráfico)")
    mapa = folium.Map(location=[13.7, -89.2], zoom_start=7)

    # Usando OpenWeatherMap API
    OWM_API_KEY = "157cfb5a57724258093e18ea5efda645"
    zonas_coords = {
        "San Salvador": [13.6929, -89.2182],
        "San Miguel": [13.4833, -88.1833],
        "Santa Ana": [13.9947, -89.5598],
        "La Libertad": [13.4764, -89.2961]
    }

    for zona_name, coords in zonas_coords.items():
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={coords[0]}&lon={coords[1]}&appid={OWM_API_KEY}&units=metric"
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            temp = data["main"]["temp"]
            desc = data["weather"][0]["description"]
            popup_text = f"{zona_name}\nClima: {desc}\nTemp: {temp}°C"
            folium.Marker(location=coords, popup=popup_text).add_to(mapa)

    st_data = st_folium(mapa, width=700, height=500)

else:
    st.warning("⚠️ No se pudieron cargar datos desde la base de datos PostgreSQL.")
