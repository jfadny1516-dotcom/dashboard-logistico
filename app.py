# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine, text
import os
import io
import requests
import folium
from streamlit_folium import st_folium

# =======================
# 🔑 API Keys
# =======================
OPENWEATHER_API_KEY = "157cfb5a57724258093e18ea5efda645"
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjgzYzM4NmEzMjkzNzRkMjQ4NWQzZGIzODNlY2Q3YmJlIiwiaCI6Im11cm11cjY0In0="

# =======================
# 📦 Conexión a la base de datos PostgreSQL
# =======================
DATABASE_URL = os.getenv("DATABASE_URL")  # Configurar en Streamlit Secrets

if not DATABASE_URL:
    st.error("❌ No se encontró DATABASE_URL en los Secrets de Streamlit")
else:
    db_for_sqlalchemy = DATABASE_URL
    if db_for_sqlalchemy.startswith("postgres://"):
        db_for_sqlalchemy = db_for_sqlalchemy.replace("postgres://", "postgresql+psycopg2://", 1)
    elif db_for_sqlalchemy.startswith("postgresql://"):
        db_for_sqlalchemy = db_for_sqlalchemy.replace("postgresql://", "postgresql+psycopg2://", 1)

    try:
        engine = create_engine(db_for_sqlalchemy, connect_args={"sslmode": "require"})
        with engine.connect() as conn:
            test = conn.execute(text("SELECT 1")).scalar()
            st.success(f"✅ Conexión a PostgreSQL establecida (prueba SELECT 1 = {test})")
    except Exception as e:
        st.error("❌ Error al conectar a la base de datos:")
        st.text(str(e))

# =======================
# 📥 Cargar datos desde PostgreSQL
# =======================
@st.cache_data
def load_data():
    if not DATABASE_URL:
        return pd.DataFrame()
    df = pd.read_sql("SELECT * FROM entregas WHERE zona IN ('San Salvador','San Miguel','Santa Ana','La Libertad')", engine)
    return df

df = load_data()

# =======================
# 📝 Interfaz de Streamlit
# =======================
st.title("📦 Dashboard Predictivo de Entregas - ChivoFast")

# ---- Agregar nuevos datos manualmente ----
st.subheader("➕ Agregar nueva entrega")
with st.form("add_delivery"):
    zona = st.selectbox("Zona", ["San Salvador","San Miguel","Santa Ana","La Libertad"])
    tipo_pedido = st.selectbox("Tipo de pedido", ["Supermercado", "Restaurante", "Tienda en línea", "Farmacia"])
    clima = st.selectbox("Clima", ["Soleado","Nublado","Lluvioso"])
    trafico = st.selectbox("Tráfico", ["Bajo","Medio","Alto"])
    tiempo_entrega = st.number_input("Tiempo de entrega (min)", min_value=1, max_value=500, value=30)
    retraso = st.number_input("Retraso (min)", min_value=0, max_value=100, value=0)
    submitted = st.form_submit_button("Agregar")
    if submitted:
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("INSERT INTO entregas (fecha, zona, tipo_pedido, clima, trafico, tiempo_entrega, retraso) VALUES (NOW(), :zona, :tipo_pedido, :clima, :trafico, :tiempo_entrega, :retraso)"),
                    {"zona": zona, "tipo_pedido": tipo_pedido, "clima": clima, "trafico": trafico, "tiempo_entrega": tiempo_entrega, "retraso": retraso}
                )
            st.success("✅ Entrega agregada correctamente")
            df = load_data()  # recargar
        except Exception as e:
            st.error(f"❌ Error al agregar entrega: {e}")

# ---- Cargar desde CSV ----
st.subheader("📁 Cargar entregas desde CSV")
uploaded_file = st.file_uploader("Selecciona un archivo CSV", type="csv")
if uploaded_file:
    try:
        csv_data = pd.read_csv(uploaded_file)
        # Eliminar columnas id_entrega si existe
        if 'id_entrega' in csv_data.columns:
            csv_data = csv_data.drop(columns=['id_entrega'])
        csv_data.to_sql("entregas", engine, if_exists="append", index=False)
        st.success("✅ CSV cargado correctamente")
        df = load_data()
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")

# ---- Exportar a Excel ----
st.subheader("📤 Exportar base a Excel")
def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="Entregas")
    writer.save()
    processed_data = output.getvalue()
    return processed_data

if not df.empty:
    excel_data = to_excel(df)
    st.download_button(label="💾 Descargar Excel", data=excel_data, file_name="entregas.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =======================
# 📊 KPIs y gráficos
# =======================
if not df.empty:
    st.subheader("📌 Indicadores Clave (KPIs)")
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

    st.subheader("🤖 Predicción de Tiempo de Entrega")
    df_ml = pd.get_dummies(df.drop(columns=["id_entrega","fecha"]), drop_first=True)
    X = df_ml.drop(columns=["tiempo_entrega"])
    y = df_ml["tiempo_entrega"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = (mean_squared_error(y_test, y_pred))**0.5
    r2 = r2_score(y_test, y_pred)
    st.write(f"MAE: {round(mae,2)} | RMSE: {round(rmse,2)} | R²: {round(r2,2)}")

    st.subheader("🔮 Estimar un nuevo pedido")
    zona_n = st.selectbox("Zona", df["zona"].unique())
    tipo_pedido_n = st.selectbox("Tipo de pedido", df["tipo_pedido"].unique())
    clima_n = st.selectbox("Clima", df["clima"].unique())
    trafico_n = st.selectbox("Tráfico", df["trafico"].unique())
    retraso_n = st.slider("Retraso estimado", 0, 30, 5)

    nuevo = pd.DataFrame([[zona_n, tipo_pedido_n, clima_n, trafico_n, retraso_n]],
                         columns=["zona","tipo_pedido","clima","trafico","retraso"])
    nuevo_ml = pd.get_dummies(nuevo)
    nuevo_ml = nuevo_ml.reindex(columns=X.columns, fill_value=0)
    prediccion = model.predict(nuevo_ml)[0]
    st.success(f"⏱️ Tiempo estimado de entrega: {round(prediccion,2)} minutos")

# =======================
# 🗺 Mapa con clima y tráfico
# =======================
st.subheader("🌎 Mapa con Clima y Tráfico en Tiempo Real")
m = folium.Map(location=[13.7, -89.2], zoom_start=7)

for zona, coords in {
    "San Salvador": [13.6929, -89.2182],
    "San Miguel": [13.4833, -88.1833],
    "Santa Ana": [13.9940, -89.5590],
    "La Libertad": [13.4849, -89.3007]
}.items():
    lat, lon = coords

    # Clima
    w_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    w_data = requests.get(w_url).json()
    temp = w_data.get("main", {}).get("temp", "N/A")
    desc = w_data.get("weather", [{}])[0].get("description", "N/A")

    # Tráfico (tiempo estimado) usando ORS
    destino = [-89.2182, 13.7]
    body = {"coordinates": [[lon, lat], destino]}
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    try:
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
