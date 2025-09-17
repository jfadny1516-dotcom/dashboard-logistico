import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine, text
import os
import requests
from email.message import EmailMessage
import smtplib
import folium
from streamlit_folium import st_folium
from io import BytesIO

# ============================================================
# 🌐 Configuración y conexión a base de datos
# ============================================================
st.header("📦 Dashboard Predictivo de Entregas - ChivoFast")
st.markdown("Análisis y predicción de tiempos de entrega usando Inteligencia Artificial")

DB_URL = st.secrets.get("DATABASE_URL")
ORS_API = st.secrets.get("ORS_API")  # OpenRouteService
OPENWEATHER_API = st.secrets.get("OPENWEATHER_API")  # OpenWeatherMap

if not DB_URL:
    st.error("❌ No se encontró DATABASE_URL en los Secrets de Streamlit")
else:
    db_for_sqlalchemy = DB_URL
    if db_for_sqlalchemy.startswith("postgres://"):
        db_for_sqlalchemy = db_for_sqlalchemy.replace("postgres://", "postgresql+psycopg2://", 1)
    elif db_for_sqlalchemy.startswith("postgresql://"):
        db_for_sqlalchemy = db_for_sqlalchemy.replace("postgresql://", "postgresql+psycopg2://", 1)

    try:
        engine = create_engine(db_for_sqlalchemy, connect_args={"sslmode": "require"})
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS entregas (
                    id_entrega SERIAL PRIMARY KEY,
                    fecha TIMESTAMP NOT NULL,
                    zona VARCHAR(50) NOT NULL,
                    tipo_pedido VARCHAR(50) NOT NULL,
                    clima VARCHAR(20) NOT NULL,
                    trafico VARCHAR(20) NOT NULL,
                    tiempo_entrega INT NOT NULL,
                    retraso INT NOT NULL
                )
            """))
            st.success("✅ Conexión establecida y tabla 'entregas' creada si no existía")
    except Exception as e:
        st.error("❌ Error al conectar a la base de datos:")
        st.text(str(e))

# ============================================================
# 📥 Función para cargar datos
# ============================================================
@st.cache_data
def load_data():
    if not DB_URL:
        return pd.DataFrame()
    df = pd.read_sql("SELECT * FROM entregas", engine)
    return df

df = load_data()

# ============================================================
# 📊 KPIs y gráficos
# ============================================================
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
    df_ml = pd.get_dummies(df.drop(columns=["id_entrega", "fecha"]), drop_first=True)
    X = df_ml.drop(columns=["tiempo_entrega"])
    y = df_ml["tiempo_entrega"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    st.write("📊 Resultados del Modelo:")
    st.write(f"MAE: {round(mae,2)} | RMSE: {round(rmse,2)} | R²: {round(r2,2)}")

    st.subheader("🔮 Estimar un nuevo pedido")
    zona = st.selectbox("Zona", ["San Salvador", "San Miguel", "Santa Ana", "La Libertad"])
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
# 💾 Exportar a Excel
# ============================================================
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="Entregas")
    writer.save()
    processed_data = output.getvalue()
    return processed_data

st.subheader("💾 Exportar datos a Excel")
if st.button("Exportar Excel"):
    excel_data = to_excel(df)
    st.download_button(label="Descargar archivo Excel", data=excel_data, file_name="entregas.xlsx")

# ============================================================
# 🗺️ Mapa de El Salvador con clima y tráfico
# ============================================================
st.subheader("🗺️ Mapa de El Salvador")
m = folium.Map(location=[13.7, -89.2], zoom_start=7)
for idx, row in df.iterrows():
    folium.Marker([row.get("lat",13.7), row.get("lon",-89.2)],
                  popup=f"{row['zona']} - {row['tipo_pedido']}").add_to(m)
st_folium(m, width=700, height=500)

# ============================================================
# 🛣️ Optimización de rutas y envío por correo (ORS)
# ============================================================
st.subheader("🛣️ Optimización de Rutas para Proveedores")

uploaded_provider_file = st.file_uploader("Cargar rutas de proveedor (CSV)", type="csv")
if uploaded_provider_file and ORS_API:
    df_rutas = pd.read_csv(uploaded_provider_file)
    if "direccion" not in df_rutas.columns:
        st.error("❌ La columna 'direccion' no existe en el CSV")
    else:
        def geocode_address(address):
            url = "https://api.openrouteservice.org/geocode/search"
            params = {"api_key": ORS_API, "text": address, "size": 1}
            response = requests.get(url, params=params)
            data = response.json()
            if data["features"]:
                coords = data["features"][0]["geometry"]["coordinates"]
                return coords[1], coords[0]
            return None, None

        df_rutas["lat"] = df_rutas["direccion"].apply(lambda x: geocode_address(x)[0])
        df_rutas["lon"] = df_rutas["direccion"].apply(lambda x: geocode_address(x)[1])

        coords_list = df_rutas[["lon","lat"]].values.tolist()

        def optimize_route(coords):
            url = "https://api.openrouteservice.org/optimization"
            headers = {"Authorization": ORS_API, "Content-Type": "application/json"}
            jobs = [{"id": i+1, "location": c} for i, c in enumerate(coords)]
            vehicles = [{"id": 1, "start": coords[0], "end": coords[0]}]
            body = {"jobs": jobs, "vehicles": vehicles}
            response = requests.post(url, headers=headers, json=body)
            return response.json()

        result = optimize_route(coords_list)

        if "routes" in result and result["routes"]:
            sequence = result["routes"][0]["steps"]
            df_rutas = df_rutas.iloc[[step["job"]-1 for step in sequence]]
            st.success("✅ Ruta optimizada calculada")

            base_url = "https://www.google.com/maps/dir/"
            rutas = "/".join(df_rutas["direccion"].str.replace(" ", "+"))
            link_maps = base_url + rutas
            st.markdown(f"[Abrir ruta optimizada en Google Maps]({link_maps})")

            correo_proveedor = st.text_input("Correo del proveedor")
            if st.button("✉️ Enviar ruta al proveedor"):
                if correo_proveedor:
                    msg = EmailMessage()
                    msg.set_content(f"Tu ruta optimizada está aquí: {link_maps}")
                    msg['Subject'] = "Ruta de Entregas Optimizada"
                    msg['From'] = "tu_correo@empresa.com"
                    msg['To'] = correo_proveedor
                    try:
                        with smtplib.SMTP("smtp.gmail.com", 587) as server:
                            server.starttls()
                            server.login("tu_correo@empresa.com", "TU_PASSWORD_APP")
                            server.send_message(msg)
                        st.success("✅ Correo enviado correctamente")
                    except Exception as e:
                        st.error(f"❌ Error al enviar correo: {e}")
                else:
                    st.warning("⚠️ Ingrese un correo válido")
        else:
            st.error("❌ No se pudo optimizar la ruta con ORS")
