import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine, text
from io import BytesIO
from datetime import datetime
import folium
from streamlit_folium import st_folium

# =========================
# CONFIGURACIÓN DE LA BASE DE DATOS
# =========================
DB_URL = "postgresql://chivofast_db_user:VOVsj9KYQdoI7vBjpdIpTG1jj2Bvj0GS@dpg-d34osnbe5dus739qotu0-a.oregon-postgres.render.com:5432/chivofast_db"
engine = create_engine(DB_URL)

st.set_page_config(page_title="Dashboard Logístico", layout="wide")
st.title("📦 Dashboard Logístico - El Salvador")

# =========================
# FUNCIONES
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_sql("SELECT * FROM public.entregas ORDER BY id_entrega ASC", engine)
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {e}")
        return pd.DataFrame()

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="Entregas")
    writer.close()
    return output.getvalue()

# =========================
# CARGAR DATOS
# =========================
df = load_data()

# =========================
# SUBIR ARCHIVO (CSV/Excel)
# =========================
st.subheader("📂 Subir entregas desde archivo")
uploaded_file = st.file_uploader("Sube un archivo Excel o CSV", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)

        st.write("✅ Vista previa de datos a insertar:")
        st.dataframe(df_upload.head())

        if st.button("🚀 Insertar en la base de datos"):
            if "id_entrega" in df_upload.columns:
                df_upload = df_upload.drop(columns=["id_entrega"])
            df_upload.to_sql("entregas", engine, schema="public", if_exists="append", index=False)
            st.success("✅ Datos insertados correctamente")
            st.cache_data.clear()
            df = load_data()
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")

# =========================
# FORMULARIO PARA NUEVA ENTREGA
# =========================
st.subheader("📝 Agregar nueva entrega manualmente")

with st.form("form_entrega", clear_on_submit=True):
    fecha = st.date_input("Fecha", datetime.today())
    zona = st.selectbox("Zona", ["San Salvador", "San Miguel", "Santa Ana", "La Libertad"])
    tipo_pedido = st.selectbox("Tipo de pedido", ["Supermercado", "Restaurante", "Farmacia", "Tienda en línea"])
    clima = st.selectbox("Clima", ["Soleado", "Nublado", "Lluvioso"])
    trafico = st.selectbox("Tráfico", ["Bajo", "Medio", "Alto"])
    tiempo_entrega = st.number_input("Tiempo de entrega (minutos)", min_value=1, max_value=180, step=1)
    retraso = st.number_input("Retraso (minutos)", min_value=0, max_value=120, step=1)

    submitted = st.form_submit_button("➕ Guardar entrega")

    if submitted:
        try:
            query = text("""
                INSERT INTO public.entregas (fecha, zona, tipo_pedido, clima, trafico, tiempo_entrega, retraso)
                VALUES (:fecha, :zona, :tipo_pedido, :clima, :trafico, :tiempo_entrega, :retraso)
            """)
            with engine.begin() as conn:
                conn.execute(query, {
                    "fecha": fecha,
                    "zona": zona,
                    "tipo_pedido": tipo_pedido,
                    "clima": clima,
                    "trafico": trafico,
                    "tiempo_entrega": tiempo_entrega,
                    "retraso": retraso
                })
            st.success("✅ Entrega agregada correctamente")
            st.cache_data.clear()
            df = load_data()
        except Exception as e:
            st.error(f"❌ Error al guardar: {e}")

# =========================
# MOSTRAR DATOS
# =========================
st.subheader("📊 Datos actuales de entregas")
if df.empty:
    st.warning("⚠️ No hay datos en la tabla de entregas.")
else:
    st.dataframe(df, use_container_width=True)

    # =========================
    # KPIs
    # =========================
    st.subheader("📌 Indicadores Clave (KPIs)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Promedio de Entrega (min)", round(df["tiempo_entrega"].mean(), 2))
    col2.metric("Retraso Promedio (min)", round(df["retraso"].mean(), 2))
    col3.metric("Total de Entregas", len(df))

    # =========================
    # Histogramas y Boxplots
    # =========================
    st.subheader("📍 Distribución de Entregas por Zona")
    st.plotly_chart(px.histogram(df, x="zona", color="zona", title="Número de Entregas por Zona"))

    st.subheader("🚦 Impacto del Tráfico en Tiempo de Entrega")
    st.plotly_chart(px.box(df, x="trafico", y="tiempo_entrega", color="trafico"))

    st.subheader("🌦️ Impacto del Clima en Tiempo de Entrega")
    st.plotly_chart(px.box(df, x="clima", y="tiempo_entrega", color="clima"))

    # =========================
    # PREDICCIÓN MACHINE LEARNING
    # =========================
    st.subheader("🤖 Predicción de Tiempo de Entrega")

    df_ml = pd.get_dummies(df.drop(columns=["id_entrega", "fecha"]), drop_first=True)

    # Mantener solo las columnas de zonas válidas
    zonas_validas = ["zona_San Miguel", "zona_Santa Ana", "zona_La Libertad"]
    for col in df_ml.columns:
        if col.startswith("zona_") and col not in zonas_validas:
            df_ml.drop(columns=[col], inplace=True)

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

    # =========================
    # Estimar nueva entrega
    # =========================
    st.subheader("🔮 Estimar un nuevo pedido")
    zona_new = st.selectbox("Zona", ["San Salvador", "San Miguel", "Santa Ana", "La Libertad"], key="zona_pred")
    tipo_pedido_new = st.selectbox("Tipo de pedido", df["tipo_pedido"].unique(), key="tipo_pred")
    clima_new = st.selectbox("Clima", df["clima"].unique(), key="clima_pred")
    trafico_new = st.selectbox("Tráfico", df["trafico"].unique(), key="trafico_pred")
    retraso_new = st.slider("Retraso estimado", 0, 30, 5)

    nuevo = pd.DataFrame([[zona_new, tipo_pedido_new, clima_new, trafico_new, retraso_new]],
                         columns=["zona","tipo_pedido","clima","trafico","retraso"])
    nuevo_ml = pd.get_dummies(nuevo)
    nuevo_ml = nuevo_ml.reindex(columns=X.columns, fill_value=0)
    prediccion = model.predict(nuevo_ml)[0]
    st.success(f"⏱️ Tiempo estimado de entrega: {round(prediccion,2)} minutos")

    # =========================
    # EXPORTAR A EXCEL
    # =========================
    st.download_button(
        label="📥 Exportar entregas a Excel",
        data=to_excel(df),
        file_name="entregas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =========================
# MAPA EL SALVADOR
# =========================
st.subheader("🌍 Mapa de zonas - Clima y tráfico (simulado)")
zonas_coords = {
    "San Salvador": [13.6929, -89.2182],
    "San Miguel": [13.4833, -88.1833],
    "Santa Ana": [13.9940, -89.5590],
    "La Libertad": [13.4849, -89.3007]
}

m = folium.Map(location=[13.7, -89.2], zoom_start=7)
for index, row in df.iterrows():
    lat, lon = zonas_coords.get(row["zona"], [13.7, -89.2])
    folium.Marker(
        [lat, lon],
        popup=f"Zona: {row['zona']}<br>Clima: {row['clima']}<br>Tráfico: {row['trafico']}<br>Tiempo: {row['tiempo_entrega']} min",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)
st_folium(m, width=700, height=500)
