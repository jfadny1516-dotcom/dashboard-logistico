import streamlit as st
import pandas as pd
from io import BytesIO
from sqlalchemy import create_engine, text
import folium
from streamlit_folium import st_folium
from datetime import datetime

# =========================
# CONFIGURACIÓN DE LA BASE DE DATOS
# =========================
DB_URL = "postgresql://chivofast_db_user:VOVsj9KYQdoI7vBjpdIpTG1jj2Bvj0GS@dpg-d34osnbe5dus739qotu0-a.oregon-postgres.render.com:5432/chivofast_db"
engine = create_engine(DB_URL)

st.set_page_config(page_title="Dashboard Logístico", layout="wide")
st.title("📦 Dashboard Logístico - El Salvador")

# =========================
# CARGAR DATOS
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_sql("SELECT * FROM public.entregas ORDER BY id_entrega ASC", engine)
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar datos: {e}")
        return pd.DataFrame()

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
                df_upload = df_upload.drop(columns=["id_entrega"])  # evitar duplicados en PK

            df_upload.to_sql("entregas", engine, schema="public", if_exists="append", index=False)
            st.success("✅ Datos insertados correctamente en la base de PostgreSQL")
            st.cache_data.clear()
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
        except Exception as e:
            st.error(f"❌ Error al guardar: {e}")

# =========================
# MOSTRAR DATOS
# =========================
st.subheader("📊 Datos actuales de entregas")
df = load_data()

if df.empty:
    st.warning("⚠️ No hay datos en la tabla de entregas.")
else:
    st.dataframe(df, use_container_width=True)

# =========================
# EXPORTAR A EXCEL
# =========================
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="Entregas")
    writer.close()
    return output.getvalue()

if not df.empty:
    st.download_button(
        label="📥 Exportar entregas a Excel",
        data=to_excel(df),
        file_name="entregas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =========================
# MAPA EL SALVADOR (Simulación de clima y tráfico)
# =========================
st.subheader("🌍 Mapa de zonas - Clima y tráfico (simulado)")

zonas = {
    "San Salvador": [13.6929, -89.2182],
    "San Miguel": [13.4833, -88.1833],
    "Santa Ana": [13.9942, -89.5597],
    "La Libertad": [13.4885, -89.3220]
}

climas = {
    "San Salvador": "Soleado, 30°C ☀️",
    "San Miguel": "Lluvioso, 25°C 🌧️",
    "Santa Ana": "Nublado, 26°C ☁️",
    "La Libertad": "Soleado, 28°C 🌞"
}

trafico = {
    "San Salvador": "Alto 🚦",
    "San Miguel": "Medio 🟡",
    "Santa Ana": "Bajo 🟢",
    "La Libertad": "Medio 🟡"
}

m = folium.Map(location=[13.6929, -89.2182], zoom_start=8)

for zona, coords in zonas.items():
    info = f"""
    <b>{zona}</b><br>
    Clima: {climas[zona]}<br>
    Tráfico: {trafico[zona]}
    """
    folium.Marker(
        location=coords,
        popup=info,
        tooltip=zona,
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

st_folium(m, width=700, height=500)
