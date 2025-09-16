import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import plotly.express as px
from datetime import datetime

# ==============================
# CONFIGURACIÓN DE CONEXIÓN
# ==============================
DB_URL = "postgresql://chivofast_db_user:VOVsj9KYQdoI7vBjpdIpTG1jj2Bvj0GS@dpg-d34osnbe5dus739qotu0-a.oregon-postgres.render.com/chivofast_db"
engine = create_engine(DB_URL)

# ==============================
# CREAR TABLA SI NO EXISTE
# ==============================
def crear_tabla():
    query = """
    CREATE TABLE IF NOT EXISTS public.entregas (
        id_entrega SERIAL PRIMARY KEY,
        fecha TIMESTAMP NOT NULL,
        zona VARCHAR(50) NOT NULL,
        tipo_pedido VARCHAR(50) NOT NULL,
        clima VARCHAR(20) NOT NULL,
        trafico VARCHAR(20) NOT NULL,
        tiempo_entrega INT NOT NULL,
        retraso INT NOT NULL
    );
    """
    with engine.begin() as conn:
        conn.execute(text(query))

crear_tabla()

# ==============================
# CARGAR DATOS
# ==============================
@st.cache_data
def load_data():
    return pd.read_sql("SELECT * FROM public.entregas ORDER BY fecha DESC", engine)

# ==============================
# INSERTAR DATOS DE PRUEBA
# ==============================
def insertar_datos_prueba():
    with engine.begin() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM public.entregas"))
        count = result.scalar()
        if count == 0:
            conn.execute(text("""
                INSERT INTO public.entregas (fecha, zona, tipo_pedido, clima, trafico, tiempo_entrega, retraso) VALUES
                ('2025-09-16 10:00', 'San Salvador', 'Normal', 'Soleado', 'Fluido', 45, 0),
                ('2025-09-16 12:30', 'San Miguel', 'Express', 'Lluvioso', 'Pesado', 60, 15),
                ('2025-09-16 14:00', 'Santa Ana', 'Normal', 'Nublado', 'Moderado', 50, 5),
                ('2025-09-16 15:30', 'La Libertad', 'Express', 'Soleado', 'Moderado', 40, 0);
            """))

insertar_datos_prueba()

# ==============================
# DASHBOARD
# ==============================
st.title("📦 Dashboard Logístico - Entregas")

df = load_data()

if df.empty:
    st.warning("⚠️ No hay datos en la base todavía.")
else:
    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Entregas", len(df))
    col2.metric("Tiempo Promedio (min)", round(df["tiempo_entrega"].mean(), 2))
    col3.metric("Retraso Promedio (min)", round(df["retraso"].mean(), 2))

    # Gráficos
    st.subheader("📊 Entregas por Zona")
    fig1 = px.histogram(df, x="zona", color="tipo_pedido", barmode="group")
    st.plotly_chart(fig1)

    st.subheader("⏱️ Retrasos por Clima")
    fig2 = px.box(df, x="clima", y="retraso", color="clima")
    st.plotly_chart(fig2)

# ==============================
# AGREGAR NUEVA ENTREGA
# ==============================
st.subheader("✏️ Agregar Nueva Entrega")

with st.form("nuevo_pedido_form"):
    fecha = st.date_input("Fecha", value=datetime.today())
    hora = st.time_input("Hora", value=datetime.now().time())
    zona = st.selectbox("Zona", ["San Salvador", "San Miguel", "Santa Ana", "La Libertad"])
    tipo_pedido = st.selectbox("Tipo de pedido", ["Normal", "Express"])
    clima = st.selectbox("Clima", ["Soleado", "Nublado", "Lluvioso"])
    trafico = st.selectbox("Tráfico", ["Fluido", "Moderado", "Pesado"])
    tiempo_entrega = st.number_input("Tiempo de entrega (min)", min_value=1, value=30)
    retraso = st.number_input("Retraso (min)", min_value=0, value=0)
    submitted = st.form_submit_button("Agregar Entrega")

    if submitted:
        fecha_hora = datetime.combine(fecha, hora)
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO public.entregas (fecha, zona, tipo_pedido, clima, trafico, tiempo_entrega, retraso)
                    VALUES (:fecha, :zona, :tipo_pedido, :clima, :trafico, :tiempo_entrega, :retraso)
                """),
                {
                    "fecha": fecha_hora,
                    "zona": zona,
                    "tipo_pedido": tipo_pedido,
                    "clima": clima,
                    "trafico": trafico,
                    "tiempo_entrega": tiempo_entrega,
                    "retraso": retraso
                }
            )
        st.success("✅ Entrega agregada correctamente")
        st.cache_data.clear()
        df = load_data()

# ==============================
# ESTIMACIÓN DE NUEVO PEDIDO
# ==============================
st.subheader("🔮 Estimación de Nuevo Pedido")

zona = st.selectbox("Zona", ["San Salvador", "San Miguel", "Santa Ana", "La Libertad"])
tipo_pedido = st.selectbox("Tipo de pedido", ["Normal", "Express"], key="pred_tipo")
clima = st.selectbox("Clima", ["Soleado", "Nublado", "Lluvioso"], key="pred_clima")
trafico = st.selectbox("Tráfico", ["Fluido", "Moderado", "Pesado"], key="pred_trafico")

if st.button("Estimar tiempo de entrega"):
    if df.empty:
        st.warning("⚠️ No hay datos para calcular la estimación.")
    else:
        filtrado = df[
            (df["zona"] == zona) &
            (df["tipo_pedido"] == tipo_pedido) &
            (df["clima"] == clima) &
            (df["trafico"] == trafico)
        ]
        if not filtrado.empty:
            tiempo_estimado = round(filtrado["tiempo_entrega"].mean(), 2)
            st.success(f"✅ Tiempo estimado de entrega: **{tiempo_estimado} min**")
        else:
            tiempo_promedio = round(df["tiempo_entrega"].mean(), 2)
            st.info(f"ℹ️ No hay datos exactos para esa combinación. Estimado general: **{tiempo_promedio} min**")
