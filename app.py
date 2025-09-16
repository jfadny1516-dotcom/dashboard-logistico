import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine, text
import os
from datetime import datetime

st.set_page_config(page_title="Dashboard ChivoFast", layout="wide")

# ============================================================
# 🔗 Conexión a PostgreSQL (Render)
# ============================================================
DATABASE_URL = os.getenv("DATABASE_URL")

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
            st.success(f"✅ Conexión a PostgreSQL establecida (SELECT 1 = {test})")
    except Exception as e:
        st.error("❌ Error al conectar a la base de datos:")
        st.text(str(e))

# ============================================================
# 📥 Función para cargar datos y crear tabla si no existe
# ============================================================
@st.cache_data
def load_data():
    try:
        engine_local = create_engine(db_for_sqlalchemy, connect_args={"sslmode": "require"})
        with engine_local.connect() as conn:
            # Crear tabla si no existe
            conn.execute(text("""
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
            """))
            # Insertar datos de prueba si la tabla está vacía
            result = conn.execute(text("SELECT COUNT(*) FROM public.entregas"))
            count = result.scalar()
            if count == 0:
                conn.execute(text("""
                    INSERT INTO public.entregas (fecha, zona, tipo_pedido, clima, trafico, tiempo_entrega, retraso)
                    VALUES
                    ('2025-09-16 10:00', 'Zona 1', 'Normal', 'Soleado', 'Fluido', 45, 0),
                    ('2025-09-16 12:30', 'Zona 2', 'Express', 'Lluvioso', 'Pesado', 60, 15),
                    ('2025-09-16 14:00', 'Zona 3', 'Normal', 'Nublado', 'Moderado', 50, 5);
                """))
            # Leer datos
            df = pd.read_sql("SELECT * FROM public.entregas ORDER BY id_entrega", conn)
        return df
    except Exception as e:
        st.error("❌ Error cargando datos de PostgreSQL")
        st.text(str(e))
        return pd.DataFrame()

df = load_data()

# ============================================================
# 📊 Dashboard con KPIs y gráficas
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

    # ============================================================
    # 🤖 Modelo de predicción
    # ============================================================
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
    # ✏️ Panel para agregar nuevas entregas
    # ============================================================
    st.subheader("✏️ Agregar nueva entrega")
    with st.form("nuevo_pedido_form"):
        fecha = st.date_input("Fecha", value=pd.to_datetime("today"))
        zona = st.selectbox("Zona", df["zona"].unique())
        tipo_pedido = st.selectbox("Tipo de pedido", df["tipo_pedido"].unique())
        clima = st.selectbox("Clima", df["clima"].unique())
        trafico = st.selectbox("Tráfico", df["trafico"].unique())
        tiempo_entrega = st.number_input("Tiempo de entrega (min)", min_value=1, value=30)
        retraso = st.number_input("Retraso (min)", min_value=0, value=0)
        submitted = st.form_submit_button("Agregar")

        if submitted:
            try:
                with engine.connect() as conn:
                    conn.execute(
                        text("""
                            INSERT INTO public.entregas (fecha, zona, tipo_pedido, clima, trafico, tiempo_entrega, retraso)
                            VALUES (:fecha, :zona, :tipo_pedido, :clima, :trafico, :tiempo_entrega, :retraso)
                        """),
                        {
                            "fecha": fecha,
                            "zona": zona,
                            "tipo_pedido": tipo_pedido,
                            "clima": clima,
                            "trafico": trafico,
                            "tiempo_entrega": tiempo_entrega,
                            "retraso": retraso
                        }
                    )
                st.success("✅ Entrega agregada correctamente")
                df = load_data()  # recarga datos
            except Exception as e:
                st.error("❌ Error agregando la entrega")
                st.text(str(e))
else:
    st.warning("⚠️ No se pudieron cargar datos desde la base de datos PostgreSQL.")
