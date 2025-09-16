import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine
import os

# ============================
# 1. Cargar datos desde PostgreSQL
# ============================
@st.cache_data
def load_data():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        st.error("❌ No se encontró DATABASE_URL en los Secrets de Streamlit")
        return pd.DataFrame()
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM entregas", engine)
    return df

df = load_data()

st.title("📦 Dashboard Predictivo de Entregas - ChivoFast")
st.markdown("Análisis y predicción de tiempos de entrega usando Inteligencia Artificial")

if not df.empty:
    # ============================
    # 2. KPIs
    # ============================
    st.subheader("📌 Indicadores Clave (KPIs)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Promedio de Entrega (min)", round(df["tiempo_entrega"].mean(), 2))
    col2.metric("Retraso Promedio (min)", round(df["retraso"].mean(), 2))
    col3.metric("Total de Entregas", len(df))

    # ============================
    # 3. Visualizaciones
    # ============================
    st.subheader("📍 Distribución de Entregas por Zona")
    fig_zona = px.histogram(df, x="zona", color="zona", title="Número de Entregas por Zona")
    st.plotly_chart(fig_zona)

    st.subheader("🚦 Impacto del Tráfico en Tiempo de Entrega")
    fig_trafico = px.box(df, x="trafico", y="tiempo_entrega", color="trafico")
    st.plotly_chart(fig_trafico)

    st.subheader("🌦️ Impacto del Clima en Tiempo de Entrega")
    fig_clima = px.box(df, x="clima", y="tiempo_entrega", color="clima")
    st.plotly_chart(fig_clima)

    # ============================
    # 4. Modelo Predictivo
    # ============================
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

    # ============================
    # 5. Predicción interactiva
    # ============================
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
else:
    st.warning("⚠️ No se pudieron cargar datos desde la base de datos PostgreSQL.")
