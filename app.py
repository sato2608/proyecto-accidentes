"""
Proyecto Integrador – Analítica de Datos (Entregable 5)
Aplicación Streamlit: víctimas de accidentes de tránsito atendidas en IPS de Bucaramanga (2018–2021)

Ejecutar en local:   streamlit run app.py
Datos por defecto:   data/dataset.csv  (dataset_completo_limpio_transformado.csv del Cuaderno 1)
"""
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, roc_auc_score,
                             average_precision_score, precision_recall_curve, r2_score,
                             mean_absolute_error, mean_squared_error, silhouette_score)

# ----------------------------------------------------------------------------
# Configuración general
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Accidentes de tránsito – Analítica", page_icon="🚑", layout="wide")
sns.set_theme(style="whitegrid")
SEED = 42
RUTA_DEFECTO = "data/dataset.csv"

# Columnas mínimas que debe traer el CSV para que la app funcione
COLUMNAS_REQUERIDAS = ["numero", "edad", "sexo", "tipo_de_vehiculo", "relacion_usuario_accidente",
                       "fecha_de_accidente", "fecha_de_ingreso_ips", "requerimiento_de_cx",
                       "politraumatismo", "tipo_afiliaci_n", "costos", "ips", "ano",
                       "tipo_administradora", "patologia_principal", "region_principal"]

# Predictores (sin fuga de información: nada posterior al ingreso, sin costos ni calendario)
NUM_CLF = ["edad", "dias_hasta_ingreso"]
CAT_PRE = ["sexo", "tipo_de_vehiculo", "relacion_usuario_accidente", "ips", "tipo_administradora", "tipo_afiliaci_n"]
CAT_DX = ["patologia_principal", "region_principal", "politraumatismo"]
NUM_REG = ["edad", "dias_hasta_ingreso", "ano", "mes_sin", "mes_cos", "dia_semana_sin", "dia_semana_cos"]
PCA_VARS = ["edad", "log_costos", "dias_hasta_ingreso", "cx", "politrauma", "es_fractura",
            "sexo_masc", "es_moto", "es_conductor"]

EXPLICACIONES = {
    "Regresión logística": (
        "Estima la probabilidad de cirugía con una combinación lineal de los predictores pasada por la "
        "función logística. Cada coeficiente se traduce en un *odds ratio*. Es interpretable y estable; "
        "fue el modelo recomendado en el informe."),
    "Árbol de decisión": (
        "Divide a los pacientes con preguntas sí/no (por ejemplo, ¿tiene fractura?) eligiendo en cada paso "
        "el corte que más reduce la impureza (Gini o entropía). La profundidad máxima controla el "
        "sobreajuste: un árbol muy profundo memoriza el entrenamiento."),
    "Random Forest": (
        "Entrena muchos árboles sobre muestras bootstrap y subconjuntos aleatorios de variables, y promedia "
        "sus votos. Reduce la varianza de un árbol individual a costa de interpretabilidad."),
    "SVM (máquina de vectores de soporte)": (
        "Busca la frontera que separa las clases con el mayor margen posible; con kernel RBF la frontera "
        "puede ser curva. Es costosa con muchos datos, por eso aquí se entrena con una muestra estratificada."),
}


# ----------------------------------------------------------------------------
# Carga y preparación (con control de errores)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Leyendo el archivo...")
def leer_csv(fuente):
    return pd.read_csv(fuente)


def validar(df):
    """Devuelve la lista de columnas requeridas que faltan."""
    return [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]


@st.cache_data(show_spinner="Preparando los datos...")
def preparar(df):
    """Reglas fijas de limpieza que no aprenden de los datos (pueden aplicarse antes de partir)."""
    d = df.copy()
    d = d.loc[~d.drop(columns=["numero"]).duplicated()].copy()          # duplicados salvo el consecutivo
    d["costos"] = pd.to_numeric(d["costos"], errors="coerce")
    d = d[d["costos"] > 0].copy()
    d["log_costos"] = np.log10(d["costos"])
    d["edad"] = pd.to_numeric(d["edad"], errors="coerce")
    fa = pd.to_datetime(d["fecha_de_accidente"], errors="coerce")
    fi = pd.to_datetime(d["fecha_de_ingreso_ips"], errors="coerce")
    d["dias_hasta_ingreso"] = (fi - fa).dt.days
    d.loc[d["dias_hasta_ingreso"] < 0, "dias_hasta_ingreso"] = np.nan   # fecha de ingreso antes del accidente
    for c in ["mes_sin", "mes_cos", "dia_semana_sin", "dia_semana_cos"]:
        if c not in d.columns:
            d[c] = 0.0
    d["cx"] = (d["requerimiento_de_cx"].astype(str).str.lower() == "si").astype(int)
    d["politrauma"] = (d["politraumatismo"].astype(str).str.lower() == "si").astype(int)
    d["sexo_masc"] = (d["sexo"].astype(str).str.lower() == "masculino").astype(int)
    d["es_moto"] = (d["tipo_de_vehiculo"].astype(str).str.lower() == "moto").astype(int)
    d["es_conductor"] = (d["relacion_usuario_accidente"].astype(str).str.lower() == "conductor").astype(int)
    d["es_fractura"] = (d["patologia_principal"].astype(str).str.lower() == "fractura").astype(int)
    for c in CAT_PRE + CAT_DX + ["requerimiento_de_cx"]:
        d[c] = d[c].astype(str)
    return d.reset_index(drop=True)


def preprocesador(num, cat):
    """Imputación + escalado / one-hot. Se ajusta SOLO con entrenamiento dentro del Pipeline."""
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                                                drop="first", sparse_output=False))]), cat),
    ])


@st.cache_data
def particion(d):
    """Partición 80/20 estratificada por cirugía, ANTES de cualquier preprocesamiento."""
    tr, te = train_test_split(d.index, test_size=0.20, stratify=d["cx"], random_state=SEED)
    return list(tr), list(te)


def fig_a_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Barra lateral: fuente de datos y navegación
# ----------------------------------------------------------------------------
st.sidebar.title("🚑 Analítica de accidentes")
archivo = st.sidebar.file_uploader("Cargar CSV (opcional)", type=["csv"],
                                   help="Si no carga nada, se usa el dataset del proyecto (data/dataset.csv).")

try:
    df_raw = leer_csv(archivo) if archivo is not None else leer_csv(RUTA_DEFECTO)
except FileNotFoundError:
    st.error(f"No se encontró `{RUTA_DEFECTO}`. Cargue el CSV desde la barra lateral.")
    st.stop()
except Exception as e:  # archivo corrupto, separador distinto, etc.
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

faltan = validar(df_raw)
if faltan:
    st.error("El archivo no tiene las columnas que necesita la aplicación: " + ", ".join(faltan) +
             ". Use el CSV generado por el Cuaderno 1 (dataset_completo_limpio_transformado.csv).")
    st.stop()

d = preparar(df_raw)
if len(d) < 200 or d["cx"].nunique() < 2:
    st.error("El archivo tiene muy pocas filas válidas o una sola clase de cirugía; no se pueden entrenar modelos.")
    st.stop()

idx_tr, idx_te = particion(d)
train, test = d.loc[idx_tr], d.loc[idx_te]
CLAVE_DATOS = f"{len(d)}-{int(d['numero'].astype(str).str.len().sum())}-{d['costos'].sum():.0f}"

pagina = st.sidebar.radio("Módulo", ["Inicio", "Carga y análisis de datos", "Segmentación (PCA y K-means)",
                                      "Regresión lineal del costo", "Modelos de clasificación", "Predicciones y descarga"])
st.sidebar.caption(f"Filas válidas: {len(d):,} · Entrenamiento: {len(train):,} · Prueba: {len(test):,}")
st.sidebar.caption(f"Semilla: {SEED}")


# ----------------------------------------------------------------------------
# Entrenamiento de modelos de clasificación (en caché por combinación de parámetros)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Entrenando el modelo...")
def entrenar_clf(nombre, con_dx, balanceado, profundidad, criterio, n_arboles, C_svm, _train, clave_datos):
    """`_train` no se usa para el caché (empieza con _); `clave_datos` identifica el dataset cargado."""
    num, cat = NUM_CLF, CAT_PRE + (CAT_DX if con_dx else [])
    cw = "balanced" if balanceado else None
    if nombre == "Regresión logística":
        clf = LogisticRegression(max_iter=3000, class_weight=cw)
    elif nombre == "Árbol de decisión":
        clf = DecisionTreeClassifier(max_depth=profundidad, criterion=criterio, min_samples_leaf=50,
                                     class_weight=cw, random_state=SEED)
    elif nombre == "Random Forest":
        clf = RandomForestClassifier(n_estimators=n_arboles, min_samples_leaf=10, n_jobs=-1,
                                     class_weight="balanced_subsample" if balanceado else None, random_state=SEED)
    else:
        # Probabilidades de la SVM calibradas con el método sigmoide (Platt)
        clf = CalibratedClassifierCV(SVC(kernel="rbf", C=C_svm, gamma="scale", class_weight=cw),
                                     method="sigmoid", cv=3, ensemble=False)
    datos = _train
    if nombre.startswith("SVM") and len(datos) > 6000:     # muestra estratificada para que sea rápido
        datos, _ = train_test_split(datos, train_size=6000, stratify=datos["cx"], random_state=SEED)

    # 1) Validación interna (75/25 dentro de ENTRENAMIENTO) para elegir el umbral sin tocar la prueba
    sub, val = train_test_split(datos, test_size=0.25, stratify=datos["cx"], random_state=SEED)
    m_val = Pipeline([("prep", preprocesador(num, cat)), ("clf", clone(clf))]).fit(sub, sub["cx"])
    prob_val = m_val.predict_proba(val)[:, 1]
    # 2) Modelo final con todo el entrenamiento
    modelo = Pipeline([("prep", preprocesador(num, cat)), ("clf", clf)]).fit(datos, datos["cx"])
    return modelo, val["cx"].values, prob_val


def umbral_optimo(y_val, prob_val, razon):
    """Umbral que minimiza  1·FP + razon·FN  en la validación interna."""
    grid = np.linspace(0.01, 0.95, 189)
    costos = [np.sum((prob_val >= t) & (y_val == 0)) + razon * np.sum((prob_val < t) & (y_val == 1)) for t in grid]
    return float(grid[int(np.argmin(costos))])


# ============================================================================
# PÁGINAS
# ============================================================================
if pagina == "Inicio":
    st.title("Víctimas de accidentes de tránsito atendidas en IPS de Bucaramanga (2018–2021)")
    st.markdown("""
**Decisión que apoya la aplicación:** priorizar a los pacientes con mayor probabilidad de requerir cirugía
y estimar el costo de su atención.

* **Fuente:** API de Datos Abiertos Colombia, conjunto `g4vd-w4ip` (datos.gov.co), limpiado en el Cuaderno 1.
* **Respuesta binaria:** `requerimiento_de_cx` (1 = requirió cirugía).
* **Respuesta continua:** `costos` en pesos colombianos (modelado como log10).
* **Regla sin fuga:** la partición entrenamiento/prueba (80/20 estratificada) se hace antes de imputar,
  escalar o codificar; todo eso se ajusta solo con entrenamiento dentro de un `Pipeline`.

Use el menú de la izquierda para navegar por los módulos.
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Atenciones", f"{len(d):,}")
    c2.metric("% con cirugía", f"{d['cx'].mean()*100:.1f} %")
    c3.metric("Costo mediano", f"${d['costos'].median():,.0f}".replace(",", "."))
    c4.metric("IPS", d["ips"].nunique())

# ----------------------------------------------------------------------------
elif pagina == "Carga y análisis de datos":
    st.title("Carga y análisis descriptivo")
    st.caption("Datos tal como se cargaron (antes de las reglas de limpieza de la app).")

    st.subheader("Primeras 10 filas")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("Estadísticas descriptivas")
    st.dataframe(df_raw.describe().T, use_container_width=True)

    st.subheader("Tipos de dato y valores faltantes (equivalente a `df.info()`)")
    info = pd.DataFrame({"Tipo": df_raw.dtypes.astype(str), "No nulos": df_raw.notna().sum(),
                         "Nulos": df_raw.isna().sum(), "% nulos": (df_raw.isna().mean() * 100).round(1)})
    st.dataframe(info, use_container_width=True)
    with st.expander("Ver salida textual de df.info()"):
        buf = io.StringIO()
        df_raw.info(buf=buf)
        st.text(buf.getvalue())

    st.subheader("Visualizaciones")
    numericas = df_raw.select_dtypes(include="number").columns.tolist()
    categoricas = [c for c in df_raw.columns if c not in numericas and 1 < df_raw[c].nunique() <= 30]

    col1, col2 = st.columns(2)
    with col1:
        var_num = st.selectbox("Histograma de una variable numérica", numericas,
                               index=numericas.index("log_costos") if "log_costos" in numericas else 0)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_raw[var_num].dropna(), bins=50, ax=ax, color="#4C72B0")
        ax.set_title(f"Distribución de {var_num}")
        st.pyplot(fig)
    with col2:
        var_cat = st.selectbox("Gráfico de barras de una variable categórica", categoricas,
                               index=categoricas.index("tipo_de_vehiculo") if "tipo_de_vehiculo" in categoricas else 0)
        vc = df_raw[var_cat].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, color="#55A868")
        ax.set_xlabel("registros"); ax.set_title(f"Frecuencia de {var_cat}")
        st.pyplot(fig)

    st.markdown("**Matriz de correlación (Spearman)**")
    vars_corr = ["edad", "log_costos", "dias_hasta_ingreso", "cx", "politrauma", "sexo_masc",
                 "es_moto", "es_conductor", "es_fractura", "ano"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(d[vars_corr].apply(pd.to_numeric, errors="coerce").corr(method="spearman"),
                annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    st.pyplot(fig)

    st.markdown("**Relación con la cirugía**")
    var_rel = st.selectbox("Porcentaje de cirugía según:", ["patologia_principal", "ips", "relacion_usuario_accidente",
                                                            "tipo_de_vehiculo", "curso_de_vida"]
                           if "curso_de_vida" in d.columns else ["patologia_principal", "ips"])
    t = (d.groupby(var_rel)["cx"].mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    t.plot.barh(ax=ax, color="#C44E52"); ax.set_xlabel("% que requirió cirugía")
    st.pyplot(fig)

# ----------------------------------------------------------------------------
elif pagina == "Segmentación (PCA y K-means)":
    st.title("Segmentación de pacientes: PCA y K-means")
    st.markdown("Variables estandarizadas con media y desviación **de entrenamiento**: " + ", ".join(PCA_VARS) + ".")

    prep = Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())])
    Ztr = prep.fit_transform(train[PCA_VARS])
    pca = PCA(random_state=SEED).fit(Ztr)
    var = pca.explained_variance_ratio_
    acum = np.cumsum(var)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(1, len(var) + 1), var * 100, label="% varianza")
        ax.plot(range(1, len(var) + 1), acum * 100, "o-", c="#C44E52", label="% acumulado")
        ax.axhline(80, ls="--", c="gray"); ax.set_xlabel("Componente"); ax.legend(); ax.set_title("Scree plot")
        st.pyplot(fig)
    with c2:
        n_pc = st.slider("Componentes a mostrar en las cargas", 2, len(PCA_VARS), 4)
        cargas = pd.DataFrame(pca.components_[:n_pc].T, index=PCA_VARS, columns=[f"PC{i+1}" for i in range(n_pc)])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cargas, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax); ax.set_title("Cargas")
        st.pyplot(fig)
    st.caption(f"Para 80 % de varianza se necesitan {int(np.argmax(acum >= .80) + 1)} componentes; "
               f"con autovalor > 1 (Kaiser) se retienen {int((pca.explained_variance_ > 1).sum())}.")

    st.subheader("Elección de k")

    @st.cache_data(show_spinner="Calculando codo y silueta...")
    def codo_silueta(Z):
        rs = np.random.RandomState(SEED)
        idx = rs.choice(len(Z), min(5000, len(Z)), replace=False)
        filas = []
        for k in range(2, 9):
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
            filas.append({"k": k, "Inercia": km.inertia_, "Silueta": silhouette_score(Z[idx], km.labels_[idx])})
        return pd.DataFrame(filas)

    tk = codo_silueta(Ztr)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.5)); ax.plot(tk["k"], tk["Inercia"], "o-"); ax.set_title("Codo"); ax.set_xlabel("k")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.5)); ax.plot(tk["k"], tk["Silueta"], "o-", c="#55A868"); ax.set_title("Silueta"); ax.set_xlabel("k")
        st.pyplot(fig)

    k = st.slider("Número de clústeres (k)", 2, 8, 4, help="En el informe se eligió k = 4 por estabilidad e interpretabilidad.")
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Ztr)
    t2 = train.assign(cluster=km.labels_)
    g = t2.groupby("cluster")
    perfil = pd.DataFrame({"n": g.size(), "% del total": (g.size() / len(t2) * 100).round(1),
                           "Edad media": g["edad"].mean().round(1),
                           "Costo mediano (COP)": g["costos"].median().round(0),
                           "% cirugía": (g["cx"].mean() * 100).round(1), "% fractura": (g["es_fractura"].mean() * 100).round(1),
                           "% politrauma": (g["politrauma"].mean() * 100).round(1), "% hombres": (g["sexo_masc"].mean() * 100).round(1),
                           "% moto": (g["es_moto"].mean() * 100).round(1), "% conductor": (g["es_conductor"].mean() * 100).round(1),
                           "Días hasta ingreso (mediana)": g["dias_hasta_ingreso"].median()})
    st.subheader("Perfiles en variables originales (entrenamiento)")
    st.dataframe(perfil, use_container_width=True)
    st.info("Con k = 4: un grupo quirúrgico de alto costo (~5 %), conductores motociclistas leves (~64 %), "
            "pasajeros o peatones leves (~29 %) e ingresos tardíos (~2 %).")

# ----------------------------------------------------------------------------
elif pagina == "Regresión lineal del costo":
    st.title("Regresión lineal múltiple: log10(costos)")
    st.markdown("Predictores: edad, días hasta ingreso, año, mes y día (cíclicos), sexo, vehículo, rol, IPS, pagador, "
                "afiliación, tipo de lesión, región, politrauma y cirugía. Se excluyen identificadores y variables posteriores.")
    cat_reg = CAT_PRE + CAT_DX + ["requerimiento_de_cx"]
    reg = Pipeline([("prep", preprocesador(NUM_REG, cat_reg)), ("lr", LinearRegression())]).fit(train, train["log_costos"])
    p_log = reg.predict(test)
    smear = np.mean(10 ** (train["log_costos"] - reg.predict(train)))   # corrección de Duan para volver a pesos
    y = test["costos"].values
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (log10)", f"{r2_score(test['log_costos'], p_log):.3f}")
    c2.metric("MAE (COP)", f"${mean_absolute_error(y, 10 ** p_log):,.0f}".replace(",", "."))
    c3.metric("RMSE (COP)", f"${np.sqrt(mean_squared_error(y, 10 ** p_log * smear)):,.0f}".replace(",", "."))
    c4.metric("Total predicho / real", f"{(10 ** p_log * smear).sum() / y.sum():.2f}")
    st.caption("10^ŷ estima el costo mediano (se usa para MAE); 10^ŷ × factor de Duan estima el costo medio (RMSE y totales).")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(test["log_costos"], p_log, s=5, alpha=.3)
        lim = [test["log_costos"].min(), test["log_costos"].max()]; ax.plot(lim, lim, "r--")
        ax.set_xlabel("log10(costo) real"); ax.set_ylabel("log10(costo) predicho"); ax.set_title("Prueba: real vs predicho")
        st.pyplot(fig)
    with c2:
        res = train["log_costos"] - reg.predict(train)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(reg.predict(train), res, s=5, alpha=.3); ax.axhline(0, c="r")
        ax.set_xlabel("ajustado"); ax.set_ylabel("residuo"); ax.set_title("Residuos vs ajustados (entrenamiento)")
        st.pyplot(fig)
    coef = pd.DataFrame({"variable": reg.named_steps["prep"].get_feature_names_out(), "coef_log10": reg.named_steps["lr"].coef_})
    coef["efecto_%"] = ((10 ** coef["coef_log10"] - 1) * 100).round(1)
    st.subheader("Coeficientes con mayor efecto")
    st.dataframe(coef.reindex(coef["coef_log10"].abs().sort_values(ascending=False).index).head(15), use_container_width=True)

# ----------------------------------------------------------------------------
elif pagina == "Modelos de clasificación":
    st.title("Construcción y evaluación de modelos: ¿requerirá cirugía?")
    nombre = st.selectbox("Algoritmo", list(EXPLICACIONES.keys()))
    st.info(EXPLICACIONES[nombre])

    c1, c2, c3 = st.columns(3)
    con_dx = c1.checkbox("Incluir diagnóstico (lesión, región, politrauma)", value=True)
    balanceado = c2.checkbox("Pesos balanceados (corrige el desbalance)", value=True)
    profundidad, criterio, n_arboles, C_svm = 8, "gini", 300, 1.0
    if nombre == "Árbol de decisión":
        profundidad = c3.slider("Profundidad máxima", 2, 15, 8)
        criterio = c3.radio("Criterio", ["gini", "entropy"], horizontal=True)
    elif nombre == "Random Forest":
        n_arboles = c3.slider("Número de árboles", 50, 500, 300, step=50)
    elif nombre.startswith("SVM"):
        C_svm = c3.select_slider("C (penalización)", [0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)
        c3.caption("Entrenado con una muestra estratificada de 6.000 pacientes.")

    modelo, y_val, prob_val = entrenar_clf(nombre, con_dx, balanceado, profundidad, criterio, n_arboles, C_svm,
                                           train, CLAVE_DATOS)
    prob = modelo.predict_proba(test)[:, 1]
    y = test["cx"].values

    st.markdown("**Umbral de decisión.** Un falso negativo (cirugía no detectada) cuesta más que un falso positivo "
                "(valoración innecesaria). Por defecto se usa el umbral que minimiza el costo en una validación "
                "interna del entrenamiento; el conjunto de prueba solo se usa para reportar.")
    c1, c2 = st.columns(2)
    razon = c1.slider("Costo de un FN frente a un FP", 1, 30, 10)
    t_opt = umbral_optimo(y_val, prob_val, razon)
    manual = c2.checkbox(f"Fijar el umbral a mano (óptimo por costo = {t_opt:.2f})", value=False)
    umbral = c2.slider("Umbral de probabilidad", 0.01, 0.95, round(t_opt, 2), 0.01) if manual else t_opt
    pred = (prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC", f"{roc_auc_score(y, prob):.3f}")
    m2.metric("PR-AUC", f"{average_precision_score(y, prob):.3f}", help=f"Línea base = prevalencia ({y.mean():.3f})")
    m3.metric("Recall (cirugías detectadas)", f"{tp / (tp + fn):.2f}" if tp + fn else "–")
    m4.metric("Alertas por 1.000 pacientes", f"{(tp + fp) / len(y) * 1000:.0f}")
    m5.metric("Costo del error", f"{fp + razon * fn:,}", delta=f"{fp + razon * fn - razon * y.sum():,} vs. sin modelo",
              delta_color="inverse")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Matriz de confusión (prueba)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred: no cx", "Pred: cx"], yticklabels=["Real: no cx", "Real: cx"])
        st.pyplot(fig)
        st.markdown("**Reporte de clasificación**")
        rep = classification_report(y, pred, target_names=["no cirugía", "cirugía"], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)
    with c2:
        st.markdown("**Curva ROC**")
        fpr, tpr, _ = roc_curve(y, prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, prob):.3f}"); ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos"); ax.legend()
        st.pyplot(fig)
        st.markdown("**Curva Precision–Recall**")
        pr, rc, _ = precision_recall_curve(y, prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rc, pr); ax.axhline(y.mean(), c="k", ls="--", label="prevalencia")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
        st.pyplot(fig)

    if nombre == "Regresión logística":
        nombres = modelo.named_steps["prep"].get_feature_names_out()
        orr = pd.DataFrame({"variable": nombres, "odds_ratio": np.exp(modelo.named_steps["clf"].coef_[0])})
        orr["|log OR|"] = np.abs(np.log(orr["odds_ratio"]))
        st.subheader("Odds ratios más influyentes")
        st.dataframe(orr.sort_values("|log OR|", ascending=False).head(12).drop(columns="|log OR|").round(3),
                     use_container_width=True)
    elif nombre in ("Árbol de decisión", "Random Forest"):
        imp = pd.Series(modelo.named_steps["clf"].feature_importances_,
                        index=modelo.named_steps["prep"].get_feature_names_out()).sort_values(ascending=False).head(12)
        st.subheader("Importancia de variables")
        st.bar_chart(imp)

    # Se guardan las predicciones para el módulo de descarga
    st.session_state["predicciones"] = pd.DataFrame({
        "numero": test["numero"].values, "edad": test["edad"].values, "ips": test["ips"].values,
        "patologia_principal": test["patologia_principal"].values, "cirugia_real": y,
        "prob_cirugia": prob.round(4), "alerta_cirugia": pred, "modelo": nombre, "umbral": umbral})

# ----------------------------------------------------------------------------
elif pagina == "Predicciones y descarga":
    st.title("Predicciones y descarga")
    if "predicciones" not in st.session_state:
        st.warning("Primero entrene un modelo en **Modelos de clasificación**; luego vuelva aquí para descargar sus predicciones.")
    else:
        p = st.session_state["predicciones"]
        st.write(f"Modelo: **{p['modelo'].iloc[0]}** · umbral {p['umbral'].iloc[0]:.2f} · {len(p):,} pacientes de prueba")
        solo_alertas = st.checkbox("Mostrar solo pacientes con alerta de cirugía")
        vista = p[p["alerta_cirugia"] == 1] if solo_alertas else p
        st.dataframe(vista.sort_values("prob_cirugia", ascending=False), use_container_width=True)
        st.download_button("⬇️ Descargar predicciones (CSV)", vista.to_csv(index=False).encode("utf-8"),
                           file_name="predicciones_cirugia.csv", mime="text/csv")

    st.divider()
    st.subheader("Predecir un paciente nuevo (regresión logística con diagnóstico)")
    modelo, y_val, prob_val = entrenar_clf("Regresión logística", True, True, 8, "gini", 300, 1.0, train, CLAVE_DATOS)
    t_nuevo = umbral_optimo(y_val, prob_val, 10)
    with st.form("nuevo"):
        c1, c2, c3 = st.columns(3)
        nuevo = {
            "edad": c1.number_input("Edad", 0, 110, 30),
            "dias_hasta_ingreso": c1.number_input("Días entre accidente e ingreso", 0, 400, 0),
            "sexo": c1.selectbox("Sexo", sorted(d["sexo"].unique())),
            "tipo_de_vehiculo": c2.selectbox("Vehículo", d["tipo_de_vehiculo"].value_counts().index.tolist()),
            "relacion_usuario_accidente": c2.selectbox("Rol", d["relacion_usuario_accidente"].value_counts().index.tolist()),
            "ips": c2.selectbox("IPS", sorted(d["ips"].unique())),
            "tipo_administradora": c3.selectbox("Pagador", d["tipo_administradora"].value_counts().index.tolist()),
            "tipo_afiliaci_n": c3.selectbox("Afiliación", d["tipo_afiliaci_n"].value_counts().index.tolist()),
            "patologia_principal": c3.selectbox("Tipo de lesión", d["patologia_principal"].value_counts().index.tolist()),
            "region_principal": c1.selectbox("Región anatómica", d["region_principal"].value_counts().index.tolist()),
            "politraumatismo": c2.selectbox("Politraumatismo", sorted(d["politraumatismo"].unique())),
        }
        enviar = st.form_submit_button("Calcular probabilidad")
    if enviar:
        pnuevo = modelo.predict_proba(pd.DataFrame([nuevo]))[0, 1]
        st.metric("Probabilidad estimada de cirugía", f"{pnuevo*100:.1f} %")
        if pnuevo >= t_nuevo:
            st.error(f"Supera el umbral {t_nuevo:.2f} (costo FN = 10 × FP): activar valoración quirúrgica temprana.")
        else:
            st.success(f"No supera el umbral {t_nuevo:.2f}: manejo según protocolo habitual.")
