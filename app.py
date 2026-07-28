from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Instacart ML Dashboard")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# ============================================================
# ENTRENAR MODELOS AL INICIAR
# ============================================================
print("Entrenando modelos...")
ruta_base = os.path.join(os.path.dirname(__file__), "archivos")
df = pd.read_csv(os.path.join(ruta_base, "dataset_mejorado.csv"))
df = df[df['total_productos'] < 40].copy()
df['days_since_prior_order'] = df['days_since_prior_order'].fillna(0)
df['pedido_7d'] = (df['days_since_prior_order'] <= 7).astype(int)
df['tamano_carrito'] = df['total_productos']

# Clustering
cluster_cols = [
    'promedio_dias_espera', 'std_dias_espera', 'total_pedidos_historicos',
    'promedio_productos', 'promedio_recompra_usuario', 'promedio_dept_usuario',
    'std_carrito', 'n_departamentos_historial', 'n_aisles_historial', 'tendencia_dias'
]
fc = df[cluster_cols].fillna(df[cluster_cols].mean())
scaler_c = StandardScaler()
df['Cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaler_c.fit_transform(fc))

estabilidad = df.groupby('Cluster')['std_dias_espera'].mean()
c_estable = estabilidad.idxmin()
c_caotico = estabilidad.idxmax()
c_regular = [c for c in range(3) if c not in (c_estable, c_caotico)][0]

# Features
exclude_all = ['days_since_prior_order', 'pedido_7d', 'tamano_carrito', 'order_id', 'user_id', 'order_number', 'Cluster']
exclude_v4_extra = ['productos_recomprados', 'ratio_recompra', 'cart_vs_avg', 'cart_x_recompra', 'cart_x_dept']

all_v1 = [c for c in df.columns if c not in exclude_all + ['total_productos']]
tc = df[all_v1 + ['pedido_7d']].corr()['pedido_7d'].drop('pedido_7d').abs()
features_v1 = [f for f in all_v1 if tc[f] >= 0.01]

all_v4 = [c for c in df.columns if c not in exclude_all + exclude_v4_extra + ['total_productos']]
tc4 = df[all_v4 + ['tamano_carrito']].corr()['tamano_carrito'].drop('tamano_carrito').abs()
features_v4 = [f for f in all_v4 if tc4[f] >= 0.01]

# Entrenar modelos por cluster
modelos = {}
for cid in [c_estable, c_regular, c_caotico]:
    dc = df[df['Cluster'] == cid]
    X1 = dc[features_v1]; y1 = dc['pedido_7d']
    X4 = dc[features_v4]; y4 = dc['tamano_carrito']
    n_neg = (y1==0).sum(); n_pos = (y1==1).sum()

    # V1 - Clasificacion (3 modelos)
    xgb_c = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=n_neg/n_pos, random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0)
    xgb_c.fit(X1, y1)

    cat_c = CatBoostClassifier(iterations=200, depth=7, learning_rate=0.08, auto_class_weights='Balanced', random_seed=42, verbose=0)
    cat_c.fit(X1, y1)

    lgbm_c = LGBMClassifier(n_estimators=200, max_depth=7, learning_rate=0.1, is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1)
    lgbm_c.fit(X1, y1)

    # V4 - Regresion (3 modelos)
    xgb_r = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
    xgb_r.fit(X4, y4)

    cat_r = CatBoostRegressor(iterations=200, depth=7, learning_rate=0.08, random_seed=42, verbose=0)
    cat_r.fit(X4, y4)

    lgbm_r = LGBMRegressor(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    lgbm_r.fit(X4, y4)

    modelos[cid] = {
        'xgb_c': xgb_c, 'cat_c': cat_c, 'lgbm_c': lgbm_c,
        'xgb_r': xgb_r, 'cat_r': cat_r, 'lgbm_r': lgbm_r
    }
    print(f"  C{cid}: entrenado ({len(dc):,} registros)")

# Metricas
nc = {c_estable: 'Estable', c_regular: 'Regular', c_caotico: 'Caotico'}
metricas = {
    c_estable: {
        'nombre': 'Estable', 'color': '#2ecc71', 'emoji': '🟢',
        'registros': int(len(df[df['Cluster']==c_estable])),
        'usuarios': int(df[df['Cluster']==c_estable]['user_id'].nunique()),
        'f1': 70.6, 'auc': 0.756, 'precision': 58.8, 'recall': 88.4,
        'mae': 2.78, 'r2': 0.544, 'umbral': 0.30,
        'prom_dias': round(df[df['Cluster']==c_estable]['days_since_prior_order'].mean(), 1),
        'pct_7d': round((df[df['Cluster']==c_estable]['days_since_prior_order'] <= 7).mean() * 100, 1),
        'mejor_modelo': 'Stacking (XGB+Cat+LGBM)',
        'descripcion': 'Clientes con patron regular y predecible de compra. Compran cada ~14 dias con baja variabilidad.',
        'accion': 'Campanas agresivas. Alta confianza en la prediccion.',
        'roi': 'Enviar promociones a este grupo genera el mayor retorno por inversion.'
    },
    c_regular: {
        'nombre': 'Regular', 'color': '#f39c12', 'emoji': '🟡',
        'registros': int(len(df[df['Cluster']==c_regular])),
        'usuarios': int(df[df['Cluster']==c_regular]['user_id'].nunique()),
        'f1': 50.2, 'auc': 0.612, 'precision': 39.8, 'recall': 66.2,
        'mae': 4.65, 'r2': 0.461, 'umbral': 0.40,
        'prom_dias': round(df[df['Cluster']==c_regular]['days_since_prior_order'].mean(), 1),
        'pct_7d': round((df[df['Cluster']==c_regular]['days_since_prior_order'] <= 7).mean() * 100, 1),
        'mejor_modelo': 'CatBoost',
        'descripcion': 'Clientes intermedios con algo de variabilidad. Compran cada ~18 dias.',
        'accion': 'Score de propension. Campanas moderadas con filtros.',
        'roi': 'Balance entre alcance y precision. Segmentar por probabilidad.'
    },
    c_caotico: {
        'nombre': 'Caotico', 'color': '#e74c3c', 'emoji': '🔴',
        'registros': int(len(df[df['Cluster']==c_caotico])),
        'usuarios': int(df[df['Cluster']==c_caotico]['user_id'].nunique()),
        'f1': 35.9, 'auc': 0.548, 'precision': 24.2, 'recall': 65.6,
        'mae': 2.40, 'r2': 0.541, 'umbral': 0.45,
        'prom_dias': round(df[df['Cluster']==c_caotico]['days_since_prior_order'].mean(), 1),
        'pct_7d': round((df[df['Cluster']==c_caotico]['days_since_prior_order'] <= 7).mean() * 100, 1),
        'mejor_modelo': 'CatBoost',
        'descripcion': 'Clientes impredecibles. Patrones de compra irregulares.',
        'accion': 'Solo scoring. No invertir en campanas agresivas.',
        'roi': 'Bajo ROI en intervenciones. Usar solo para analisis.'
    }
}

# ============================================================
# RUTAS
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "metricas": metricas,
        "clusters_ord": [c_estable, c_regular, c_caotico],
        "total_registros": int(len(df)),
        "total_usuarios": int(df['user_id'].nunique()),
        "target_media": float(round(df['days_since_prior_order'].mean(), 1)),
        "target_mediana": float(round(df['days_since_prior_order'].median(), 1)),
        "pct_7d": float(round((df['days_since_prior_order'] <= 7).mean() * 100, 1)),
    })

@app.get("/api/predict")
async def predict(
    cluster: int = 0,
    total_productos: int = 8,
    productos_recomprados: int = 3,
    promedio_dias_espera: float = 17.0,
    std_dias_espera: float = 10.0,
    total_pedidos_historicos: int = 10,
    promedio_productos: float = 8.0,
    promedio_recompra_usuario: float = 0.5,
    promedio_dept_usuario: float = 3.0,
    std_carrito: float = 3.0,
    n_departamentos_historial: int = 5,
    n_aisles_historial: int = 8,
    tendencia_dias: float = 0.0,
    n_products_historial: int = 30,
    order_hour_of_day: int = 12,
    order_dow: int = 3,
    order_number: int = 5
):
    """Prediccion real usando los modelos entrenados por cluster"""
    cid = cluster if cluster in modelos else c_estable
    m = modelos[cid]

    ratio_recompra = productos_recomprados / max(total_productos, 1)
    cart_vs_avg = total_productos / max(promedio_productos, 1)
    hour_sin = np.sin(2 * np.pi * order_hour_of_day / 24)
    hour_cos = np.cos(2 * np.pi * order_hour_of_day / 24)
    dow_sin = np.sin(2 * np.pi * order_dow / 7)
    dow_cos = np.cos(2 * np.pi * order_dow / 7)
    es_finde = 1 if order_dow in [5, 6] else 0
    es_horario_laboral = 1 if 9 <= order_hour_of_day <= 17 else 0
    es_noche = 1 if order_hour_of_day >= 19 or order_hour_of_day <= 5 else 0
    order_number_norm = order_number / 100
    es_primera_compra = 1 if order_number == 1 else 0
    cart_x_recompra = total_productos * ratio_recompra
    cart_x_dept = total_productos * promedio_dept_usuario
    std_x_tendencia = std_dias_espera * tendencia_dias
    recompra_x_std = promedio_recompra_usuario * std_carrito
    lifecycle_pos = order_number / (total_pedidos_historicos + 1)
    es_pedido_reciente = 1 if order_number >= total_pedidos_historicos - 2 else 0
    densidad_compra = n_products_historial / (total_pedidos_historicos + 1)

    raw_features = {
        'promedio_dias_espera': promedio_dias_espera, 'std_dias_espera': std_dias_espera,
        'total_pedidos_historicos': total_pedidos_historicos, 'promedio_productos': promedio_productos,
        'promedio_recompra_usuario': promedio_recompra_usuario, 'promedio_dept_usuario': promedio_dept_usuario,
        'std_carrito': std_carrito, 'n_departamentos_historial': n_departamentos_historial,
        'n_aisles_historial': n_aisles_historial, 'tendencia_dias': tendencia_dias,
        'n_products_historial': n_products_historial,
        'hour_sin': hour_sin, 'hour_cos': hour_cos, 'dow_sin': dow_sin, 'dow_cos': dow_cos,
        'es_finde': es_finde, 'es_horario_laboral': es_horario_laboral, 'es_noche': es_noche,
        'total_productos': total_productos, 'productos_recomprados': productos_recomprados,
        'ratio_recompra': ratio_recompra, 'cart_vs_avg': cart_vs_avg,
        'order_number_norm': order_number_norm, 'es_primera_compra': es_primera_compra,
        'cart_x_recompra': cart_x_recompra, 'cart_x_dept': cart_x_dept,
        'std_x_tendencia': std_x_tendencia, 'recompra_x_std': recompra_x_std,
        'lifecycle_pos': lifecycle_pos, 'es_pedido_reciente': es_pedido_reciente,
        'densidad_compra': densidad_compra,
        'avg_product_popularity': 0.5, 'avg_product_reorder_rate': ratio_recompra,
        'max_product_popularity': 0.7
    }

    # V1 - Stacking
    df_v1 = pd.DataFrame([{f: raw_features.get(f, 0) for f in features_v1}])
    prob_xgb = float(m['xgb_c'].predict_proba(df_v1)[:, 1][0])
    prob_cat = float(m['cat_c'].predict_proba(df_v1)[:, 1][0])
    prob_lgbm = float(m['lgbm_c'].predict_proba(df_v1)[:, 1][0])
    prob_stack = (prob_xgb + prob_cat + prob_lgbm) / 3

    umbral = metricas[cid]['umbral']
    pred_7d = "SI" if prob_stack >= umbral else "NO"

    # V4 - CatBoost
    df_v4 = pd.DataFrame([{f: raw_features.get(f, 0) for f in features_v4}])
    pred_carrito_xgb = max(1, round(float(m['xgb_r'].predict(df_v4)[0])))
    pred_carrito_cat = max(1, round(float(m['cat_r'].predict(df_v4)[0])))
    pred_carrito_lgbm = max(1, round(float(m['lgbm_r'].predict(df_v4)[0])))
    pred_carrito_stack = round((pred_carrito_xgb + pred_carrito_cat + pred_carrito_lgbm) / 3)

    return {
        "cluster": cid,
        "cluster_nombre": nc[cid],
        "v1": {
            "prob_xgb": round(prob_xgb * 100, 1),
            "prob_cat": round(prob_cat * 100, 1),
            "prob_lgbm": round(prob_lgbm * 100, 1),
            "prob_stacking": round(prob_stack * 100, 1),
            "prediccion": pred_7d,
            "umbral": umbral,
            "confianza": "Alta" if metricas[cid]['f1'] > 60 else ("Media" if metricas[cid]['f1'] > 40 else "Baja")
        },
        "v4": {
            "pred_xgb": pred_carrito_xgb,
            "pred_cat": pred_carrito_cat,
            "pred_lgbm": pred_carrito_lgbm,
            "pred_stacking": pred_carrito_stack
        }
    }

@app.get("/api/cluster_info")
async def cluster_info(cluster: int = 0):
    cid = cluster if cluster in metricas else c_estable
    return metricas[cid]
