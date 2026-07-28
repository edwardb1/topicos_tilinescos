import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  GRAFICOS DE DESEMPEÑO - VARIABLE OBJETIVO")
print("=" * 70)

# ============================================================
# 1. CARGAR Y PREPARAR
# ============================================================
print("\n[1/4] Cargando datos...")
ruta = r"C:/Users/edwar/OneDrive/Desktop/Universidad/topicos avanzados II/archivos/dataset_mejorado.csv"
df = pd.read_csv(ruta)
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
df['Cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(fc))

estabilidad = df.groupby('Cluster')['std_dias_espera'].mean()
c_estable = estabilidad.idxmin()
c_caotico = estabilidad.idxmax()
c_regular = [c for c in range(3) if c not in (c_estable, c_caotico)][0]
nc = {c_estable: 'ESTABLE', c_regular: 'REGULAR', c_caotico: 'CAOTICO'}
clusters_ord = [c_estable, c_regular, c_caotico]
colores = {c_estable: '#2ecc71', c_regular: '#f39c12', c_caotico: '#e74c3c'}

# Features
exclude_v1 = ['days_since_prior_order', 'pedido_7d', 'tamano_carrito', 'order_id', 'user_id', 'order_number', 'Cluster', 'total_productos']
all_v1 = [c for c in df.columns if c not in exclude_v1]
tc = df[all_v1 + ['pedido_7d']].corr()['pedido_7d'].drop('pedido_7d').abs()
features_v1 = [f for f in all_v1 if tc[f] >= 0.01]

exclude_v4 = exclude_v1 + ['productos_recomprados', 'ratio_recompra', 'cart_vs_avg', 'cart_x_recompra', 'cart_x_dept']
all_v4 = [c for c in df.columns if c not in exclude_v4]
tc4 = df[all_v4 + ['tamano_carrito']].corr()['tamano_carrito'].drop('tamano_carrito').abs()
features_v4 = [f for f in all_v4 if tc4[f] >= 0.01]

print(f"  Registros: {len(df):,} | Features V1: {len(features_v1)} | V4: {len(features_v4)}")

# ============================================================
# 2. ENTRENAR Y RECOLECTAR PREDICCIONES
# ============================================================
print("\n[2/4] Entrenando modelos y recolectando predicciones...")
predicciones = {}

for cid in clusters_ord:
    dc = df[df['Cluster'] == cid].copy()
    X1 = dc[features_v1].copy()
    y1 = dc['pedido_7d'].copy()
    X4 = dc[features_v4].copy()
    y4 = dc['tamano_carrito'].copy()
    groups = dc['user_id'].values
    n_neg = (y1==0).sum(); n_pos = (y1==1).sum()

    all_prob_xgb, all_prob_cat, all_prob_lgbm = [], [], []
    all_pred_xgb_v4, all_pred_cat_v4, all_pred_lgbm_v4 = [], [], []
    all_real_v1, all_real_v4, all_idx = [], [], []

    gkf = GroupKFold(3)
    for tr_i, te_i in gkf.split(X1, y1, groups):
        # V1
        Xtr, Xte = X1.iloc[tr_i], X1.iloc[te_i]
        ytr, yte = y1.iloc[tr_i], y1.iloc[te_i]

        mx = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=n_neg/n_pos, random_state=42, n_jobs=-1, eval_metric='logloss', verbosity=0)
        mx.fit(Xtr, ytr)
        all_prob_xgb.extend(mx.predict_proba(Xte)[:,1])

        mc = CatBoostClassifier(iterations=200, depth=7, learning_rate=0.08, auto_class_weights='Balanced', random_seed=42, verbose=0)
        mc.fit(Xtr, ytr)
        all_prob_cat.extend(mc.predict_proba(Xte)[:,1])

        ml = LGBMClassifier(n_estimators=200, max_depth=7, learning_rate=0.1, is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1)
        ml.fit(Xtr, ytr)
        all_prob_lgbm.extend(ml.predict_proba(Xte)[:,1])

        all_real_v1.extend(yte.values)

        # V4
        X4tr, X4te = X4.iloc[tr_i], X4.iloc[te_i]
        y4tr, y4te = y4.iloc[tr_i], y4.iloc[te_i]

        mx4 = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
        mx4.fit(X4tr, y4tr)
        all_pred_xgb_v4.extend(mx4.predict(X4te))

        mc4 = CatBoostRegressor(iterations=200, depth=7, learning_rate=0.08, random_seed=42, verbose=0)
        mc4.fit(X4tr, y4tr)
        all_pred_cat_v4.extend(mc4.predict(X4te))

        ml4 = LGBMRegressor(n_estimators=200, max_depth=7, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
        ml4.fit(X4tr, y4tr)
        all_pred_lgbm_v4.extend(ml4.predict(X4te))

        all_real_v4.extend(y4te.values)
        all_idx.extend(dc.iloc[te_i].index)

    predicciones[cid] = {
        'prob_xgb': np.array(all_prob_xgb),
        'prob_cat': np.array(all_prob_cat),
        'prob_lgbm': np.array(all_prob_lgbm),
        'real_v1': np.array(all_real_v1),
        'pred_xgb_v4': np.array(all_pred_xgb_v4),
        'pred_cat_v4': np.array(all_pred_cat_v4),
        'pred_lgbm_v4': np.array(all_pred_lgbm_v4),
        'real_v4': np.array(all_real_v4),
        'idx': np.array(all_idx)
    }
    print(f"  C{cid} {nc[cid]}: OK")

ruta_reportes = r"C:/Users/edwar/OneDrive/Desktop/Universidad/topicos avanzados II/REPORTES/"

# ============================================================
# 3. GENERAR GRAFICOS
# ============================================================
print("\n[3/4] Generando graficos...")
plt.style.use('seaborn-v0_8-whitegrid')

# ────────────────────────────────────────────────────────────
# GRAFICO 1: Distribucion de la variable objetivo por cluster
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('DISTRIBUCION DE LA VARIABLE OBJETIVO\n days_since_prior_order', fontsize=16, fontweight='bold')

# 1a: Histograma general
ax = axes[0, 0]
ax.hist(df['days_since_prior_order'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
ax.axvline(x=7, color='red', linestyle='--', lw=2, label='Umbral 7d')
ax.axvline(x=df['days_since_prior_order'].mean(), color='orange', linestyle='--', lw=2, label=f'Media={df["days_since_prior_order"].mean():.1f}')
ax.set_title('Distribucion General', fontweight='bold')
ax.set_xlabel('Days Since Prior Order')
ax.set_ylabel('Frecuencia')
ax.legend()

# 1b: Boxplot por cluster
ax = axes[0, 1]
data_box = [df[df['Cluster']==c]['days_since_prior_order'].values for c in clusters_ord]
bp = ax.boxplot(data_box, tick_labels=[f'C{c}\n{nc[c]}' for c in clusters_ord], patch_artist=True)
for patch, c in zip(bp['boxes'], clusters_ord):
    patch.set_facecolor(colores[c])
    patch.set_alpha(0.7)
ax.axhline(y=7, color='red', linestyle='--', lw=2, label='Umbral 7d')
ax.set_title('Dias entre Pedidos por Cluster', fontweight='bold')
ax.set_ylabel('Days Since Prior Order')
ax.legend()

# 1c: KDE por cluster
ax = axes[1, 0]
for c in clusters_ord:
    subset = df[df['Cluster']==c]['days_since_prior_order']
    subset.plot.kde(ax=ax, color=colores[c], label=f'C{c} {nc[c]}', lw=2)
ax.axvline(x=7, color='red', linestyle='--', lw=2, alpha=0.5)
ax.set_title('Densidad de Dias entre Pedidos', fontweight='bold')
ax.set_xlabel('Days Since Prior Order')
ax.legend()

# 1d: Proporcion <=7d por cluster
ax = axes[1, 1]
props = [(df[df['Cluster']==c]['days_since_prior_order'] <= 7).mean()*100 for c in clusters_ord]
bars = ax.bar([f'C{c}\n{nc[c]}' for c in clusters_ord], props, color=[colores[c] for c in clusters_ord], edgecolor='black', alpha=0.8)
for b, p in zip(bars, props):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{p:.1f}%', ha='center', fontweight='bold')
ax.set_title('Proporcion de Pedidos <= 7 dias por Cluster', fontweight='bold')
ax.set_ylabel('%')
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(ruta_reportes + 'grafico1_distribucion_target.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] grafico1_distribucion_target.png")

# ────────────────────────────────────────────────────────────
# GRAFICO 2: ROC Curves y Precision-Recall (V1)
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('V1: CURVAS ROC Y PRECISION-RECALL (Pedido en <=7d)', fontsize=14, fontweight='bold')

for idx, cid in enumerate(clusters_ord):
    p = predicciones[cid]
    prob_stack = (p['prob_xgb'] + p['prob_cat'] + p['prob_lgbm']) / 3
    real = p['real_v1']

    # ROC
    fpr, tpr, _ = roc_curve(real, prob_stack)
    auc_val = auc(fpr, tpr)
    axes[idx].plot(fpr, tpr, color=colores[cid], lw=2.5, label=f'Stacking AUC={auc_val:.3f}')

    # ROC individual
    for name, prob in [('XGB', p['prob_xgb']), ('Cat', p['prob_cat']), ('LGBM', p['prob_lgbm'])]:
        fpr_i, tpr_i, _ = roc_curve(real, prob)
        auc_i = auc(fpr_i, tpr_i)
        axes[idx].plot(fpr_i, tpr_i, '--', alpha=0.5, lw=1, label=f'{name} AUC={auc_i:.3f}')

    axes[idx].plot([0,1], [0,1], 'k--', lw=1, alpha=0.5)
    axes[idx].set_title(f'C{cid} {nc[cid]}\nROC Curve', fontweight='bold')
    axes[idx].set_xlabel('False Positive Rate')
    axes[idx].set_ylabel('True Positive Rate')
    axes[idx].legend(fontsize=7, loc='lower right')

plt.tight_layout()
plt.savefig(ruta_reportes + 'grafico2_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] grafico2_roc_curves.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('V1: CURVA PRECISION-RECALL (Pedido en <=7d)', fontsize=14, fontweight='bold')

for idx, cid in enumerate(clusters_ord):
    p = predicciones[cid]
    prob_stack = (p['prob_xgb'] + p['prob_cat'] + p['prob_lgbm']) / 3
    real = p['real_v1']

    prec, rec, _ = precision_recall_curve(real, prob_stack)
    pr_auc = auc(rec, prec)
    axes[idx].plot(rec, prec, color=colores[cid], lw=2.5, label=f'Stacking AP={pr_auc:.3f}')

    baseline = real.mean()
    axes[idx].axhline(y=baseline, color='gray', linestyle='--', lw=1, alpha=0.5, label=f'Baseline={baseline:.2f}')
    axes[idx].set_title(f'C{cid} {nc[cid]}\nPrecision-Recall', fontweight='bold')
    axes[idx].set_xlabel('Recall')
    axes[idx].set_ylabel('Precision')
    axes[idx].legend(fontsize=8)

plt.tight_layout()
plt.savefig(ruta_reportes + 'grafico3_precision_recall.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] grafico3_precision_recall.png")

# ────────────────────────────────────────────────────────────
# GRAFICO 3: Matrices de confusion por cluster
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('V1: MATRICES DE CONFUSION (Umbral optimizado)', fontsize=14, fontweight='bold')

umbrales = {c_estable: 0.30, c_regular: 0.40, c_caotico: 0.50}

for idx, cid in enumerate(clusters_ord):
    p = predicciones[cid]
    prob_stack = (p['prob_xgb'] + p['prob_cat'] + p['prob_lgbm']) / 3
    pred = (prob_stack >= umbrales[cid]).astype(int)
    cm = confusion_matrix(p['real_v1'], pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No (7d+)', 'Si (<=7d)'],
                yticklabels=['No (7d+)', 'Si (<=7d)'],
                cbar=False, annot_kws={'size': 14})
    axes[idx].set_title(f'C{cid} {nc[cid]}\n(umbral={umbrales[cid]:.2f})', fontweight='bold')
    axes[idx].set_ylabel('Real')
    axes[idx].set_xlabel('Predicho')

plt.tight_layout()
plt.savefig(ruta_reportes + 'grafico4_matriz_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] grafico4_matriz_confusion.png")

# ────────────────────────────────────────────────────────────
# GRAFICO 4: V4 Predicho vs Real (scatter)
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('V4: TAMANO DEL CARRITO - PREDICHO vs REAL', fontsize=16, fontweight='bold')

for idx, cid in enumerate(clusters_ord):
    p = predicciones[cid]
    real = p['real_v4']

    for row, (name, pred) in enumerate([
        ('XGBoost', p['pred_xgb_v4']),
        ('CatBoost', p['pred_cat_v4']),
    ]):
        ax = axes[row, idx]
        ax.scatter(real, pred, alpha=0.1, s=5, color=colores[cid])
        lim = max(real.max(), pred.max()) + 1
        ax.plot([0, lim], [0, lim], 'r--', lw=2, label='Perfecto')
        mae = mean_absolute_error(real, pred)
        r2 = r2_score(real, pred)
        ax.set_title(f'C{cid} {nc[cid]} - {name}\nMAE={mae:.2f} | R2={r2:.3f}', fontweight='bold')
        ax.set_xlabel('Real')
        ax.set_ylabel('Predicho')
        ax.legend(fontsize=8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

plt.tight_layout()
plt.savefig(ruta_reportes + 'grafico5_pred_vs_real_v4.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] grafico5_pred_vs_real_v4.png")

# ────────────────────────────────────────────────────────────
# GRAFICO 5: Residuales V4
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('V4: ANALISIS DE RESIDUALES (CatBoost)', fontsize=14, fontweight='bold')

for idx, cid in enumerate(clusters_ord):
    p = predicciones[cid]
    real = p['real_v4']
    pred = p['pred_cat_v4']
    residuales = real - pred

    axes[idx].scatter(pred, residuales, alpha=0.15, s=5, color=colores[cid])
    axes[idx].axhline(y=0, color='red', linestyle='--', lw=2)
    axes[idx].set_title(f'C{cid} {nc[cid]}\nResiduales (media={residuales.mean():.2f})', fontweight='bold')
    axes[idx].set_xlabel('Predicho')
    axes[idx].set_ylabel('Residual (Real - Predicho)')

plt.tight_layout()
plt.savefig(ruta_reportes + 'grafico6_residuales_v4.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] grafico6_residuales_v4.png")

# ────────────────────────────────────────────────────────────
# GRAFICO 6: Dias_since_prior_order vs Prediccion (V1 prob)
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('V1: PROBABILIDAD DE COMPRA EN 7d vs DIAS REALES', fontsize=14, fontweight='bold')

for idx, cid in enumerate(clusters_ord):
    p = predicciones[cid]
    prob_stack = (p['prob_xgb'] + p['prob_cat'] + p['prob_lgbm']) / 3
    real_days = df.loc[p['idx'], 'days_since_prior_order'].values

    axes[idx].scatter(real_days, prob_stack, alpha=0.15, s=5, color=colores[cid])
    axes[idx].axvline(x=7, color='red', linestyle='--', lw=2, label='Umbral 7d')
    axes[idx].axhline(y=umbrales[cid], color='gray', linestyle=':', lw=1.5, label=f'Umbral={umbrales[cid]:.2f}')
    axes[idx].set_title(f'C{cid} {nc[cid]}', fontweight='bold')
    axes[idx].set_xlabel('Dias reales entre pedidos')
    axes[idx].set_ylabel('Probabilidad predicha (<=7d)')
    axes[idx].legend(fontsize=8)
    axes[idx].set_xlim(0, 50)

plt.tight_layout()
plt.savefig(ruta_reportes + 'grafico7_prob_vs_dias_reales.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] grafico7_prob_vs_dias_reales.png")

# ────────────────────────────────────────────────────────────
# GRAFICO 7: Feature Importance (CatBoost V1)
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle('V1: TOP 10 FEATURES - CatBoost (Importancia)', fontsize=14, fontweight='bold')

for idx, cid in enumerate(clusters_ord):
    dc = df[df['Cluster']==cid]
    X1 = dc[features_v1]; y1 = dc['pedido_7d']
    n_neg = (y1==0).sum(); n_pos = (y1==1).sum()
    mc = CatBoostClassifier(iterations=200, depth=7, learning_rate=0.08, auto_class_weights='Balanced', random_seed=42, verbose=0)
    mc.fit(X1, y1)
    imp = pd.Series(mc.feature_importances_, index=features_v1).nlargest(10)

    axes[idx].barh(range(len(imp)), imp.values, color=colores[cid], alpha=0.8, edgecolor='black')
    axes[idx].set_yticks(range(len(imp)))
    axes[idx].set_yticklabels(imp.index, fontsize=9)
    axes[idx].set_title(f'C{cid} {nc[cid]}', fontweight='bold')
    axes[idx].set_xlabel('Importancia')
    axes[idx].invert_yaxis()

plt.tight_layout()
plt.savefig(ruta_reportes + 'grafico8_feature_importance_v1.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] grafico8_feature_importance_v1.png")

# ────────────────────────────────────────────────────────────
# GRAFICO 8: Feature Importance (CatBoost V4)
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle('V4: TOP 10 FEATURES - CatBoost (Importancia)', fontsize=14, fontweight='bold')

for idx, cid in enumerate(clusters_ord):
    dc = df[df['Cluster']==cid]
    X4 = dc[features_v4]; y4 = dc['tamano_carrito']
    mc4 = CatBoostRegressor(iterations=200, depth=7, learning_rate=0.08, random_seed=42, verbose=0)
    mc4.fit(X4, y4)
    imp = pd.Series(mc4.feature_importances_, index=features_v4).nlargest(10)

    axes[idx].barh(range(len(imp)), imp.values, color=colores[cid], alpha=0.8, edgecolor='black')
    axes[idx].set_yticks(range(len(imp)))
    axes[idx].set_yticklabels(imp.index, fontsize=9)
    axes[idx].set_title(f'C{cid} {nc[cid]}', fontweight='bold')
    axes[idx].set_xlabel('Importancia')
    axes[idx].invert_yaxis()

plt.tight_layout()
plt.savefig(ruta_reportes + 'grafico9_feature_importance_v4.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] grafico9_feature_importance_v4.png")

# ────────────────────────────────────────────────────────────
# GRAFICO 9: Resumen ejecutivo con metricas clave
# ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 10))
ax.axis('off')

ax.text(0.5, 0.97, 'RESUMEN EJECUTIVO - DESEMPEÑO DEL MODELO',
        ha='center', va='top', fontsize=18, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.93, 'Variable Objetivo: days_since_prior_order  |  Clasificacion: Compra en <=7 dias  |  Regresion: Tamano del carrito',
        ha='center', va='top', fontsize=10, color='gray', transform=ax.transAxes)

headers = ['Cluster', 'Registros', 'F1 Clase+\n(V1)', 'AUC\n(V1)', 'MAE\n(V4)', 'R2\n(V4)', 'Mejor\nModelo', 'Umbral']
table_data = []

for cid in clusters_ord:
    p = predicciones[cid]
    prob_stack = (p['prob_xgb'] + p['prob_cat'] + p['prob_lgbm']) / 3
    real_v1 = p['real_v1']
    pred_v1 = (prob_stack >= umbrales[cid]).astype(int)
    f1 = f1_score(real_v1, pred_v1)*100
    fpr, tpr, _ = roc_curve(real_v1, prob_stack)
    auc_val = auc(fpr, tpr)
    mae = mean_absolute_error(p['real_v4'], p['pred_cat_v4'])
    r2 = r2_score(p['real_v4'], p['pred_cat_v4'])

    best = 'CatBoost' if mae == min(mean_absolute_error(p['real_v4'], p['pred_cat_v4']),
                                     mean_absolute_error(p['real_v4'], p['pred_xgb_v4']),
                                     mean_absolute_error(p['real_v4'], p['pred_lgbm_v4'])) else 'XGBoost'

    table_data.append([
        f'C{cid} {nc[cid]}',
        f'{len(df[df["Cluster"]==cid]):,}',
        f'{f1:.1f}%',
        f'{auc_val:.3f}',
        f'{mae:.2f}',
        f'{r2:.3f}',
        best,
        f'{umbrales[cid]:.2f}'
    ])

tbl = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
               colColours=['#2c3e50']*8)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.3, 2.0)
for i in range(3):
    for j in range(8):
        tbl[i+1, j].set_facecolor('#ecf0f1' if i%2==0 else '#ffffff')

ax.text(0.5, 0.08, 'Modelos: XGBoost (Optuna) + CatBoost + LightGBM | Validacion: GroupKFold (user_id)',
        ha='center', fontsize=9, color='gray', transform=ax.transAxes)

plt.tight_layout()
plt.savefig(ruta_reportes + 'grafico10_resumen_ejecutivo.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] grafico10_resumen_ejecutivo.png")

# ============================================================
# 4. RESUMEN
# ============================================================
print(f"\n{'='*70}")
print("  10 GRAFICOS GENERADOS EN REPORTES/")
print(f"{'='*70}")
print("""
  grafico1_distribucion_target.png      - Distribucion de days_since_prior_order
  grafico2_roc_curves.png               - Curvas ROC (AUC) por cluster
  grafico3_precision_recall.png         - Curvas Precision-Recall por cluster
  grafico4_matriz_confusion.png         - Matrices de confusion V1
  grafico5_pred_vs_real_v4.png          - Scatter Predicho vs Real (V4)
  grafico6_residuales_v4.png            - Analisis de residuales (V4)
  grafico7_prob_vs_dias_reales.png      - Probabilidad vs dias reales
  grafico8_feature_importance_v1.png    - Top 10 features V1 (CatBoost)
  grafico9_feature_importance_v4.png    - Top 10 features V4 (CatBoost)
  grafico10_resumen_ejecutivo.png       - Tabla resumen ejecutiva
""")
