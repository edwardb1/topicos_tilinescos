import pandas as pd
import numpy as np

print("=" * 60)
print("  FEATURE ENGINEERING V4 - FEATURES COMPLETAS")
print("=" * 60)

# 1. Cargar datos
print("\n[1/8] Cargando datos...")
ruta_base = r"C:/Users/edwar/OneDrive/Desktop/Universidad/topicos avanzados II/archivos/"
df_orders = pd.read_csv(ruta_base + "orders.csv")
df_prior = pd.read_csv(ruta_base + "order_products__prior.csv")
df_train_products = pd.read_csv(ruta_base + "order_products__train.csv")
df_products = pd.read_csv(ruta_base + "products.csv")
df_aisles = pd.read_csv(ruta_base + "aisles.csv")
df_departments = pd.read_csv(ruta_base + "departments.csv")

print(f"  Orders: {len(df_orders):,}")
print(f"  Prior: {len(df_prior):,}")
print(f"  Train: {len(df_train_products):,}")
print(f"  Products: {len(df_products):,}")
print(f"  Aisles: {len(df_aisles):,}")
print(f"  Departments: {len(df_departments):,}")

# 2. Limpieza base
print("\n[2/8] Limpiando datos...")
df_orders['days_since_prior_order'] = df_orders['days_since_prior_order'].fillna(0)

# 3. Unir products con aisles y departments
df_products_full = df_products.merge(df_aisles, on='aisle_id', how='left')
df_products_full = df_products_full.merge(df_departments, on='department_id', how='left')

# 4. PERFILES DE USUARIO (historial prior)
print("\n[3/8] Construyendo perfiles de usuario...")
df_historial = df_orders[df_orders['eval_set'] == 'prior'].copy()

# Estadisticas temporales
perfil_temporal = df_historial.groupby('user_id').agg(
    promedio_dias_espera=('days_since_prior_order', 'mean'),
    std_dias_espera=('days_since_prior_order', 'std'),
    total_pedidos_historicos=('order_number', 'max'),
    total_dias_historia=('days_since_prior_order', 'sum'),
    ultimo_order_number=('order_number', 'max')
).reset_index()
perfil_temporal['std_dias_espera'] = perfil_temporal['std_dias_espera'].fillna(0)

# Tendencia: ultimo pedido vs promedio
ultimo_pedido = df_historial.groupby('user_id').agg(
    ultimo_dias=('days_since_prior_order', 'last')
).reset_index()
perfil_temporal = perfil_temporal.merge(ultimo_pedido, on='user_id', how='left')
perfil_temporal['tendencia_dias'] = (
    perfil_temporal['ultimo_dias'] - perfil_temporal['promedio_dias_espera']
).fillna(0)

# NUEVA: Posicion normalizada en la vida del usuario (0-1)
max_order_global = df_orders['order_number'].max()
perfil_temporal['order_number_norm'] = perfil_temporal['ultimo_order_number'] / max_order_global

print(f"  Perfiles temporales: {len(perfil_temporal):,} usuarios")

# 5. FEATURES DEL CARRITO (historial)
print("\n[4/8] Calculando features del carrito...")
df_prior_con = df_prior.merge(
    df_products_full[['product_id', 'department_id', 'aisle_id', 'department', 'aisle']],
    on='product_id', how='left'
)

# Stats por orden
carrito_por_orden = df_prior_con.groupby('order_id').agg(
    total_productos=('product_id', 'count'),
    productos_recomprados=('reordered', 'sum'),
    num_departamentos=('department_id', 'nunique'),
    num_aisles=('aisle_id', 'nunique')
).reset_index()
carrito_por_orden['ratio_recompra'] = carrito_por_orden['productos_recomprados'] / carrito_por_orden['total_productos']

# Merge con user_id
orders_hist_user = df_historial[['order_id', 'user_id']].copy()
carrito_user = orders_hist_user.merge(carrito_por_orden, on='order_id', how='inner')

# Perfil de carrito por usuario
perfil_carrito = carrito_user.groupby('user_id').agg(
    promedio_productos=('total_productos', 'mean'),
    std_carrito=('total_productos', 'std'),
    min_carrito=('total_productos', 'min'),
    max_carrito=('total_productos', 'max'),
    promedio_recompra_usuario=('ratio_recompra', 'mean'),
    promedio_dept_usuario=('num_departamentos', 'mean'),
    n_departamentos_historial=('num_departamentos', 'max'),
    n_aisles_historial=('num_aisles', 'max')
).reset_index()
perfil_carrito['std_carrito'] = perfil_carrito['std_carrito'].fillna(0)

print(f"  Perfiles de carrito: {len(perfil_carrito):,} usuarios")

# 6. NUEVAS: Features globales por departamento y aisle
print("\n[5/8] Calculando features globales por departamento/aisle...")

# Reorder ratio global por departamento (de todo el historial)
dept_global = df_prior_con.groupby('department_id').agg(
    global_dept_reorder=('reordered', 'mean'),
    global_dept_count=('product_id', 'count')
).reset_index()

# Reorder ratio global por aisle
aisle_global = df_prior_con.groupby('aisle_id').agg(
    global_aisle_reorder=('reordered', 'mean'),
    global_aisle_count=('product_id', 'count')
).reset_index()

# NUEVAS: Features globales por PRODUCTO
print("  Features por producto...")
producto_global = df_prior_con.groupby('product_id').agg(
    product_purchase_count=('order_id', 'count'),
    product_reorder_rate=('reordered', 'mean')
).reset_index()
# Popularidad como percentil
producto_global['product_popularity'] = producto_global['product_purchase_count'].rank(pct=True)
producto_global['product_reorder_rate'] = producto_global['product_reorder_rate'].fillna(0)

print(f"  Departamentos: {len(dept_global)} | Aisles: {len(aisle_global)} | Productos: {len(producto_global)}")

# 7. NUEVAS: Uniques products per user
print("\n[6/8] Calculando products unicos por usuario...")
n_products_user = df_prior_con.merge(
    df_historial[['order_id', 'user_id']], on='order_id', how='left'
).groupby('user_id')['product_id'].nunique().reset_index(name='n_products_historial')

# Unir todos los perfiles
perfil_usuario = perfil_temporal.merge(perfil_carrito, on='user_id', how='left')
perfil_usuario = perfil_usuario.merge(n_products_user, on='user_id', how='left')
perfil_usuario = perfil_usuario.fillna(0)

# 8. DATASET FINAL (train orders)
print("\n[7/8] Construyendo dataset final...")

df_train_orders = df_orders[df_orders['eval_set'] == 'train'][[
    'order_id', 'user_id', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order'
]].copy()

# Carrito del train con products info
df_train_con = df_train_products.merge(
    df_products_full[['product_id', 'department_id', 'aisle_id', 'department', 'aisle']],
    on='product_id', how='left'
)

# NUEVAS: Agregar features de productos al carrito train
df_train_con = df_train_con.merge(
    producto_global[['product_id', 'product_popularity', 'product_reorder_rate']],
    on='product_id', how='left'
)
df_train_con['product_popularity'] = df_train_con['product_popularity'].fillna(0.5)
df_train_con['product_reorder_rate'] = df_train_con['product_reorder_rate'].fillna(0.5)

# Stats carrito train (basicas + productos)
carrito_train = df_train_con.groupby('order_id').agg(
    total_productos=('product_id', 'count'),
    productos_recomprados=('reordered', 'sum'),
    num_departamentos=('department_id', 'nunique'),
    num_aisles=('aisle_id', 'nunique'),
    avg_product_popularity=('product_popularity', 'mean'),
    avg_product_reorder_rate=('product_reorder_rate', 'mean'),
    max_product_popularity=('product_popularity', 'max'),
).reset_index()
carrito_train['ratio_recompra'] = carrito_train['productos_recomprados'] / carrito_train['total_productos']

# Merge train + carrito + perfil
df_final = df_train_orders.merge(carrito_train, on='order_id', how='inner')
df_final = df_final.merge(perfil_usuario, on='user_id', how='left')

# FEATURES DERIVADAS DEL PEDIDO ACTUAL
print("\n[8/8] Creando features derivadas...")

# Ciclicas
df_final['hour_sin'] = np.sin(2 * np.pi * df_final['order_hour_of_day'] / 24)
df_final['hour_cos'] = np.cos(2 * np.pi * df_final['order_hour_of_day'] / 24)
df_final['dow_sin'] = np.sin(2 * np.pi * df_final['order_dow'] / 7)
df_final['dow_cos'] = np.cos(2 * np.pi * df_final['order_dow'] / 7)

# Binarias
df_final['es_finde'] = df_final['order_dow'].isin([5, 6]).astype(int)
df_final['es_horario_laboral'] = df_final['order_hour_of_day'].apply(lambda h: 1 if 9 <= h <= 17 else 0)
df_final['es_noche'] = df_final['order_hour_of_day'].apply(lambda h: 1 if h >= 19 or h <= 5 else 0)

# NUEVAS: Ratio tamano carrito vs promedio usuario
df_final['cart_vs_avg'] = df_final['total_productos'] / (df_final['promedio_productos'] + 1)

# NUEVA: Posicion normalizada del pedido en la vida del usuario
df_final['order_number_norm'] = df_final['order_number'] / max_order_global

# NUEVA: Primera compra
df_final['es_primera_compra'] = (df_final['order_number'] == 1).astype(int)

# NUEVAS: Features de interaccion (carrito x usuario)
df_final['cart_x_recompra'] = df_final['total_productos'] * df_final['ratio_recompra']
df_final['cart_x_dept'] = df_final['total_productos'] * df_final['promedio_dept_usuario']
df_final['std_x_tendencia'] = df_final['std_dias_espera'] * df_final['tendencia_dias']
df_final['recompra_x_std'] = df_final['promedio_recompra_usuario'] * df_final['std_carrito']

# NUEVAS: Posicion relativa en la vida del usuario
df_final['lifecycle_pos'] = df_final['order_number'] / (df_final['total_pedidos_historicos'] + 1)
df_final['es_pedido_reciente'] = (df_final['order_number'] >= df_final['total_pedidos_historicos'] - 2).astype(int)

# NUEVAS: Densidad de compra (productos unicos / total historico)
df_final['densidad_compra'] = df_final['n_products_historial'] / (df_final['total_pedidos_historicos'] + 1)

# Rellenar nulos
df_final = df_final.fillna(0)

# EXPORTAR
print("\nExportando dataset final...")
columnas_finales = [
    # Target
    'days_since_prior_order',
    # Temporal del pedido actual
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'es_finde', 'es_horario_laboral', 'es_noche',
    # Carrito actual
    'total_productos', 'productos_recomprados', 'ratio_recompra',
    # NUEVAS: Productos en el carrito
    'avg_product_popularity', 'avg_product_reorder_rate', 'max_product_popularity',
    # Carrito derivado
    'cart_vs_avg', 'order_number_norm', 'es_primera_compra',
    # Interacciones
    'cart_x_recompra', 'cart_x_dept', 'std_x_tendencia', 'recompra_x_std',
    # Lifecycle
    'lifecycle_pos', 'es_pedido_reciente', 'densidad_compra',
    # Perfil usuario
    'promedio_dias_espera', 'std_dias_espera', 'total_pedidos_historicos',
    'tendencia_dias',
    'promedio_productos', 'std_carrito', 'promedio_recompra_usuario',
    'promedio_dept_usuario', 'n_departamentos_historial', 'n_aisles_historial',
    'n_products_historial',
    # Referencia
    'order_id', 'user_id', 'order_number'
]

# Evitar duplicados
columnas_finales = list(dict.fromkeys(columnas_finales))

df_export = df_final[columnas_finales].copy()
ruta_salida = ruta_base + "dataset_mejorado.csv"
df_export.to_csv(ruta_salida, index=False)

print(f"\n{'=' * 60}")
print(f"  DATASET V4 GENERADO")
print(f"{'=' * 60}")
print(f"  Registros: {len(df_export):,}")
print(f"  Columnas:  {len(columnas_finales)}")
print()
print(f"  FEATURES FINALES: {len([c for c in columnas_finales if c not in ['order_id', 'user_id', 'order_number', 'days_since_prior_order']])}")
print()
print(f"  NUEVAS vs V3:")
print(f"    + cart_vs_avg (ratio carrito vs promedio usuario)")
print(f"    + order_number_norm (posicion en vida del usuario)")
print(f"    + es_primera_compra (primer pedido)")
print(f"    + n_products_historial (productos unicos)")
print()
print(f"  Columnas: {columnas_finales}")
