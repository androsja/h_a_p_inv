import os
import glob
import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

# 1. Recolectar datos
from shared.config import ML_DATASET_FILE

if not ML_DATASET_FILE.exists():
    print("Aún no hay suficientes trades (0 registros) para entrenar una IA.")
    exit()

df = pd.read_csv(ML_DATASET_FILE)
df['target'] = df['is_win']

if len(df) < 20:
    print(f"\n⏳ Se han detectado {len(df)} trades, pero el Machine Learning requiere al menos 20 para encontrar patrones confiables.")
    
# Mostrar el número de trades registrados
wins = len(df[df['target'] == 1])
losses = len(df[df['target'] == 0])
winrate = (wins/len(df))*100
print(f"📊 DATASET CREADO: {len(df)} Trades Históricos capturados en secreto.")
print(f"   Ganadores: {wins} | Perdedores: {losses} (Acuracidad base simulador: {winrate:.1f}%)")

if len(df) < 20:
    print(f"\n⏳ Se han detectado {len(df)} trades, pero el Machine Learning requiere al menos 20 para encontrar patrones confiables.")
    print("Por favor, deja que el bot simule unos activos más y vuelve a intentarlo.")
    exit()

# Entrenar un Decision Tree (Árbol de Decisión) rápido para sacar deducciones lógicas ("Insights Humanos")
features_cols = [c for c in df.columns if c not in ['symbol', 'pnl', 'target']]
X = df[features_cols]
y = df['target']

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

print("\n🧠 REVELACIÓN DE LA IA (Reglas Descubiertas por el Árbol de Decisión):")
# Guardar el modelo RandomForest robusto para usarlo en tiempo real
import joblib
import os

# Entrenamos el Random Forest para guardarlo como el motor real (Mejor que un solo arbol)
rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, class_weight='balanced')
rf_model.fit(X, y)

os.makedirs('data', exist_ok=True)
joblib.dump(rf_model, 'data/ai_model.joblib')
print("\n✅ MODELO RANDOM FOREST GUARDADO EN: data/ai_model.joblib (Activo para Live Trading y Simulación)")

# Mostrar la importancia de cada indicador (Feature Importance) del Tree Random Forest
print("\n📈 PESO DE CADA INDICADOR (¿Cuál es el culpable de hacerte perder o ganar?):")
importances = pd.Series(rf_model.feature_importances_, index=features_cols).sort_values(ascending=False)
for feature, imp in importances.items():
    print(f"   - {feature.upper():13}: {imp*100:.1f}% de importancia.")
