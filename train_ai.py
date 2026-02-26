import os
import glob
import ast
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

# 1. Recolectar datos
data = []
for log_file in glob.glob('/app/logs/historial_*.log'):
    with open(log_file, 'r') as f:
        for line in f:
            if 'SAVING TRADE:' in line:
                try:
                    # Extraer partes de la línea
                    # Formato: ... SAVING TRADE: MSFT Pnl=-10.5 Features={'rsi': 50.1, ...}
                    parts = line.split('SAVING TRADE: ')[1]
                    symbol = parts.split(' Pnl=')[0]
                    pnl_str = parts.split(' Pnl=')[1].split(' Features=')[0]
                    features_str = parts.split(' Features=')[1].strip()
                    
                    pnl = float(pnl_str)
                    features = ast.literal_eval(features_str)
                    
                    row = features.copy()
                    row['symbol'] = symbol
                    row['pnl'] = pnl
                    row['target'] = 1 if pnl > 0 else 0  # 1 si ganó (win/TP), 0 si perdió (StopLoss/Whipsaw)
                    
                    data.append(row)
                except Exception as e:
                    pass

df = pd.DataFrame(data)

if df.empty:
    print("Aún no hay suficientes trades (0 registros) para entrenar una IA.")
    exit()
    
# Mostrar el número de trades registrados
wins = len(df[df['target'] == 1])
losses = len(df[df['target'] == 0])
winrate = (wins/len(df))*100
print(f"📊 DATASET CREADO: {len(df)} Trades Históricos capturados en secreto.")
print(f"   Ganadores: {wins} | Perdedores: {losses} (Acuracidad base simulador: {winrate:.1f}%)")

if len(df) < 20:
    print("\n⏳ (Se requieren al menos 20 trades para que el Machine Learning detecte patrones. Dejaremos que termine el ciclo actual).")
    exit()

# Entrenar un Decision Tree (Árbol de Decisión) rápido para sacar deducciones lógicas ("Insights Humanos")
features_cols = [c for c in df.columns if c not in ['symbol', 'pnl', 'target']]
X = df[features_cols]
y = df['target']

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

print("\n🧠 REVELACIÓN DE LA IA (Reglas Descubiertas por el Árbol de Decisión):")
tree_rules = export_text(clf, feature_names=features_cols)
# Imprimir reglas limpias
print(tree_rules)

# Guardar el modelo entrenado para usarlo ráfido en tiempo real
import joblib
import os
os.makedirs('/app/data', exist_ok=True)
joblib.dump(clf, '/app/data/ai_model.joblib')
print("\n✅ MODELO GUARDADO EN: /app/data/ai_model.joblib (Listo para Inferencia Ultra-Rápida)")

# Mostrar la importancia de cada indicador (Feature Importance)
print("\n📈 PESO DE CADA INDICADOR (¿Cuál es el culpable de hacerte perder o ganar?):")
importances = pd.Series(clf.feature_importances_, index=features_cols).sort_values(ascending=False)
for feature, imp in importances.items():
    print(f"   - {feature.upper():13}: {imp*100:.1f}% de importancia.")
