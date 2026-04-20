import json
from datetime import datetime

try:
    with open('data/completed_simulations.json') as f:
        data = json.load(f)
    
    durations = []
    trades = []
    for b in data:
        try:
            start = datetime.strptime(b['launched_at'], '%Y-%m-%d %H:%M')
            end = datetime.strptime(b['finished_at'], '%Y-%m-%d %H:%M')
            durations.append((end - start).total_seconds() / 60)
            trades.append(b['trades'])
        except:
            continue
            
    if durations:
        avg_dur = sum(durations)/len(durations)
        avg_trades = sum(trades)/len(trades)
        print(f"AVG_DURATION_MINS={avg_dur:.2f}")
        print(f"AVG_TRADES_PER_SIM={avg_trades:.2f}")
        print(f"TOTAL_SIMS={len(data)}")
    else:
        print("NO_DATA")
except Exception as e:
    print(f"ERROR: {e}")
