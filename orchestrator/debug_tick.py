import json
from pathlib import Path

mastery_json = json.loads(Path('/app/data/mastery_hub.json').read_text())
assets = json.loads(Path('/app/assets.json').read_text()).get('assets', [])
symbols = [a['symbol'] for a in assets]

def get_symbol_status(sym):
    return mastery_json.get(sym, {})

def sort_priority(item):
    if item['status'] in ['ELITE', 'READY_FOR_LIVE']: return 100 + item['rank']
    if item['status'] == 'LEARNING': return 50 + item['rank']
    return item['rank']

active_symbols = {'LRCX', 'WFC', 'BX', 'QCOM', 'NOW', 'BK', 'O', 'UNH', 'TXN', 'RTX'}
symbol_ranks = []
for sym in symbols:
    if sym in active_symbols: continue
    d = get_symbol_status(sym)
    symbol_ranks.append({'symbol': sym, 'rank': d.get('rank', 0), 'status': d.get('status', 'UNKNOWN')})

symbol_ranks.sort(key=sort_priority, reverse=True)

print("Next 10 to launch:")
for x in symbol_ranks[:10]:
    d = get_symbol_status(x['symbol'])
    pvc_stage = d.get('pvc_stage', 'SCOUTING')
    pri = sort_priority(x)
    print(f"  {x['symbol']:8} pri={pri:6.1f}  pvc={pvc_stage:12}  status={x['status']}")
