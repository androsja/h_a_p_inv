from datetime import datetime

def parse_dt(dt_str):
    if not dt_str: return None
    # Tomar solo los primeros 10 caracteres (YYYY-MM-DD) y normalizar
    clean_date = str(dt_str)[:10].replace("/", "-")
    try:
        return datetime.strptime(clean_date, "%Y-%m-%d")
    except Exception as e:
        print(f"Error parsing {dt_str}: {e}")
        return None

sim_start_date = "2024-01-01"
# What if sim_end_date was empty?
sim_end_date = "2025-12-31" 
current_ts = "2024-05-29 09:30:00"

start = parse_dt(sim_start_date)
end = parse_dt(sim_end_date) or datetime.now()
curr = parse_dt(current_ts)

print(f"Start: {start}")
print(f"End: {end}")
print(f"Curr: {curr}")

if not start or not end or not curr:
    print("One or more dates are None!")
else:
    total_delta = (end - start).total_seconds()
    curr_delta = (curr - start).total_seconds()
    
    print(f"Total Delta: {total_delta}")
    print(f"Curr Delta: {curr_delta}")
    
    if total_delta <= 0:
        print("Total Delta <= 0! Returning 100.0")
    else:
        pct = (curr_delta / total_delta) * 100.0
        print(f"PCT: {pct}")
