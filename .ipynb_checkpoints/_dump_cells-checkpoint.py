import json, sys
sys.stdout.reconfigure(encoding="utf-8")
nb = json.load(open("Fit_Ataei.ipynb", encoding="utf-8"))
for i in [2, 16, 17, 19, 20, 39, 40, 42, 43, 55]:
    print(f"=== CELL {i} ===")
    print("".join(nb["cells"][i]["source"]))
    print()
