import os

# Streamlit App Entry Point
if __name__ == "__main__":
    dashboard_path = os.path.join(os.path.dirname(__file__), "8_Python_Ninjas_5_Dashboard.py")
    with open(dashboard_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, dashboard_path, "exec"))
