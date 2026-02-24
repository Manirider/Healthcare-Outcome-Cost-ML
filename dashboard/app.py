import os
import sys
import importlib.util

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

spec = importlib.util.spec_from_file_location(
    "app", os.path.join(project_root, "app.py"))
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)

if hasattr(app_module, "main"):
    app_module.main()
else:
    st.error("Could not find main() in app.py")

