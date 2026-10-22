import os
import sys
import tempfile

# Tests never touch the real research database / checkpoints in runtime/.
os.environ["TWINSIM_HOME"] = tempfile.mkdtemp(prefix="twinsim_test_")
os.environ.setdefault("TWINSIM_BACKEND", "numpy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
