"""Test embedder standalone - no chunker import"""
import sys, os, traceback
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.dirname(__file__))

try:
    print("importing embedder...", flush=True)
    from src.embedder.embed import Embedder
    print("creating...", flush=True)
    e = Embedder()
    print(f"model_path: {e.model_path}", flush=True)
    print("loading model...", flush=True)
    v = e.embed("test")
    print(f"dim={len(v)}", flush=True)
    print("DONE", flush=True)
except Exception as ex:
    print(f"ERROR: {ex}", flush=True)
    traceback.print_exc()
