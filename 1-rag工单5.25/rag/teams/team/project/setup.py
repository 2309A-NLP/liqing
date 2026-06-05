"""setup.py — for pip install -e ."""
from setuptools import setup

setup(
    name="rag-question-answering",
    version="1.0.0",
    packages=["src", "src.api", "src.loader", "src.chunker",
              "src.embedder", "src.store", "src.retriever",
              "src.memory", "src.generator"],
    package_dir={"": "."},
    python_requires=">=3.10",
)
