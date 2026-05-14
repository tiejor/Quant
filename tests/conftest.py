"""Shared fixtures for factor tests."""
import os
import sys
import pytest
import duckdb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "quant.duckdb")


def has_data():
    """Check if production DuckDB exists and has data."""
    if not os.path.exists(DB_PATH):
        return False
    try:
        db = duckdb.connect(DB_PATH, read_only=True)
        count = db.execute("SELECT COUNT(*) FROM daily_basic").fetchone()[0]
        db.close()
        return count > 0
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not has_data(),
    reason="需要 data/quant.duckdb 中有数据（先运行 python src/main.py --skip-pipeline 或 pipeline）",
)


@pytest.fixture
def tmp_db():
    """临时 DuckDB 文件，测试结束自动删除."""
    import tempfile
    os.makedirs("tests/data", exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".duckdb", dir="tests/data")
    os.close(fd)
    os.remove(path)  # DuckDB will create it fresh
    yield path
    if os.path.exists(path):
        os.remove(path)
    # Also clean up WAL file if present
    wal = path + ".wal"
    if os.path.exists(wal):
        os.remove(wal)


requires_tushare = pytest.mark.skipif(
    not os.getenv("TUSHARE_TOKEN"),
    reason="需要 TUSHARE_TOKEN 环境变量",
)
