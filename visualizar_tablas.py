import duckdb
from src.config import *

conn = duckdb.connect(PATH_DATA_BASE_DB)
print(conn.execute("SHOW TABLES").fetchdf())
conn.close()