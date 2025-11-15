import duckdb

conn = duckdb.connect("PATH_A_TU_ARCHIVO.duckdb")
print(conn.execute("SHOW TABLES").fetchdf())
conn.close()