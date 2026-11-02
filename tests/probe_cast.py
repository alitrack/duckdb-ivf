import duckdb, numpy as np
con = duckdb.connect()
# does DOUBLE[]::FLOAT[] actually convert payload?
r = con.execute("SELECT [1.0, 2.0, 3.0]::DOUBLE[]::FLOAT[] AS x").fetchall()
print("type after cast:", con.execute("SELECT typeof([1.0,2.0,3.0]::DOUBLE[]::FLOAT[])").fetchall())
x = r[0][0]
print("python sees:", [ (type(v).__name__, v) for v in x])
# raw check: read back as numpy and compare to expected
arr = np.array(con.execute("SELECT [0.5, -1.5, 2.25]::DOUBLE[]::FLOAT[]").fetchall()[0][0], dtype=np.float64)
print("expected [0.5,-1.5,2.25] got:", arr)
