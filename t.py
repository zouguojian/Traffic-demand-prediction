# -- coding: utf-8 --
import pyarrow as pa
import pyarrow.parquet as pq
data = pq.read_table('yellow_tripdata_2022-01.parquet').to_pandas()

print(data.keys())
print(data.shape)
print(data.values[:12])
