from pyproj import Transformer
import pandas as pd
import os

# x,y from BAYSIS
points = [
    (669529.74, 5337096.61),
    (668285.48, 5338221.35),
    (669686.78, 5340308.06),
    (671275.7, 5341591.58),
    (665407.5, 5339805.27),
    (672460.59, 5353567.02),
    (673930.46, 5353084.41),
    (676969.22, 5353968.51),
    (669498.93, 5352692.92),
    (666260.94, 5341452.32),
    (668615.87, 5344082.72),
    (668399.3, 5349370.73),
    (669125.6, 5356025.35),
    (675241.02, 5357141.69),
    (682711.49, 5350863.73),
    (685979.35, 5352558.46),
    (682236.73, 5350583.67),
    (668464.59, 5350228.43),
    (664214.06, 5352596.05),
    (668864.92, 5342722.1),
    (678632.31, 5347379.96),
    (678524.71, 5349921.64),
]

# EPSG:25832 → WGS84
transformer = Transformer.from_crs(
    25832,
    4326,
    always_xy=True
)

results=[]

for x,y in points:
    lon,lat = transformer.transform(x,y)

    results.append({
        "x":x,
        "y":y,
        "lat":lat,
        "lon":lon
    })

df=pd.DataFrame(results)

print(df.round(6))

# 保存位置
save_dir = r"D:\SUMO_A9_Project\detectors\data from BAYSIS SVZ"

# 如果文件夹不存在自动创建
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(
    save_dir,
    "zst_wgs84.csv"
)

df.to_csv(save_path, index=False)

print(f"\nsaved -> {save_path}")