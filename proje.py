import os
import matplotlib.pyplot as plt

# Veri setimizin bulunduğu ana dizin
data_dir = "data/plantvillage_tomato"

# Sınıfları (hastalık türlerini) listeleyelim
classes = os.listdir(data_dir)
print(f"Toplam Sınıf Sayısı: {len(classes)}")
print(f"Sınıflar: {classes}")

# Her sınıftaki görsel sayısını kontrol edelim
class_counts = {}
for c in classes:
    class_path = os.path.join(data_dir, c)
    if os.path.isdir(class_path):
        class_counts[c] = len(os.listdir(class_path))

# Sonuçları yazdıralım
for c, count in class_counts.items():
    print(f"{c}: {count} görsel")