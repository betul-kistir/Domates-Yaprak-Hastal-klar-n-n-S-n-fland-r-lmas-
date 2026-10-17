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

#####################################################

import torch
from torchvision import datasets, transforms

# 1. Temel dönüşümleri tanımlayalım (Görseli 224x224 boyutuna getirip Tensöre çeviriyoruz)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 2. Veri setini yasal dizinden yükleyelim
data_dir = "data/plantvillage_tomato"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

print(f"Toplam görsel sayısı: {len(dataset)}")
print(f"Sınıf isimleri: {dataset.classes}")

# Veri setini yasal dizinden yükleyelim
data_dir = "data/plantvillage_tomato"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

print(f"Toplam görsel sayısı: {len(dataset)}")
print(f"Sınıf isimleri: {dataset.classes}")