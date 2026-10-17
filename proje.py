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


##########################################################################

from torch.utils.data import DataLoader, random_split

# 3. Veriyi %80 Eğitim ve %20 Doğrulama olarak ayırma
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 4. DataLoader'ları oluşturma (32'şerli paketler)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Eğitim Seti Boyutu: {len(train_dataset)}")
print(f"Doğrulama Seti Boyutu: {len(val_dataset)}")

###################################################################

import torchvision.models as models
import torch.nn as nn

# 1. Önceden eğitilmiş ResNet18 modelini çağırıyoruz
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 2. ResNet18'in son katmanındaki çıkış sayısını bizim sınıf sayımıza (10) göre güncelliyoruz
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))

print("ResNet18 modeli başarıyla yüklendi ve 10 sınıfa uyarlandı!")