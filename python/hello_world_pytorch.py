import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. Cihaz (Device) Ayarı - Kurduğumuz GPU'yu aktif edelim!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eğitim için kullanılan cihaz: {device}\n")

# 2. Veri Setinin Hazırlanması (MNIST)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Eğitim ve Test verilerini indir/yükle
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. Modelin Tanımlanması (Basit bir CNN - Evrişimli Sinir Ağı)
class BasitCNN(nn.Module):
    def __init__(self):
        super(BasitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7) # Flatten işlemi
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = BasitCNN().to(device) # Modeli GPU'ya gönderiyoruz

# 4. Kayıp Fonksiyonu (Loss) ve Optimizasyon
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Modelin Eğitimi (Training Loop)
epochs = 3
kayip_gecmisi = [] # Grafik için kayıpları tutacağız

print("Eğitim başlıyor...")
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad() # Gradientleri sıfırla

        outputs = model(inputs) # İleri besleme (Forward)
        loss = criterion(outputs, labels) # Kaybı hesapla
        loss.backward() # Geri yayılım (Backward)
        optimizer.step() # Ağırlıkları güncelle

        running_loss += loss.item()
        
    epoch_loss = running_loss / len(train_loader)
    kayip_gecmisi.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Ortalama Kayıp: {epoch_loss:.4f}")

print("Eğitim tamamlandı!\n")

# 6. Test Aşaması
doğru = 0
toplam = 0
model.eval() # Modeli test moduna al

with torch.no_grad(): # Test sırasında gradient hesaplamaya gerek yok
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, tahminler = torch.max(outputs.data, 1)
        toplam += labels.size(0)
        doğru += (tahminler == labels).sum().item()

print(f"10.000 Test Görseli Üzerinde Model Başarısı (Accuracy): %{100 * doğru / toplam:.2f}")

# 7. Görselleştirme (Matplotlib)
plt.figure(figsize=(12, 5))

# Sol Grafik: Eğitim Kaybı (Loss Curve)
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), kayip_gecmisi, marker='o', color='b')
plt.title('Eğitim Kaybı (Training Loss)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, epochs + 1))
plt.grid(True)

# Sağ Grafik: Tahmin Örnekleri
plt.subplot(1, 2, 2)
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Görselleri ve tahminleri CPU'ya geri çekiyoruz
images_cpu = images.to('cpu')
labels_cpu = labels.to('cpu')
outputs = model(images.to(device))
_, preds = torch.max(outputs, 1)
preds_cpu = preds.to('cpu')

# İlk 6 görseli ekrana basalım
for i in range(6):
    ax = plt.subplot(2, 3, i + 1)
    # Tensor'u numpy array'e çevirip boyutunu ayarlıyoruz
    img = images_cpu[i] / 2 + 0.5 # Unnormalize işlemi
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)).squeeze(), cmap='gray')
    
    # Gerçek ve Tahmin edilen etiketleri başlık olarak yazalım
    color = 'green' if preds_cpu[i] == labels_cpu[i] else 'red'
    ax.set_title(f"Tahmin: {preds_cpu[i].item()}\nGerçek: {labels_cpu[i].item()}", color=color)
    ax.axis('off')

plt.tight_layout()
plt.show()