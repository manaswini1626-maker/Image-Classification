import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(
    root=r'C:\Users\Manaswini\Downloads\archive (2)',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root=r'C:\Users\Manaswini\Downloads\archive (2)',
    train=False,
    download=True,
    transform=transform
)
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f'Train: {len(train_dataset)} images | Val: {len(test_dataset)} images')

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),   
            nn.ReLU(),
            nn.Linear(64, 10),   
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

NUM_EPOCHS = 10
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(NUM_EPOCHS):
    
    model.train()
    running_loss = 0.0
    train_correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        train_correct += (preds == labels).sum().item()

    train_loss = running_loss / len(train_dataset)
    train_acc = train_correct / len(train_dataset)

    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()

    val_loss /= len(test_dataset)
    val_acc = val_correct / len(test_dataset)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    print(f'Epoch {epoch+1}/{NUM_EPOCHS} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}')

import matplotlib.pyplot as plt
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

sample_image, sample_label = train_dataset[0]
sample_image = sample_image.unsqueeze(0).to(device)

first_conv_layer = model.features[0]

model.eval()
with torch.no_grad():
    feature_maps = first_conv_layer(sample_image)

print('Feature map shape:', feature_maps.shape)

num_filters = feature_maps.shape[1]
fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    if i < num_filters:
        fmap = feature_maps[0, i].cpu().numpy()
        ax.imshow(fmap, cmap='viridis')
        ax.set_title(f'Filter {i}')
    ax.axis('off')

plt.suptitle('Feature Maps from First Conv Layer', fontsize=14)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import torch

model.eval()
images, labels = next(iter(train_loader))
img = images[0].unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img)
    prediction = torch.argmax(output, dim=1).item()

img_display = images[0].permute(1, 2, 0).cpu().numpy() 
plt.imshow(img_display)  
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.title(f"Prediction: {classes[prediction]}")
plt.title(f"Prediction: {prediction}")
plt.axis('off')
plt.show()