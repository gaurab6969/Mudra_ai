import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
# import joblib
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define categories
Categories = ['alopodmo', 'ankush', 'ardhachandra', 'bhramar', 'chatur', 'ghronik', 'hongshashyo', 'kangul', 'khotkamukh', 'kodombo', 'kopitho', 
              'kortorimukho', 'krishnaxarmukh', 'mrigoshirsho', 'mukul']
datadir = 'IMAGES/'

# Define image dataset class
class MudraDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load and preprocess images
data, labels = [], []
transform = transforms.Compose([
    transforms.ToTensor()
])

for i, category in enumerate(Categories):
    path = os.path.join(datadir, category)
    print(f'Loading... category: {category}')
    for img_name in os.listdir(path):
        img_array = imread(os.path.join(path, img_name))
        img_resized = resize(img_array, (150, 150, 3))
        data.append(img_resized)
        labels.append(i)
    print(f'Loaded category: {category} successfully')

data = np.array(data, dtype=np.float32)
labels = np.array(labels)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

# Convert data to PyTorch tensors
train_dataset = MudraDataset(x_train, y_train, transform=transform)
test_dataset = MudraDataset(x_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define SVM model using PyTorch
class SVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SVM, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten
        return self.fc(x)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 150 * 150 * 3
num_classes = len(Categories)
model = SVM(input_dim, num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 15
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Evaluate model
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print(f"The model is {accuracy*100:.2f}% accurate")
print(classification_report(y_true, y_pred, target_names=Categories))
print('Confusion Matrix:\n', conf_matrix)
# Save the trained model
torch.save(model.state_dict(), 'mudra_model_4.pth')

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=Categories, yticklabels=Categories)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
