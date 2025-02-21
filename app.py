import os 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# import numpy as np

# Define the same SVM model class
class SVM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SVM, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten
        return self.fc(x)

# Define categories
Categories = ['alopodmo', 'ankush', 'ardhachandra', 'bhramar', 'chatur', 
              'ghronik', 'hongshashyo', 'kangul', 'khotkamukh', 'kodombo', 'kopitho', 'kortorimukho', 'krishnaxarmukh', 'mrigoshirsho', 'mukul']

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 150 * 150 * 3
num_classes = len(Categories)

model = SVM(input_dim, num_classes).to(device)
model.load_state_dict(torch.load('mudra_model_4.pth', map_location=device))  # Load trained weights
model.eval()  # Set to evaluation mode

# Define image transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((150, 150)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization
])

# Function to predict Mudra hand sign
def predict_mudra(image_path):
    image = Image.open(image_path).convert('RGB')  # Open and convert to RGB
    image = transform(image).unsqueeze(0).to(device)  # Apply transforms and add batch dimension

    with torch.no_grad():
        output = model(image)  # Get model predictions
        probabilities = torch.nn.functional.softmax(output, dim=1)  # Convert to probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()  # Get highest probability class

    print(f"Predicted Mudra: {Categories[predicted_class]}")
    return Categories[predicted_class]

# Example Usage
image_path = 'C:/Users/debasish kurmi/f/IMAGES/ghronik/0192.jpg'  # Replace with actual image path
predicted_class = predict_mudra(image_path)
