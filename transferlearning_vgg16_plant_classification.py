import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def process_dataset(dataset_dir, data_transforms, class_to_label):
    images = []
    labels = []

    for class_name in tqdm(class_to_label.keys(), desc=f"Processing {os.path.basename(dataset_dir)}"):
        class_folder = os.path.join(dataset_dir, class_name)
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            try:
                with Image.open(img_path) as img:
                    img_tensor = data_transforms(img)
                    images.append(img_tensor.numpy())
                    labels.append(class_to_label[class_name])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    images_tensor = torch.tensor(images, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    images_tensor = images_tensor.to(device)
    labels_tensor = labels_tensor.to(device)
    return images_tensor, labels_tensor

train_dir = "/content/drive/MyDrive/Plant_Dataset/Train"
validation_dir = "/content/drive/MyDrive/Plant_Dataset/Validation"
test_dir = "/content/drive/MyDrive/Plant_Dataset/Test"
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class_names = os.listdir(train_dir)
class_to_label = {class_name: idx for idx, class_name in enumerate(class_names)}


train_images_tensor, train_labels_tensor = process_dataset(train_dir, data_transforms, class_to_label )
val_images_tensor, val_labels_tensor = process_dataset(validation_dir, data_transforms, class_to_label)
test_images_tensor, test_labels_tensor = process_dataset(test_dir, data_transforms, class_to_label)

train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

validation_dataset = TensorDataset(val_images_tensor, val_labels_tensor)
validation_loader = DataLoader(validation_dataset, batch_size=60, shuffle=False)

test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=150, shuffle=False)

model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).to(device)

for param in model.parameters():
    param.requires_grad = False

num_classes = 3
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.0005)

epochs = 25

best_model_path = "/content/best_model.pth"
best_val_loss = float('inf')
train_loss_list = []
val_loss_list = []
train_accuracy_list = []
val_accuracy_list = []
for epoch in range(epochs):
    model.train()
    loss_epoch = 0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        loss_epoch += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss_epoch/len(train_loader)}")
    train_loss_list.append(loss_epoch/len(train_loader))
    train_accuracy_list.append(accuracy)

    model.eval()
    val_loss_epoch = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss_epoch += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss_epoch / len(validation_loader)
    val_accuracy = 100 * correct / total

    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_accuracy}%")
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_accuracy)
    if val_loss < best_val_loss:
        best_eph = epoch + 1
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)

print(f"Best Val Loss {best_val_loss}")
print(f"Best Epoch is {best_eph}")

print("Loading the best model for testing...")
model.load_state_dict(torch.load(best_model_path))
model.to(device)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

all_preds = []
all_labels = []

model.eval()

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

conf_matrix = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
class_names = ["Healthy", "Powedery", "Rusty"]
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

precision, recall, f1_score, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
print("Confusion Matrix:")
print(conf_matrix)

print("\nPrecision per class:", precision)
print("Recall per class:", recall)
print("F1-Score per class:", f1_score)
print("Support (number of samples per class):", support)

report = classification_report(all_labels, all_preds, target_names=class_names)
print("\nClassification Report:")
print(report)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Training Loss')
plt.title('Training and Validation Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracy_list, label='Validation Loss')
plt.title('Training and Validation Accuracy')
plt.plot(val_accuracy_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.legend()
plt.ylabel('Loss')