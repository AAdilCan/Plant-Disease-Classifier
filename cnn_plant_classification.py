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
            except Exception as ex:
                print(f"Error processing {img_path}: {ex}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    images_tensor = torch.tensor(images, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    images_tensor = images_tensor.to(device)
    labels_tensor = labels_tensor.to(device)
    return images_tensor, labels_tensor

train_dir = "/content/drive/MyDrive/Plant_Dataset/Train"
validation_dir = "/content/drive/MyDrive/Plant_Dataset/Validation"

data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class_names = os.listdir(train_dir)
class_to_label = {class_name: idx for idx, class_name in enumerate(class_names)}
train_images_tensor, train_labels_tensor = process_dataset(train_dir, data_transforms, class_to_label )
val_images_tensor, val_labels_tensor = process_dataset(validation_dir, data_transforms, class_to_label)

train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

validation_dataset = TensorDataset(val_images_tensor, val_labels_tensor)
validation_loader = DataLoader(validation_dataset, batch_size=60, shuffle=False)

class NeuralNet(nn.Module):

  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 12, 5)
    self.conv2 = nn.Conv2d(12, 18, 3)
    self.conv3 = nn.Conv2d(18, 24, 3)

    self.pool = nn.MaxPool2d(2,2)

    self.fc1 = nn.Linear(24 * 14 * 14, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 3)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))

    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x

class NeuralNet_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.conv2 = nn.Conv2d(12, 24, 3)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(24 * 30 * 30, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def train(model, train_loader, optimizer, loss_function):
    correct = 0
    total = 0
    running_loss = 0

    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Training Loss: {running_loss / len(train_loader): .4f}, Accuracy: {accuracy:.2f}%")
    return running_loss / len(train_loader), accuracy

def validate(model, val_loader, loss_function):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation Loss: {running_loss / len(val_loader): .4f}, Accuracy: {accuracy:.2f}%')
    return running_loss / len(val_loader), accuracy

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def test(model, test_loader, best_weight, device):
    model.eval()
    model.load_state_dict(torch.load(best_weight))
    model.to(device)

    running_loss = 0
    total = 0
    correct = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            running_loss += loss.item()
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    num_batches = len(test_loader)
    avg_loss = running_loss / num_batches
    accuracy = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_loader.dataset.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    print("\nClassification Report:")
    report = classification_report(all_labels, all_predictions, target_names=test_loader.dataset.classes)
    print(report)

    return avg_loss, accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data = torchvision.datasets.ImageFolder(root='/content/drive/MyDrive/Plant_Dataset/Test', transform=data_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=150, shuffle=False, num_workers = 2)

net = NeuralNet().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr= 0.03125, momentum = 0.9, weight_decay=0.0025)

num_epoch = 25
min_loss = 9999

val_loss_list = []
val_acc_list = []
train_loss_list = []
train_acc_list = []
for epoch in range(num_epoch):
    print(f"Epoch {epoch+1}...")

    train_loss, train_acc = train(net, train_loader, optimizer, loss_function)

    validation_loss, val_acc = validate(net, validation_loader, loss_function)

    val_loss_list.append(validation_loss)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)

    if validation_loss < min_loss:
        min_loss = validation_loss
        best_epoch = str(epoch + 1)
        model_weight = net.state_dict()

print(f"Best Epoch is {best_epoch}, Validation Loss: {min_loss:.4f}")
torch.save(model_weight, '/content/drive/MyDrive/Plant_Dataset/best_weight_new.pth')

import matplotlib.pyplot as plt

def plot_training_history(train_loss_list, val_loss_list, train_acc_list, val_acc_list):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Training Loss', color='blue')
    axs[0].plot(range(1, len(val_loss_list) + 1), val_loss_list, label='Validation Loss', color='red')
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(range(1, len(train_acc_list) + 1), train_acc_list, label='Training Accuracy', color='blue')
    axs[1].plot(range(1, len(val_acc_list) + 1), val_acc_list, label='Validation Accuracy', color='red')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

plot_training_history(train_loss_list, val_loss_list, train_acc_list, val_acc_list)

test(net, test_loader, '/content/drive/MyDrive/Plant_Dataset/best_weight_new.pth', device)