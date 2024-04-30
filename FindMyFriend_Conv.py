import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths to the training and test data
train_data_path = "C:/Users/HP/Downloads/FindMyFriend/training"
test_data_path = "C:/Users/HP/Downloads/FindMyFriend/testing"

# Load the datasets
training_data = datasets.ImageFolder(root=train_data_path, transform=transform)
test_data = datasets.ImageFolder(root=test_data_path, transform=transform)

# Define the categories
categories = ['Abby', 'Dylan', 'Erica', 'Izzy', 'Kacey']

# Define the Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 128 * 128, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 128 * 128)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the CNN model
model = CNN()

# Define the data loaders
train_dataloader = DataLoader(training_data, batch_size=5)
test_dataloader = DataLoader(test_data, batch_size=5)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Define functions for training and testing loops
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    train_loss=[]
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return train_loss

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    accuracy = correct * 100
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy, test_loss

# Training loop
train_accuracy_val = []
test_accuracy_val = []
test_loss_val = []
train_loss_val=[]
epochs = 50

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    train_accuracy, train_loss = test_loop(train_dataloader, model, loss_fn)
    test_accuracy, test_loss = test_loop(test_dataloader, model, loss_fn)
    train_accuracy_val.append(train_accuracy)
    test_accuracy_val.append(test_accuracy)
    test_loss_val.append(test_loss)
    train_loss_val.append(train_loss)
print("Training done!")

# After training, you can evaluate a sample image
sample_num = 1
eval_dataloader = DataLoader(training_data, batch_size=1)

for sample in eval_dataloader:
    inputs, _ = sample
    with torch.no_grad():
        r = model(inputs)
        break  # Break after processing the first sample

print('Neural network output pseudo-probabilities:', r)
predicted_class = categories[torch.argmax(r).item()]
print('Predicted class:', predicted_class)

ima = training_data[sample_num][0]
print('Inputs sample - min,max,mean,std:', ima.min().item(), ima.max().item(), ima.mean().item(), ima.std().item())
ima = (ima - ima.mean())/ ima.std()
print('Inputs sample normalized - min,max,mean,std:', ima.min().item(), ima.max().item(), ima.mean().item(), ima.std().item())
iman = ima.permute(1, 2, 0) # needed to be able to plot
plt.imshow(iman)

# plt.figure(figsize=(10, 5))
# plt.plot(range(1, epochs + 1), train_accuracy_val, label='Train Accuracy')
# plt.plot(range(1, epochs + 1), test_accuracy_val, label='Test Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.show()

#Plots
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), test_loss_val, label='Test Loss')
plt.plot(range(1,epochs + 1), train_loss_val, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
