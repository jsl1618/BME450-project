# PyTorch train script
# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

# We will use the Custom Datase
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Normalize, Grayscale

transform=transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

train_data_path="C:/Users/HP/Downloads/FindMyFriend/training"
test_data_path="C:/Users/HP/Downloads/FindMyFriend/testing"
training_data=datasets.ImageFolder(root=train_data_path,transform=transform)
test_data=datasets.ImageFolder(root=test_data_path,transform=transform)


# let us print some data:

categories = ['Abby', 'Dylan', 'Erica', 'Izzy', 'Kacey']

# select a random sample from the training set
sample_num = 4
print(training_data[sample_num])
print('Inputs sample - image size:', training_data[sample_num][0].shape)
print('Label:', training_data[sample_num][1], '\n')

import matplotlib.pyplot as plt

ima = training_data[sample_num][0]
print('Inputs sample - min,max,mean,std:', ima.min().item(), ima.max().item(), ima.mean().item(), ima.std().item())
ima = (ima - ima.mean())/ ima.std()
print('Inputs sample normalized - min,max,mean,std:', ima.min().item(), ima.max().item(), ima.mean().item(), ima.std().item())
iman = ima.permute(1, 2, 0) # needed to be able to plot
plt.imshow(iman)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(512*512*3, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        output = self.l3(x)
        return output

# input_image_size=512
# num_channels=3
# expected_input_size = input_image_size * input_image_size * num_channels
# print("Expected input size:", expected_input_size)

# sample_input = training_data[sample_num][0]  # Get a sample input
# print(training_data[sample_num][0].shape)
# print("Input size before flattening:", sample_input.size())

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    train_loss=[]
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        #print("Batch size:", X.size(0))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"train loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            #print("Batch size:", X.size(0))
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    accuracy=100*correct
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy,test_loss

    # training!

model = Net()

train_dataloader = DataLoader(training_data, batch_size=5)
test_dataloader = DataLoader(test_data, batch_size=5)

learning_rate = 1e-3
batch_size = 5

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
train_accuracy_val=[]
test_accuracy_val=[]
test_loss_val=[]
train_loss_val=[]

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    train_accuracy,train_loss=test_loop(train_dataloader,model,loss_fn)
    test_accuracy,test_loss=test_loop(test_dataloader,model,loss_fn)
    train_accuracy_val.append(train_accuracy)
    test_accuracy_val.append(test_accuracy)
    test_loss_val.append(test_loss)
    train_loss_val.append(train_loss)
print("Done!")


sample_num = 1 # select a random sample

eval_dataloader = DataLoader(training_data, batch_size=1)

# Iterate over the DataLoader
for sample in eval_dataloader:
    inputs, _ = sample
    with torch.no_grad():
        r = model(inputs)
        break  # Break after processing the first sample

print('neural network output pseudo-probabilities:', r)
print('neural network output class number:', torch.argmax(r).item())
print('neural network output, predicted class:', categories[torch.argmax(r).item()])

ima = training_data[sample_num][0]
print('Inputs sample - min,max,mean,std:', ima.min().item(), ima.max().item(), ima.mean().item(), ima.std().item())
ima = (ima - ima.mean())/ ima.std()
print('Inputs sample normalized - min,max,mean,std:', ima.min().item(), ima.max().item(), ima.mean().item(), ima.std().item())
iman = ima.permute(1, 2, 0) # needed to be able to plot
plt.imshow(iman)

#Plots
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, epochs + 1), test_loss_val, label='Test Loss')
# plt.plot(range(1, epochs + 1), train_loss_val, label='Train Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_accuracy_val, label='Train Accuracy')
plt.plot(range(1, epochs + 1), test_accuracy_val, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()