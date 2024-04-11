# PyTorch train script
# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

# We will use the Custom Dataset

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Normalize, Grayscale
import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for cls_name in self.classes:
            class_dir = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                images.append((img_path, self.class_to_idx[cls_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations to be applied to the images
transform = transforms.Compose([
    transforms.Resize((50, 50)),  # Resize to a uniform size
    transforms.ToTensor(),        # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# transform=transforms.Compose([
#     transforms.Resize((50,50)),
#     Grayscale(num_output_channels=1),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,),(0.5,))
# ])

# Define your custom dataset
custom_dataset_train = CustomDataset(root_dir='C:/Users/HP/Downloads/FindMyFriend/Train', transform=transform)
custom_dataset_test=CustomDataset(root_dir='C:/Users/HP/Downloads/FindMyFriend/Test', transform=transform)

# Create dataloaders for training and testing
train_dataloader = DataLoader(custom_dataset_train, batch_size=27, shuffle=True)
test_dataloader = DataLoader(custom_dataset_test, batch_size=5, shuffle=False)
#
# training_data = datasets.CIFAR10(
#     root="data",
#     train=True,
#     download=True,
#     transform=transform
# )

# test_data = datasets.CIFAR10(
#     root="data",
#     train=False,
#     download=True,
#     transform=transform
# )

# let us print some data:

categories = ['Erica', 'Dylan', 'Abby', 'Kasey', 'Justine']

# select a random sample from the training set
sample_num = 143
print(train_dataloader[sample_num])
print('Inputs sample - image size:', train_dataloader[sample_num][0].shape)
print('Label:', train_dataloader[sample_num][1], '\n')

import matplotlib.pyplot as plt

ima = train_dataloader[sample_num][0]
print('Inputs sample - min,max,mean,std:', ima.min().item(), ima.max().item(), ima.mean().item(), ima.std().item())
ima = (ima - ima.mean())/ ima.std()
print('Inputs sample normalized - min,max,mean,std:', ima.min().item(), ima.max().item(), ima.mean().item(), ima.std().item())
iman = ima.permute(1, 2, 0) # needed to be able to plot
plt.imshow(iman)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(50*50*3, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 10)

    def forward(self, x):
        #print("input shape:", x.shape)
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        output = self.l3(x)
        #print("Shape of l1 weights:", self.l1.weight.shape)
        #print("Shape of l2 weights:", self.l2.weight.shape)
        #print("Shape of l3 weights:", self.l3.weight.shape)
        return output

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        #print("Batch size:", X.size(0))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # training!

model = Net()

train_dataloader = DataLoader(train_dataloader, batch_size=64)
test_dataloader = DataLoader(test_dataloader, batch_size=64)

learning_rate = 1e-3
batch_size = 64

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


sample_num = 143 # select a random sample

with torch.no_grad():
    r = model(train_dataloader[sample_num][0])

print('neural network output pseudo-probabilities:', r)
print('neural network output class number:', torch.argmax(r).item())
print('neural network output, predicted class:', categories[torch.argmax(r).item()])