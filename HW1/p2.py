import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()
        

        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class SimpleNet(nn.Module):
    def __init__(self,num_classes=10):
        super(SimpleNet,self).__init__()

        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3,out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        
        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6
                                 ,self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128,out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1,128)
        output = self.fc(output)
        return output

#Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
train_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 32

#Load the training set
train_set = CIFAR10(root="./data",train=True,transform=train_transformations,download=True)

#Create a loder for the training set
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=4)


#Define transformations for the test set
test_transformations = transforms.Compose([
   transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

])

#Load the test set, note that train is set to False
test_set = CIFAR10(root="./data",train=False,transform=test_transformations,download=True)

#Create a loder for the test set, note that both shuffle is set to false for the test loader
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=4)


#Create model, optimizer and loss function
model = SimpleNet(num_classes=10)


optimizer = Adam(model.parameters(), lr=0.001,weight_decay=0.0001) #0.001
loss_fn = nn.CrossEntropyLoss()

#Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):

    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr




def save_models(epoch):
    torch.save(model.state_dict(), "cifar10model_{}.model".format(epoch))
    print("Checkpoint saved")

def test():
    model.eval()
    test_acc = 0.0
    test_loss=0.0
    # Tsacc=[]
    # Tsloss=[]
    for i, (images, labels) in enumerate(test_loader):
     

        #Predict classes using images from the test set
        outputs = model(images)
        #Compute the loss based on the predictions and actual labels
        loss = loss_fn(outputs,labels)
        #Backpropagate the loss
        loss.backward()
        _,prediction = torch.max(outputs.data, 1)
        # prediction = prediction.cpu().numpy()
        test_loss += loss.cpu().data * images.size(0)
        prediction=prediction.type(torch.IntTensor)
        labels.data=labels.data.type(torch.IntTensor)
        test_acc += torch.sum(prediction == labels.data)
    # Tsacc.append(test_acc)
    # Tsloss.append(test_loss)
    # np.savetxt("Tsacc.csv", Tsacc, fmt='%s')
    # np.savetxt("Tsloss.csv", Tsacc, fmt='%s')
    #Compute the average acc and loss over all 10000 test images
    test_acc = test_acc / 10000
    test_loss = test_loss / 10000
    return test_acc,test_loss

def train(num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        # Tracc=[]
        # Trloss=[]
        for i, (images, labels) in enumerate(train_loader):

            #Clear all accumulated gradients
            optimizer.zero_grad()
            #Predict classes using images from the test set
            outputs = model(images)
            #Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs,labels)
            #Backpropagate the loss
            loss.backward()

            #Adjust parameters according to the computed gradients
            optimizer.step()
            train_loss += loss.cpu().data * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            prediction=prediction.type(torch.IntTensor)
            labels.data=labels.data.type(torch.IntTensor)
            train_acc += torch.sum(prediction == labels.data)
        # Tracc.append(train_acc)
        # Trloss.append(train_loss)
        # np.savetxt("Tracc.csv", Tracc, fmt='%s')
        # np.savetxt("Trloss.csv", Trloss,fmt='%s')
        #Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        #Compute the average acc and loss over all 50000 training images
        train_acc = train_acc / 50000
        train_loss = train_loss / 50000

        #Evaluate on the test set
        test_acc,test_loss = test()
    

        # Save the model if the test acc is greater than our current best
        if test_acc > best_acc:
            save_models(epoch)
            best_acc = test_acc


        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,test_acc))
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(train_acc, color='blue', label="training")
        ax1.plot(test_acc, color='red', label='train')
        ax1.set_title('PLot of Epoch vs Accuracy')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        ax2.plot(train_loss, color='blue', label="training")
        ax2.plot(test_loss, color='red', label='train')
        ax2.set_title('PLot of Epoch vs loss')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
    

if __name__ == "__main__":
    train(30) #200