import os
import time
from cv2 import compare
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import transforms as T
from torch.utils.tensorboard import SummaryWriter


folderDir = '' # for colab run with Google drive mounted.

# SAVE_DIR = folderDir + "Code"
SAVE_DIR = folderDir + "model"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
# MODEL_PATH = os.path.join(SAVE_DIR, "Rest50Best_RandomErase.pth")
MODEL_PATH = os.path.join(SAVE_DIR, "Rest50Best.pth")
LOG_PATH = os.path.join(SAVE_DIR, "TensorBoardLog.json")
DIR_TRAIN = folderDir+ "Data/train/"
DIR_TEST = folderDir+ "Data/test/"

def get_train_transform():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomCrop(204),
        T.ToTensor(),
        T.Normalize((0, 0, 0),(1, 1, 1)),
        T.RandomErasing()
    ])
    
def get_val_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0, 0, 0),(1, 1, 1))
    ])
class CatDogDataset(Dataset):
    
    def __init__(self, imgs, class_to_int, dir = DIR_TRAIN, mode = "train", transforms = None):
        
        super().__init__()
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms
        self.dir = dir
        
    def __getitem__(self, idx):
        
        image_name = self.imgs[idx]
        
        img = Image.open(self.dir + image_name)
        img = img.resize((224, 224))
        
        if self.mode == "train" or self.mode == "val":
        
            ### Preparing class label
            label = self.class_to_int[image_name.split(".")[0]]
            label = torch.tensor(label, dtype = torch.float32)

            ### Apply Transforms on image
            img = self.transforms(img)

            return img, label
        
        elif self.mode == "test":
            
            ### Apply Transforms on image
            img = self.transforms(img)

            return img
            
    def __len__(self):
        return len(self.imgs)

class_to_int = {"dog" : 0, "cat" : 1}
int_to_class = {0 : "dog", 1 : "cat"}
objNames = ['dog','cat']

class Q5_train():

    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.imgs = os.listdir(DIR_TRAIN) 
        self.test_imgs = os.listdir(DIR_TEST)
        self.writer = SummaryWriter(os.path.join(SAVE_DIR,'ResNet50_experience'))
        
        self.model = resnet50(pretrained = True)

        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1, bias = True),
            nn.Sigmoid()
        )
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0001)
        # Learning Rate Scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 5, gamma = 0.5)
        #Loss Function
        self.criterion = nn.BCELoss()
        # # Loading model to device
        self.model.to(self.device)

        #Parameter
        self.EPOCHS = 50
        self.best_val_acc = 0
        # Logs - Helpful for plotting after training finishes
        self.train_logs = {"loss" : [], "accuracy" : [], "time" : []}
        self.val_logs = {"loss" : [], "accuracy" : [], "time" : []}
        pass

    def load_train_dataset(self):
        train_imgs, val_imgs = train_test_split(self.imgs, test_size = 0.25)
        train_dataset = CatDogDataset(train_imgs, class_to_int, mode = "train", transforms = get_train_transform())
        val_dataset = CatDogDataset(val_imgs, class_to_int, mode = "val", transforms = get_val_transform())

        self.train_data_loader = DataLoader(
            dataset = train_dataset,
            num_workers = 8,
            batch_size = 32,
            shuffle = True
        )

        self.val_data_loader = DataLoader(
            dataset = val_dataset,
            num_workers = 8,
            batch_size = 32,
            shuffle = True
        )

    def load_test_dataset(self):
        test_dataset = CatDogDataset(self.test_imgs, class_to_int, mode = "test", transforms = get_val_transform())
        self.test_data_loader = DataLoader(
            dataset = test_dataset,
            num_workers = 8,
            batch_size = 32,
            shuffle = True
        )
    
    def show_before_after(self, path_compare:str = None):
        for images, labels in self.train_data_loader:
            fig, ax = plt.subplots(figsize = (50, 50))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, 8).permute(1,2,0))
            break
        plt.show()
        if path_compare:
            acc_value = []
            with open(path_compare, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    acc_value.append(float(line))
            def bar_plot(acc_value, label_names = ['Before Random Erasing', 'After Random Erasing']):
                fig = plt.figure()
                plt.bar(label_names, acc_value)
                plt.show()
            
            bar_plot(acc_value)


    @staticmethod
    def accuracy(preds, trues):
        preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
        ### Calculating accuracy by comparing predictions with true labels
        acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
        ### Summing over all correct predictions
        acc = np.sum(acc) / len(preds)
        return (acc * 100)

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = []
        epoch_acc = []
        start_time = time.time()
        for i, (images, labels) in enumerate(self.train_data_loader):
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            labels = labels.reshape((labels.shape[0], 1)) 
            self.optimizer.zero_grad()
            preds = self.model(images)
            _loss = self.criterion(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)
            acc = self.accuracy(preds, labels)
            epoch_acc.append(acc)
            #Backward
            # writer.add_scalar('Loss (item)', loss, i)
            # writer.add_scalar('Accuracy (item)', acc, i)
            _loss.backward()
            self.optimizer.step()
        ###Overall Epoch Results
        end_time = time.time()
        total_time = end_time - start_time
        ###Acc and Loss
        epoch_loss = np.mean(epoch_loss)
        epoch_acc = np.mean(epoch_acc)
        ###Storing results to logs
        self.train_logs["loss"].append(epoch_loss)
        self.train_logs["accuracy"].append(epoch_acc)
        self.train_logs["time"].append(total_time)
        return epoch_loss, epoch_acc, total_time

    def val_one_epoch(self, best_val_acc):
        self.model.eval()
        ### Local Parameters
        epoch_loss = []
        epoch_acc = []
        start_time = time.time()
        
        ###Iterating over data loader
        with torch.no_grad():
            for images, labels in self.val_data_loader:
                #Loading images and labels to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                labels = labels.reshape((labels.shape[0], 1)) # [N, 1] - to match with preds shape
                #Forward
                preds = self.model(images)
                #Calculating Loss
                _loss = self.criterion(preds, labels)
                loss = _loss.item()
                epoch_loss.append(loss)
                #Calculating Accuracy
                acc = self.accuracy(preds, labels)
                epoch_acc.append(acc)
            
            ###Overall Epoch Results
            end_time = time.time()
            total_time = end_time - start_time
            
            ###Acc and Loss
            epoch_loss = np.mean(epoch_loss)
            epoch_acc = np.mean(epoch_acc)
            
            ###Storing results to logs
            self.val_logs["loss"].append(epoch_loss)
            self.val_logs["accuracy"].append(epoch_acc)
            self.val_logs["time"].append(total_time)
            
            ###Saving best model
            if epoch_acc > self.best_val_acc:
                self.best_val_acc = epoch_acc
                torch.save(self.model.state_dict(), MODEL_PATH)
            return epoch_loss, epoch_acc, total_time

    def train(self):

        for epoch in range(self.EPOCHS):
            ###Training
            print('start training epoch ', epoch)
            loss, acc, _time = self.train_one_epoch()
            self.writer.add_scalar('Loss (epoch)', loss, epoch)
            self.writer.add_scalar('Accuracy (epoch)', acc, epoch)
            #Print Epoch Details
            print("\nTraining")
            print("Epoch {}, Loss : {:.4f}, Acc : {:.4f}, Time : {:.4f}".format(epoch+1, loss, acc, _time))
            
            ###Validation
            loss, acc, _time = self.val_one_epoch(self.best_val_acc)
                    
            #Print Epoch Details
            print("\nValidating")
            print("Epoch {}Epoch {}, Loss : {:.4f}, Acc : {:.4f}, Time : {:.4f}".format(epoch+1, loss, acc, _time))
        
        self.writer.export_scalars_to_json(LOG_PATH)
        self.writer.close()
    
    def show_plot_acc_loss(self):
        #Loss
        plt.title("Loss")
        plt.plot(np.arange(1, 11, 1), self.train_logs["loss"], color = 'blue')
        plt.plot(np.arange(1, 11, 1), self.val_logs["loss"], color = 'yellow')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

        #Accuracy
        plt.title("Accuracy")
        plt.plot(np.arange(1, 11, 1), self.train_logs["accuracy"], color = 'blue')
        plt.plot(np.arange(1, 11, 1), self.val_logs["accuracy"], color = 'yellow')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.show()

class Q5_test():
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.test_imgs = os.listdir(DIR_TEST)
        
        self.model = resnet50()
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1, bias = True),
            nn.Sigmoid()
        )
        self.load_model(MODEL_PATH)
        #Loss Function
        self.criterion = nn.BCELoss()
        # # Loading model to device
        self.model.to(self.device)
        pass
    def load_model(self, path):
        if os.path.exists(path):
            if torch.cuda.is_available():
                checkpoint = torch.load(path)
            else:
                checkpoint = torch.load(path, map_location=torch.device('cpu')) 
            # print(checkpoint['epoch'])
            self.model.load_state_dict(checkpoint)
    def load_test_dataset(self):
        test_dataset = CatDogDataset(self.test_imgs, class_to_int, dir=DIR_TEST, mode = "test", transforms = get_val_transform())
        self.test_data_loader = DataLoader(
            dataset = test_dataset,
            num_workers = 8,
            batch_size = 32,
            shuffle = True
        )
        self.test_dataset = test_dataset

    def predict(self, index, show_image = True):
        self.model.eval()
        if index < 0:
            index = 0
        elif index > len(self.test_imgs):
            index = len(self.test_imgs) -1 

        input = self.test_dataset[index]
        loader = DataLoader(input, batch_size= 1, shuffle= False,
            num_workers= 1, pin_memory= False
        )
        with torch.no_grad():
            image = loader.dataset
            image = torch.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
            image = image.to(self.device)
            output = self.model(image)
            if output.data > 0.5:
                label = "Class: Cat"
            else:
                label = "Class: Dog"

            X = image.cpu().detach().numpy().transpose([0,2,3,1])[0]
            if show_image:
                self.plot_image(X, label)
    @staticmethod
    def plot_image(image, label):
        plt.imshow(image, interpolation='spline16')
        plt.title(label)
        plt.show()
if __name__ == "__main__":
    train_model = Q5_train()
    train_model.load_train_dataset()
    train_model.show_before_after(r'model/compareModel.txt')
    # train_model.train()
    # train_model.show_plot_acc_loss()

    # test_model = Q5_test()
    # test_model.load_test_dataset()
    # test_model.predict(2515)