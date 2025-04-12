import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import models
from torch.autograd import Variable
import time
import sys
import seaborn as sn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import time

class video_dataset(Dataset):
    def __init__(self,video_names,labels,sequence_length = 60,transform = None):
        self.video_names = video_names
        self.labels = labels
        self.transform = transform
        self.count = sequence_length
    def __len__(self):
        return len(self.video_names)
    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0,a)
        temp_video = video_path.split('/')[-1]
        #print(temp_video)
        label = self.labels.iloc[(self.labels.loc[self.labels["file"] == temp_video].index.values[0]),1]
        if(label == 'FAKE'):
          label = 0
        if(label == 'REAL'):
          label = 1
        for i,frame in enumerate(self.frame_extract(video_path, first_frame)):
          frames.append(self.transform(frame))
          if(len(frames) == self.count):
            break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        #print("length:" , len(frames), "label",label)
        return frames,label
    def frame_extract(self,path, start_frame):
      vidObj = cv2.VideoCapture(path)
      vidObj.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image


#count the number of fake and real videos
def number_of_real_and_fake_videos(data_list):
  header_list = ["file","label"]
  lab = pd.read_csv('./Data/Gobal_metadata.csv',names=header_list)
  fake = 0
  real = 0
  for i in data_list:
    temp_video = i.split('/')[-1]
    label = lab.iloc[(lab.loc[lab["file"] == temp_video].index.values[0]),1]
    if(label == 'FAKE'):
      fake+=1
    if(label == 'REAL'):
      real+=1
  return real,fake

class Model(nn.Module):
    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT) #Residual Network CNN
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(torch.mean(x_lstm,dim = 1)))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStoppingWithCheckpoint:
    def __init__(self, patience=5, min_delta=0, checkpoint_path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metrics = {"Actual": None, "Predict": None, "Loss": float('inf'), "Accuracy": None}
        self.counter = 0
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, metrics, model):
        val_loss = metrics['Loss']
        if val_loss < self.best_metrics['Loss'] - self.min_delta:
            self.best_metrics = metrics
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)  # Lưu checkpoint
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(torch.load(self.checkpoint_path))


def train_epoch(epoch, num_epochs, data_loader, model, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (inputs, targets) in enumerate(data_loader):
        if torch.cuda.is_available():
            targets = targets.type(torch.cuda.LongTensor)
            inputs = inputs.cuda()
        _,outputs = model(inputs)
        loss  = criterion(outputs,targets.type(torch.cuda.LongTensor))
        acc = calculate_accuracy(outputs, targets.type(torch.cuda.LongTensor))
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d / %d] [Loss: %f, Acc: %.2f%%]"
            % (
                epoch,
                num_epochs,
                i,
                len(data_loader),
                losses.avg,
                accuracies.avg))
    return losses.avg,accuracies.avg


def test(model, data_loader ,criterion):
    print()
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    pred = []
    true = []
    count = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if torch.cuda.is_available():
                targets = targets.cuda().type(torch.cuda.FloatTensor)
                inputs = inputs.cuda()
            _,outputs = model(inputs)
            loss = torch.mean(criterion(outputs, targets.type(torch.cuda.LongTensor)))
            acc = calculate_accuracy(outputs,targets.type(torch.cuda.LongTensor))
            _,p = torch.max(outputs,1)
            true += (targets.type(torch.cuda.LongTensor)).detach().cpu().numpy().reshape(len(targets)).tolist()
            pred += p.detach().cpu().numpy().reshape(len(p)).tolist()
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            sys.stdout.write(
                "\r[Batch %d / %d]  [Loss: %f, Acc: %.2f%%] Testing..."
                % (
                    i,
                    len(data_loader),
                    losses.avg,
                    accuracies.avg
                )
            )
            if torch.isnan(inputs.float()).any():
                print("Warning: inputs contain NaN")

            if torch.isnan(outputs).any():
                print("Warning: outputs contain NaN")

        print('\nAccuracy {}'.format(accuracies.avg))
    return true,pred,losses.avg,accuracies.avg


def print_confusion_matrix(y_true, y_pred,save_dir, labels=["Fake", "Real"]):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f'True Positive (TP) = {tp}')
    print(f'False Positive (FP) = {fp}')
    print(f'False Negative (FN) = {fn}')
    print(f'True Negative (TN) = {tn}')
    print('\n')

    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    sn.set(font_scale=1.4)
    plt.figure(figsize=(6, 5))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16}, cbar=False)

    plt.ylabel('Actual label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.title('Confusion Matrix', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(os.path.join(save_dir, 'Confusion Matrix.png'))
    # plt.show()

    # Tính các chỉ số hiệu suất
    accuracy = (tp + tn) / cm.sum()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print(f"AUC Score: {auc_score:.2%}")

def plot_loss(train_loss_avg, test_loss_avg, num_epochs, save_dir):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss_avg, 'g-o', label='Training Loss')
    plt.plot(epochs, test_loss_avg, 'b-s', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'), dpi=300, bbox_inches='tight')
    # plt.show()


def plot_accuracy(train_accuracy, test_accuracy, num_epochs, save_dir):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracy, 'g-o', label='Training Accuracy')
    plt.plot(epochs, test_accuracy, 'b-s', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'), dpi=300, bbox_inches='tight')
    # plt.show()


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()
    return 100* n_correct_elems / batch_size


def main():
    with open("./Models/video_files.pkl", "rb") as f:
        video_files = pickle.load(f)
    print("Đã đọc lại video_files từ video_files.pkl")

    header_list = ["file", "label"]
    labels = pd.read_csv('./Data/Gobal_metadata.csv', names=header_list)
    train_videos, valid_videos = train_test_split(video_files, test_size=0.2, random_state=42)
    print("Total Train : ", len(train_videos))
    print("Total Test : ", len(valid_videos))

    print("TRAIN: ", "Real:", number_of_real_and_fake_videos(train_videos)[0], " Fake:",
          number_of_real_and_fake_videos(train_videos)[1])
    print("TEST: ", "Real:", number_of_real_and_fake_videos(valid_videos)[0], " Fake:",
          number_of_real_and_fake_videos(valid_videos)[1])

    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    
    length = 80
    train_data = video_dataset(train_videos, labels, sequence_length=length, transform=train_transforms)
    val_data = video_dataset(valid_videos, labels, sequence_length=length, transform=test_transforms)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4, persistent_workers=True)
    valid_loader = DataLoader(val_data, batch_size=2, shuffle=True, num_workers=4, persistent_workers=True)

    # Khởi tạo model
    model = Model(2).cuda()

    # learning rate
    lr = 1e-5  # 0.001

    # Number of epochs & Early Stopping Set up
    num_epochs = 20
    patience = 5

    checkpoint = input("Input Model Name: ")
    save_dir = f"./Process/{checkpoint}"
    print("Start training...")
    start_time = time.time()
    checkpoint_path = f"./Models/{checkpoint}.pt"
    early_stopping = EarlyStoppingWithCheckpoint(patience=patience, checkpoint_path=checkpoint_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Sử dụng khi mất cân bằng dữ liệu giữa hai lớp Real (trọng số 1) và Fake  (Trọng số 15)
    # class_weights = torch.from_numpy(np.asarray([1,15])).type(torch.FloatTensor).cuda()
    # criterion = nn.CrossEntropyLoss(weight = class_weights).cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    train_loss_avg = []
    train_accuracy = []
    test_loss_avg = []
    test_accuracy = []
    for epoch in range(1, num_epochs + 1):
        l, acc = train_epoch(epoch, num_epochs, train_loader, model, criterion, optimizer)
        train_loss_avg.append(l)
        train_accuracy.append(acc)
        true, pred, tl, t_acc = test(model, valid_loader, criterion)
        test_loss_avg.append(tl)
        test_accuracy.append(t_acc)
        metrics = {"Actual": true, "Predict": pred, "Loss": tl, "Accuracy": t_acc}
        early_stopping(metrics, model)
        if early_stopping.early_stop:
            print(f"Early Stopping After {patience} Epochs Not Improve")
            break
    plot_loss(train_loss_avg, test_loss_avg, len(train_loss_avg), save_dir)
    plot_accuracy(train_accuracy, test_accuracy, len(train_accuracy), save_dir)

    print(confusion_matrix(early_stopping.best_metrics['Actual'], early_stopping.best_metrics['Predict']))
    print_confusion_matrix(early_stopping.best_metrics['Actual'],early_stopping.best_metrics['Predict'], save_dir)
    end_time = time.time()
    print(f"Training end successfully! Time: {(end_time - start_time)/60:.3f} min")
    # plt.show()

if __name__ == "__main__":
    main()
    # print(Model(2).cuda().parameters)
    # total_params = sum(p.numel() for p in Model(2).parameters())
    # print("Total Number Of Params: ",total_params)
