# case1

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


# 이미지 출력함수
def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 그래프 출력 함수
def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


# 모델 파라미터 정의
class Config():
    training_dir = "./ORL/trainingwhite"
    testing_dir = "./ORL/testingwhite"
    # training_dir = "./image/ORL/train"
    # testing_dir = "./image/ORL/test"
    train_batch_size = 10
    train_number_epochs = 100



# 이미지 쌍을 입력으로 받아서
# 이미지 쌍 간의 유사도(similarity)를 예측하는 Siamese Neural Network을 학습시키기 위한 데이터셋 클래스
class SiameseNetworkDataset(Dataset):

    # 클래스의 인스턴스를 초기화하는 메서드 
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset  # 이미지 저장된 폴더경로
        self.transform = transform  # 이미지 변환
        self.should_invert = should_invert  # 이미지 쌍 반전시킬지

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)  # img0 이미지를 무작위로 가져옴
        should_get_same_class = random.randint(0, 1)  # 변수에 0이나 1을 랜덤으로 할당
        if should_get_same_class:  # 1이 선택되면 같은 클래스의 img1 이미지 선택
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:  # 2가 선택되면 다른 클래스의 이미지로 img1을 선택
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")  # grayscale
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:  # 이미지 데이터 전처리
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        #print(img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)))
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SiameseNetworkTestDataset(Dataset):

    # 클래스의 인스턴스를 초기화하는 메서드
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset  # 이미지 저장된 폴더경로
        self.transform = transform  # 이미지 변환
        self.should_invert = should_invert  # 이미지 쌍 반전시킬지

    def __getitem__(self, index):
        img0_tuple = ['./ORL/testingwhite/c0/main.png', 0]  # img0 이미지를 무작위로 가져옴
        should_get_same_class = 0  # 변수에 0이나 1을 랜덤으로 할당
        if should_get_same_class == 0:  # 1이 선택되면 같은 클래스의 img1 이미지 선택
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1] and img1_tuple[1] not in alreadySet:
                    alreadySet.append(img1_tuple[1])
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")  # grayscale
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:  # 이미지 데이터 전처리
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

folder_dataset = dset.ImageFolder(root=Config.training_dir)  # 트레이닝 데이터 받아오기

siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset, transform=transforms.Compose(
    [transforms.Resize((100, 100)), transforms.ToTensor()]), should_invert=False)
vis_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=0, batch_size=8)
dataiter = iter(vis_dataloader)  # 시각화를 위한 데이터로더

example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0], example_batch[1]), 0)

class SiameseNetwork(nn.Module):  # Siamese neural network 모델 클래스
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(  # 3개의 레이어를 가진 은닉층
            nn.ReflectionPad2d(1),  # 패딩 추가
            nn.Conv2d(1, 4, kernel_size=3),  # 1개의 input, 4개의 output 3*3 커널 크기로
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),  # 4개의 input, 8개의 output 3*3 커널 크기로
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),  # 8개의 input, 8개의 output 3*3 커널 크기로
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(  # 3개의 레이어를 가지는 완전연결층
            nn.Linear(8 * 100 * 100, 500),  # 8x100x100의 입력 데이터에 대해 500개의 출력을 갖는 레이어
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),  # 500개의 입력을 받아 500개의 출력을 갖는 레이어
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))  # 500개의 입력을 받아 5개의 출력을 갖는 레이어

    def forward_once(self, x):  # 한 이미지를 CNN에 입력으로 넣는다
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):  # forward_once 메서드를 두 번 호출하여 두 이미지에 대해 각각 벡터값을 출력하고 반환
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):  # 대조 손실 클래스
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):  # margin은 모델의 임계값
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)  # cnn으로 추출한 벡터의 거리를 유클리드 거리법을 통하여 계산
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # output1과 output2가 같은 클래스인지(1) 다른 클래스인지(0)에 따라 손실을 계산 

        return loss_contrastive


train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=0, batch_size=Config.train_batch_size)

print(train_dataloader)
net = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number = 0

for epoch in range(0, Config.train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data  # 학습 데이터 셋 설정
        img0, img1, label = img0, img1, label
        optimizer.zero_grad()  # 옵티마이저 초기화
        output1, output2 = net(img0, img1)  # 모델에 학습 이미지 쌍 넣기
        loss_contrastive = criterion(output1, output2, label)  # loss 계산
        loss_contrastive.backward()  # 역전파 작업 수행(가중치를 계산)
        optimizer.step()  # 모델의 가중치(weights)를 갱신
        if i % 10 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

# torch.save(net.state_dict(), 'model.pth')

# net = SiameseNetwork()
# net.load_state_dict(torch.load('model.pth'))

show_plot(counter, loss_history)

alreadySet = []

folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkTestDataset(imageFolderDataset=folder_dataset_test, transform=transforms.Compose(
    [transforms.Resize((100, 100)), transforms.ToTensor()]), should_invert=False)

test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)


for i in range(10):
    _, x1, label2 = next(dataiter)
    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = net(Variable(x0), Variable(x1))
    print(output1, output2)
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
