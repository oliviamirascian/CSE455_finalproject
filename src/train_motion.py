import os
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import models

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from torch.optim import Adam

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DeblurMotion(Dataset):
    def __init__(self, image_data, labels, transforms):
        self.image_data = image_data
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        blur_image = cv2.imread(f"../input/motion_blurred/{self.image_data[index]}")

        if self.transforms:
            blur_image = self.transforms(blur_image)

        grayscale_image = cv2.imread(f"../input/grayscaled/{self.labels[index]}")
        grayscale_image = self.transforms(grayscale_image)
        return blur_image, grayscale_image


def train(model, train_loader, optimizer, criterion):
    train_loss = 0.0
    model.train()

    for i, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)
    print("Train Loss:", str(round(train_loss, 5)))
    return train_loss


def validate(model, val_loader, criterion, epoch, model_name):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
    val_loss = val_loss / len(val_loader)
    print("Validation Loss:", str(round(val_loss, 5)))
    if epoch == 0:
        save_deblurred_image(data.cpu().data, name=f"../output/motion_deblurred/motion_blurred.jpg")
        save_deblurred_image(target.cpu().data, name=f"../output/motion_deblurred/motion_original.jpg")
    save_deblurred_image(output.cpu().data, name=f"../output/motion_deblurred/motion_deblurred{epoch}_{model_name}.jpg")
    return val_loss


def save_deblurred_image(img, name):
    img = img.view(img.size(0), 3, 256, 256)
    save_image(img, name)


def main():
    gauss_blur = os.listdir('../input/motion_blurred/')
    grayscale = os.listdir('../input/grayscaled')

    x_input = []
    for i in range(len(gauss_blur)):
        x_input.append(gauss_blur[i])

    y_input = []
    for i in range(len(grayscale)):
        y_input.append(grayscale[i])

    (x_train, x_val, y_train, y_val) = train_test_split(x_input, y_input, test_size=0.25)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    batch_size = 1
    train_data = DeblurMotion(x_train, y_train, transform)
    val_data = DeblurMotion(x_val, y_val, transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model1 = models.SRCNN1().to(device)
    model2 = models.SRCNN2().to(device)
    model3 = models.SRCNN3().to(device)

    epochs = 10
    lr = 5e-5
    models_list = [model1, model2, model3]
    model_names = ['model1', 'model2', 'model3']
    train_losses = []
    val_losses = []
    for (model, model_name) in zip(models_list, model_names):
        for epoch in range(epochs):
            optimizer = Adam(model.parameters(), lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=5,
                factor=0.1,
                verbose=True
            )
            criterion = nn.MSELoss()
            print(f"{model_name}: Epoch {epoch + 1} of {epochs}")
            train_loss = train(model, train_loader, optimizer, criterion)
            val_loss = validate(model, val_loader, criterion, epoch, model_name)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

        plt.figure(figsize=(10, 7))
        plt.plot(train_losses, color='blue', label='Train Loss')
        plt.plot(val_losses, color='red', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{model_name} Motion Deblur Training and Validation Loss')
        plt.savefig(f'../output/motion_loss_{model_name}.png')
        plt.show()
        train_losses = []
        val_losses = []


if __name__ == "__main__":
    main()