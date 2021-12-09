import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from Unet import Unet
from DataSet import LiverDataset
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transforms = transforms.Compose([
    transforms.ToTensor()
])

def train_model(model, criterion, optimizer, dataload, num_epochs = 20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            print("%d/%d, train_loss: %0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch % d loss:%0.3f" %(epoch, epoch_loss / step))
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model

def train(args):
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    liver_dataset = LiverDataset("./data/train", transform = x_transforms, target_transform = y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    train_model(model, criterion, optimizer, dataloaders)

def test(args):
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckpt, map_location = 'cpu'))
    liver_dataset = LiverDataset("./data/val", transform = x_transforms, target_transform = y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size = 1)
    model.eval()

    plt.ion()
    with torch.no_grad():
        for x, _ in dataloaders:
            y = model(x).sigmoid()
            img_y = torch.squeeze(y).numpy()
            plt.imshow(img_y)
            plt.pause(0.01)
        plt.show()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch-size', type=int, default = 8)
    parse.add_argument("--ckpt", type=str, default = './weights_19.pth', help="the path of model weight file")

    args = parse.parse_args()

    # train(args)
    test(args)