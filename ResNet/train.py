import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
from model import resnet34
import sys
from tqdm import tqdm

from torchvision.models import resnet

decive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(decive)

data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229,0.224,0.225])
    ]),
    "val":transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
}

data_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
image_path =  os.path.join(data_root, "Flower_data", "flower_data")
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

train_dataset = datasets.ImageFolder(root=image_path + "/train",
                            transform=data_transform["train"])
train_num = len(train_dataset)

flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(cla_dict, indent=4)
with open('D:/pytorch/Flower_data/flower_data/class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size=16
train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True, num_workers=0)
validata_dataset = datasets.ImageFolder(root=image_path + "/val",
                    transform=data_transform["val"])
val_num = len(validata_dataset)
validata_loader = torch.utils.data.DataLoader(validata_dataset,
                    batch_size=batch_size, shuffle=False, num_workers=0)

net = resnet34(num_classes=5)
net.to(decive)
model_weight_path = "resnet34-333f7ec4.pth"
missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
inchannel = net.fc.in_features
print(inchannel)
net.fc = nn.Linear(inchannel, 5)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

epochs = 1
best_acc = 0.0
save_path = "ResNet/resnet34_mine.pth"
train_steps = len(train_loader)

for epoch in range(epochs):
    net.train()
    running_loss = 0.0

    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(decive))
        loss = loss_function(logits, labels.to(decive))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        '''rate = (step + 1)/len(train_loader)
        a = "*" * int(rate * 50)
        b = "*" * int((1 - rate) * 50)
        print("\rtrain loss:{:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100),a,b,loss),end="")
    print()'''
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss)

    net.eval()
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(validata_loader, file=sys.stdout)
        for val_data in val_bar:
            val_imgaes, val_labels = val_data
            outputs = net(val_imgaes.to(decive))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(decive)).sum().item()
            val_bar.desc = "valid epoch[{}/{}]".format(epoch+1,epochs)
            val_accurate = acc/val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %(epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

                

print("Finlished Training")
