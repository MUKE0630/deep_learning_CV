import os
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet
#from torchvision.models import AlexNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image_path = "../../Flower_data/tulip.jpg"
    assert os.path.exists(image_path), "file: '{}' dose not exist".format(image_path)
    img = Image.open(image_path)

    plt.imshow(img)
    img = torch.unsqueeze(data_transform(img), dim=0)

    json_path = "../../Flower_data/flower_data/class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path,"r") as f:
        class_indict = json.load(f)

    model = AlexNet(num_classes=5).to(device)

    weight_path = "./Alexnet.pth"
    weight_path1 = "./alexnet-owt-4df8aa71.pth"
    assert os.path.exists(weight_path), "file: '{}' dose not exist.".format(weights_path)


    #in_features = model.classifier[6].in_features
    #model.classifier[6] = nn.Linear(in_features, 1000)
    model.load_state_dict(torch.load(weight_path))
    #model.classifier[6] = nn.Linear(in_features, 5)

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "net:{}  class: {}   prob: {:.3}".format("AlexNet",class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()