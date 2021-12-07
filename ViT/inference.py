import os
import json

import torch
from  PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

from ViT_model_test import vit_base_patch16_224_in21k as create_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    image_path = "../DataSet/tst.jpg" # 准备预测的图片
    assert os.path.exists(image_path), "file: '{}' does not exist.".format(image_path)
    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim = 0)

    json_path = '../DataSet/flower_photos/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exists.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    model = create_model(num_classes = 5, has_logits = False).to(device)
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location = device))
    model.eval()

    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim = 0)
        predict_class = torch.argmax(predict).numpy()

    print_res = "class: {} prob: {:.3}".format(class_indict[str(predict_class)], predict[predict_class].numpy())

    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()

if __name__ == '__main__':
    main()