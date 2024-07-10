# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    inference.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了用于在模型应用端进行推理，返回模型输出的流程
#               ★★★请在空白处填写适当的语句，将模型推理应用流程补充完整★★★
# -----------------------------------------------------------------------

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms
import matplotlib.pyplot as plt


def inference(image_path, model, device):
    """定义模型推理应用的流程。
    :param image_path: 输入图片的路径
    :param model: 训练好的模型
    :param device: 模型推理使用的设备，即使用哪一块CPU、GPU进行模型推理
    """
    # 将模型置为评估（测试）模式
    model.eval()

    image = Image.open(image_path).convert('RGB')


    plt.imshow(image)
    plt.title("Input Image")


    image=ToTensor()(image).unsqueeze(0)

    # 将图像和模型移动到指定设备
    image = image.to(device)
    model = model.to(device)

    # 2. 进行推理
    with torch.no_grad():
        output = model(image)

    # 3. 处理模型输出
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # 获取概率最高的类
    top_prob, top_class = probabilities.topk(1, dim=0)

    return top_class.item(), top_prob.item()


if __name__ == "__main__":
    # 指定图片路径
    image_path = "./images/test/signs/img_0043.png"

    # 加载训练好的模型
    model = torch.load('./models/model.pkl')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    print(device)
    # 显示图片，输出预测结果
    a,b=inference(image_path, model, device)
    print("识别手势数字为：",a,"\n概率为：",b)
    plt.show()