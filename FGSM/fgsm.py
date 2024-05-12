# %% [markdown]
# ## 1. Requirements

# %%
import os
import numpy as np
import json
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# %%

# %matplotlib inline

# %% [markdown]
# ## 2. Set Args

# %%
eps = 0.007
use_cuda = torch.cuda.is_available()

# %% [markdown]
# ## 3. Prepare Data

# %%
# https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json



# transform = transforms.Compose([
#     transforms.Resize((299, 299)),
#     transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
# ])

class_idx = json.load(open("F:\\MyCodes\\Defences\\imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
def image_folder_custom_label(root, transform, custom_label=idx2label) :
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']
    old_data = dsets.ImageFolder(root = root, transform = transform)
    old_classes = old_data.classes
    label2idx = {}
    for i, item in enumerate(idx2label) :
        label2idx[item] = i
    new_data = dsets.ImageFolder(root = root, transform = transform, 
                                 target_transform = lambda x : custom_label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx
    
    print(new_data.root)
    print('数据加载完成')
    return new_data



# normal_data = image_folder_custom_label(root = 'F:/Datasets/image_30_clean/', transform = transform, custom_label = idx2label)
# normal_loader = Data.DataLoader(normal_data, batch_size=10, shuffle=False)


# def imshow(img, title):
#     npimg = img.numpy()
#     fig = plt.figure(figsize = (5, 15))
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.title(title)
#     plt.show()


# normal_iter = iter(normal_loader)
# images, labels = normal_iter.next()


# imshow(torchvision.utils.make_grid(images, normalize=True), [normal_data.classes[i] for i in labels])


# ## 4. Download the Inception v3


# device = torch.device("cuda" if use_cuda else "cpu")
# model = models.inception_v3(pretrained=True).to(device)
# model.eval()

# 展示干净图像测试结果
# print("True Image & Predicted Label")

# model.eval()

# correct = 0
# total = 0

# for images, labels in normal_loader:
    
#     images = images.to(device)
#     labels = labels.to(device)
#     outputs = model(images)
    
#     _, pre = torch.max(outputs.data, 1)
    
#     total += 1
#     correct += (pre == labels).sum()
    
#     imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [normal_data.classes[i] for i in pre])
        
# print('Accuracy of test text: %f %%' % (100 * float(correct) / total))

# ## 5. Adversarial Attack


def fgsm_attack(images, labels, model, device, loss = nn.CrossEntropyLoss(),  eps=eps) :
    # model.eval()
    # images = images.to(device)
    # labels = labels.to(device)
    images.requires_grad = True
    outputs = model(images)

    model.zero_grad()
    # output[2]是分类输出的tensor
    cost = loss(outputs[2], labels).to(device)
    cost.backward()
    
    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    
    return attack_images




def fgsm_attack_file(normal_loader,model,device):
    loss = nn.CrossEntropyLoss()
    print("Attack Image & Predicted Label")
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    for images, labels in normal_loader:
        images = fgsm_attack(model, loss, images, labels, eps).to(device)
        
        labels = labels.to(device)
        print(labels)
        print(normal_data.classes[labels])
        outputs = model(images) # 输出预测标签
        
        for i in range(len(images)):
            image_name = f'{total}.png'  # 按照顺序命名图像
            save_folder = 'F:/Datasets/img_fgsm_e2'
            save_path = os.path.join(save_folder,image_name)
            # print(save_path)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            # 将对抗图像保存到指定路径
            
            save_image(images[i], save_path)
            total += 1
        
        # _, pre = torch.max(outputs.data, 1)
        
        
        #correct += (pre == labels).sum()
        print(total)
        
        #imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [normal_data.classes[i] for i in pre])
        
    # print('Accuracy of test text: %f %%' % (100 * float(correct) / total))
    print('ok')
