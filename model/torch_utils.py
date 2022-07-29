# import numpy as np
import io
import torch
import torchvision.transforms as tt
# from torchvision.datasets import ImageFolder
# from torch.utils.data.dataloader import DataLoader
from io import BytesIO
from PIL import Image
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ['Shih-Tzu', 'Rhodesian ridgeback', 'Beagle', 'English foxhound', 'Border terrier', 'Australian terrier', 
           'Golden retriever', 'Old English sheepdog', 'Samoyed', 'Dingo']

model = torch.load('model.pth')
model.eval()

def transform_image(image_bytes):
    tfms = tt.Compose([tt.RandomCrop(160), 
                       tt.ToTensor()])
    image = Image.open(image_bytes)
    return tfms(image).unsqueeze(0)


def get_prediction(image_tensor):
    # Convert to a batch of 1
    tensor = transform_image(BytesIO(image_tensor))
    xb = tensor.to(device)
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim = 1)
    # Retrieve the class label
    return classes[preds[0].item()]


# print('Predicted:', get_prediction('/home/slam/dog-clf/ILSVRC2012_val_00002701.JPEG'))