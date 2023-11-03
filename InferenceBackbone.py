import torch
from IndexMapping import create_index_map
from torchvision import models
from torchvision import datasets
import os
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

class ImageInferencePipeline:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor()
                                             ])

    # @staticmethod
    # def create_index_map(dataset_dir='DataSet'):
    #     train_path = os.path.join(dataset_dir, 'train')
    #     train_dataset = datasets.ImageFolder(train_path)
    #     idx_to_labels = {y:x for x,y in train_dataset.class_to_idx.items()}
    #     return idx_to_labels

    def infer(self, image_paths):
        results = {}
        for image_path in image_paths:
            image = self.transform(Image.open(image_path))
            image = image.to(self.device)
            output = self.model(image.unsqueeze(0))
            probabilities = F.softmax(output, dim=1)
            results[image_path] = probabilities.squeeze().cpu().detach().numpy()
        return results

