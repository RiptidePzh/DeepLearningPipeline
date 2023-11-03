from InferenceBackbone import *
from IndexMapping import *
import torch
import torch.nn as nn

if __name__ == '__main__':
    # Call GPU backend
    device = torch.device("mps" if torch.has_mps else "cpu")
    # Define config dictionary
    config = dict(
        epochs=30,
        classes=5,
        batch_size=32,
        learning_rate=0.005,
        dataset_dir='Bald_DataSet'
    )
    # Load a pretrained model
    from torchvision.models import densenet161
    model = densenet161()
    model.classifier = nn.Linear(model.classifier.in_features, config['classes'])

    # Load the saved model weights
    model.load_state_dict(torch.load('Checkpoint/Deployed CV Model/DenseNet 161 Bald Test/best-0.438.pth'))

    # list of image paths
    image_paths = ['Bald_Dataset/val/二级/2023-07-20 11.33.37_店长_三级 二级.jpg',
                   'Bald_Dataset/val/二级/2023-07-20 22.45.41_黄龙王莹_四级 二级.png']

    # Create the inference pipeline
    pipeline = ImageInferencePipeline(model, device=device)

    # Perform inference on the images
    results = pipeline.infer(image_paths)

    # Create Index Mapping using training dataset directory
    index_map = create_index_map_test(config['dataset_dir'])
    print(index_map)
    # Print the results
    for image_path, probabilities in results.items():
        print("Image:", image_path)
        for class_index, class_name in index_map.items():
            print("Class:", class_name)
            print("Probability:", probabilities[class_index])