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
        classes=4,
        batch_size=32,
        learning_rate=0.005,
        dataset_dir='../../DataSet'
    )
    # Load a pretrained model
    from torchvision.models import densenet161
    model = densenet161()
    model.classifier = nn.Linear(model.classifier.in_features, config['classes'])

    # Load the saved model weights
    model.load_state_dict(torch.load('Checkpoint/Deployed CV Test/DenseNet 161 Test003/best-0.750.pth'))

    # list of image paths
    image_paths = ['DataSet/train/10-20%/2023-07-15 11.16.12_郭汝倩_10%-20%.png',
                   'DataSet/train/10-20%/2023-07-15 11.16.12_郭汝倩_10%-20%（1）.png']

    # Create the inference pipeline
    pipeline = ImageInferencePipeline(model, device=device)

    # Perform inference on the images
    results = pipeline.infer(image_paths)

    # Create Index Mapping using training dataset directory
    index_map = create_index_map_test('../../DataSet')

    # Print the results
    for image_path, probabilities in results.items():
        print("Image:", image_path)
        for class_index, class_name in index_map.items():
            print("Class:", class_name)
            print("Probability:", probabilities[class_index])