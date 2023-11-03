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
    from torchvision.models import resnet101
    model = resnet101()
    model.fc = nn.Linear(model.fc.in_features, config['classes'])

    # Load the saved model weights
    model.load_state_dict(torch.load('../../Checkpoint/Hair_Seg_CV_2/ResNet 101/best-0.938.pth'))

    # list of image paths
    image_paths = ['/Users/zihan/Documents/Pycharm_Env/DeepLearningPipeline/DataSet/train/10-20%/2023-07-15 11.16.12_郭汝倩_10%以下.png',
                   '/Users/zihan/Documents/Pycharm_Env/DeepLearningPipeline/DataSet/train/10-20%/2023-07-15 11.16.12_郭汝倩_10%-20%（1）.png',
                   '/Users/zihan/Documents/Pycharm_Env/DeepLearningPipeline/DataSet/train/40-60%/2023-07-19 12.16.19_王玉玲_40%-50%（1）.png',
                   '/Users/zihan/Documents/Pycharm_Env/DeepLearningPipeline/DataSet/train/40-60%/2023-07-21 14.19.07_王晶晶_40%-50% 70%以上.png']

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
