import torch
import wandb
from TrainingBackbone import *

if __name__ == '__main__':
    # Call GPU backend
    device = torch.device("mps" if torch.has_mps else "cpu")
    # Start Training Server (First time need to login into wandb)
    # TODO: Convert to local training server
    wandb.login()
    # Define config dictionary
    config = dict(
        epochs=80,
        classes=5,
        batch_size=32,
        step_size=5,
        gamma=0.5,
        dataset_dir='Bald_DataSet'
    )
    # Load a pretrained model
    from torchvision.models import resnet152, ResNet152_Weights
    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1).to(device)
    model.fc = nn.Linear(model.fc.in_features, config['classes'])
    # Config Training Pipeline
    pipeline = PytorchTrainingPipeline(model,config,device)
    # Starting Training Pipeline
    pipeline.model_pipeline(project_name='Bald_Seg_CV_2',
                            run_name='ResNet 152')
    # Clean and Save Run
    wandb.finish()