from TrainingBackbone import *


if __name__ == '__main__':
    # Call GPU backend
    device = torch.device("mps" if torch.has_mps else "cpu")
    # Start Training Server (First time need to login into wandb)
    # TODO: Convert to local training server
    wandb.login()
    # Define config dictionary
    config = dict(
        epochs=30,
        classes=4,
        batch_size=32,
        step_size=5,
        gamma=0.5,
        dataset_dir='DataSet'
    )
    # Load a pretrained model
    from torchvision.models import vit_l_16, ViT_L_16_Weights
    model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).to(device)
    model.heads[0] = nn.Linear(model.heads[0].in_features, config['classes'])
    # Config Training Pipeline
    pipeline = PytorchTrainingPipeline(model,config,device)
    # Starting Training Pipeline
    pipeline.model_pipeline(project_name='Test_Classification_Project',
                            run_name='Test_vit_l_16')
    # Clean and Save Run
    wandb.finish()