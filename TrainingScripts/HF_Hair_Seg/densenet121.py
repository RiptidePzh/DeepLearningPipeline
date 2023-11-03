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
    from timm import create_model
    model = create_model(model_name='densenet121',
                         pretrained=True,
                         num_classes=config['classes'])

    # Config Training Pipeline
    pipeline = PytorchTrainingPipeline(model,config,device)
    # Starting Training Pipeline
    pipeline.model_pipeline(project_name='HF_Hair_Seg',
                            run_name='densenet161')
    # Clean and Save Run
    wandb.finish()