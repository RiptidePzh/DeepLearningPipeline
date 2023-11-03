from TrainingBackbone import *

class PytorchTrainingPipelineTest(PytorchTrainingPipeline):
    def __init__(self, model, hyerparameters, device):
        super().__init__(model, hyerparameters, device)


    @staticmethod
    def make_transform():
        ### Remove normalization, add autoaugment, change crop size to 640
        # make preprocessing module for dataset
        # Preprocessing Module for Training dataset
        train_transform = transforms.Compose([transforms.RandomResizedCrop(518),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(
                                                  mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                                              ])
        # Preprocessing Module for Testing dataset
        test_transform = transforms.Compose([transforms.Resize(518),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(
                                                 mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                                             ])
        return train_transform, test_transform


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
    from torchvision.models import vit_h_14, ViT_H_14_Weights
    model = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1).to(device)
    model.heads[0] = nn.Linear(model.heads[0].in_features, config['classes'])
    # Config Training Pipeline
    pipeline = PytorchTrainingPipelineTest(model,config,device)
    # Starting Training Pipeline
    pipeline.model_pipeline(project_name='Test_Classification_Project',
                            run_name='Test_vit_h_14')
    # Clean and Save Run
    wandb.finish()