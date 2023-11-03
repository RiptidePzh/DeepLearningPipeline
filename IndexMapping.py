def create_index_map():
    idx_to_labels = {}
    idx_to_labels[0] = '0%-20%'
    idx_to_labels[1] = '20%-40'
    idx_to_labels[2] = '40%-60%'
    idx_to_labels[3] = '60%+'
    return idx_to_labels


def create_index_map_test(dataset_dir='DataSet'):
    from torchvision import datasets
    import os
    train_path = os.path.join(dataset_dir, 'train')
    train_dataset = datasets.ImageFolder(train_path)
    idx_to_labels = {y:x for x,y in train_dataset.class_to_idx.items()}
    return idx_to_labels