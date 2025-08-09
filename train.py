from scripts import utils, build_dataset
from scripts.extract_features import FeatureExtractor
from data.dataloader import FewShotSampler
from data.dataset import ProtoDataset
from pathlib import Path
from torch.utils.data import DataLoader
from model import encoder, protonet, loss
from torchinfo import summary
import os
import torch


def check_training_samples(dataset_path):
    print("Checking availability of the development dataset in current local folder")
    if not os.path.exists(dataset_path):
        print(
            "Development dataset folder can not be found\nNow retrieving .zip file from the web")
        os.system(
            "wget https://zenodo.org/record/6482837/files/Development_Set.zip?download=1")
        os.system(f"unzip {dataset_path}")


def get_unextracted_files(dataset_path):
    """
    Get the list of all files which audio features is unextracted.

    Args:
        dataset_path (str | pathlib.Path): Path of Development_Set

    Returns:
        unextracted_files (list['str'] | list[PosixPath]): List of all
            unextracted files.
    """
    csv_files = utils.get_all_csv(dataset_path)
    unextracted_files = []
    for file in csv_files:
        h5_file = file.replace("csv", "h5")
        if not os.path.exists(h5_file):
            unextracted_files.append(file)

    return unextracted_files


def fit(protonet, optimizer, trainloader, loss_fn):
    # TODO: In progress
    """
    Fit data with label through encoder and protonet. Produces embedding that
    can be used on inference.

    Args:
        protonet (nn.Module): Prototypical Network.
        optimizer (nn.optim): Training Optimizer.
        trainloader (data.DataLoader): Pytorch Dataloader of the training set.
        loss_fn (nn.Module): Loss Function to calculate loss between prototype
                             and real label.
    """
    print("Start training...")

    protonet.train()
    optimizer.zero_grad()

    train_loss = []

    for idx, data in enumerate(trainloader):
        features, labels = data[0], data[1]
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()

        embeddings = protonet(features)
        loss = loss_fn(-embeddings, labels)

        loss.backward()
        optimizer.step()

        train_loss.append(loss.data.item())


def main():
    config_path = os.path.join(Path.cwd(), "config.yaml")
    dataset_path = os.path.join(Path.cwd(), "dataset/Development_Set")
    config = utils.open_yaml_namedtuple(config_path)

    # NOTE: Training pipeline:
    # - Extract features if features has not yet extracted from training samples
    check_training_samples(dataset_path)
    unextracted_files = get_unextracted_files(dataset_path)

    if len(unextracted_files) != 0:
        feat_extractor = FeatureExtractor(dataset_path, config,
                                          unextracted_files)
        feat_extractor.get_features()

    # - Sample features based on CSV metadata and bundle them into HDF5 files
    features = config.features
    for feat in features:
        if not os.path.exists(os.path.join(dataset_path,
                              f"Training_Set/train_{feat}.h5")):
            build_dataset.run(config, dataset_path)

    # - Create dataset class from bundled HDF5 file
    trainset = ProtoDataset('melspec', config)
    labels = torch.Tensor(trainset.label)
    n_ways = config.train_params.n_ways
    k_shots = config.train_params.k_shots
    trainloader = DataLoader(trainset, batch_sampler=FewShotSampler(labels,
                                                                    n_ways,
                                                                    k_shots,
                                                                    shuffle=True,
                                                                    include_query=True))

    # - Pass batch through resnet model
    model = encoder.ResNet()
    proto_net = protonet.ProtoNet(model, 1, 2096, device="cpu")
    # using cpu for debugging. replace with cuda later
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train_params.lr,
                                 weight_decay=config.train_params.weight_decay)
    loss_fn = loss.AngularContrastiveLoss(config.train_params.margin,
                                          config.train_params.alpha,
                                          device="cpu")

    # check output shape using torchinfo summary
    ex_data = next(iter(trainloader))[0]
    print(summary(model, ex_data.shape))

    epoch = config.train_params.epoch
    current_ep = 1

    for ep in range(current_ep, epoch+1):
        fit(proto_net, optimizer, trainloader, loss_fn)

    # - Get feature maps from resnet
    # - calculate its prototypes
    # - Refine prototype using ACL


if __name__ == "__main__":
    main()
