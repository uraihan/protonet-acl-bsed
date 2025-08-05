from scripts import dataset, utils
from scripts.extract_features import FeatureExtractor
from scripts.data_sampler import DataSampler
# NOTE: Training pipeline:
# - Check if features already extracted from all training samples
# - Create dataset class from the extracted features and corresponding CSV files
# - Pass batch through resnet model
# - Get feature maps from resnet
# - calculate its prototypes
# - Refine prototype using ACL
config_path = "config.yaml"
dataset_path = "dataset/Development_Set"
config = utils.open_yaml_namedtuple(config_path)

feat_extractor = FeatureExtractor(dataset_path, config)
feat_extractor.get_features()
