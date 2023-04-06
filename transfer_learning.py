import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.nodes import FARMReader
from haystack.utils import fetch_archive_from_http

reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True)
data_dir = "./training/"
reader.train(data_dir=data_dir, train_filename="QA_training_data.json", use_gpu=True, n_epochs=1, save_dir="my_model")

