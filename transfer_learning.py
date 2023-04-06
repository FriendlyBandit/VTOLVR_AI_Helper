import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.nodes import FARMReader
from haystack.utils.early_stopping import EarlyStopping
early_stopping = EarlyStopping(
    metric='top_n_accuracy',
    save_dir='early_stop_model',
    mode='max',
    patience=10,
    min_delta=0.001,
    min_evals=0,
)

reader = FARMReader(model_name_or_path="vblagoje/bart_lfqa", use_gpu=True)
data_dir = "./training/"
reader.train(data_dir=data_dir, 
             train_filename="QA_training_data.json", 
             dev_filename="QA_training_data.json", 
             use_gpu=True, 
             save_dir="my_model_generative", 
             early_stopping=early_stopping)

