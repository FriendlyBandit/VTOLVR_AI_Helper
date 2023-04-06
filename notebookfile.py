import logging
from haystack.utils import print_answers
import os
from haystack.document_stores import InMemoryDocumentStore
from haystack import Pipeline
from haystack.nodes import TextConverter, PreProcessor
from haystack.nodes import BM25Retriever
from haystack.nodes import Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import RAGenerator


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

document_store = InMemoryDocumentStore(use_bm25=True)

doc_dir = './documents'

indexing_pipeline = Pipeline()
text_converter = TextConverter()
preprocessor = PreProcessor(
    clean_whitespace=True,
    clean_header_footer=True,
    clean_empty_lines=True,
    split_by="passage",
    split_length=5,
    split_overlap=0,
    split_respect_sentence_boundary=False,
)

indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter"])
indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])

files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline.run_batch(file_paths=files_to_index)

retriever = BM25Retriever(document_store=document_store)

generator = RAGenerator()
pipeline = GenerativeQAPipeline(generator=generator, retriever=retriever)
# reader = FARMReader(model_name_or_path='my_model')
# pipeline = ExtractiveQAPipeline(reader, retriever)

query = ""
while(True):
    query = input()
    if query == 'q':
        break
    prediction = pipeline.run(
        query=query,
        params={
            "Retriever": {"top_k": 10},
        }
    )
    # prediction = pipeline.run(
    #     query=query,
    #     params={
    #         "Retriever": { 
    #             "top_k":10
    #         },
    #         "Reader":{
    #             "top_k":5
    #         }
    #     }
    # )

    print_answers(
    prediction,
    details="medium" ## Choose from `minimum`, `medium` and `all`)
    )

