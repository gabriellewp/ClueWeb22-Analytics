import sys
import orjson, csv, json
import gzip
import glob, os

from pyserini.search import SimpleSearcher
from pyserini.index import IndexReader
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from sklearn.cluster import KMeans

def read_file_dynamic(file_path, start_line=0, num_lines=5):
    """
    Reads a specified range of lines from a gzipped JSON file.

    Args:
        file_path (str): Path to the gzipped file.
        start_line (int): The line number to start reading from (0-indexed).
        num_lines (int): The number of lines to read from the start line.

    """
    with gzip.open(file_path, 'rt') as f:
        # Skip to the start line
        for _ in range(start_line):
            next(f, None)

        # Read the specified number of lines
        for i, line in enumerate(f, start=start_line):
            if i >= start_line + num_lines:
                break
            item = orjson.loads(line)
            features = list(item.keys())
            print(f"Item {i - start_line + 1}: {item}")
            print(f"Features: {features}")

def read_file(file_path):
    """
    Reads a gzipped JSON lines file and prints the first 5 items and their features.
    Args:
        file_path (str): The path to the gzipped JSON lines file.
    Returns:
        None
    """
    
    with gzip.open(file_path, 'rt') as f:
        for i, line in enumerate(f):
            if i >=5:
                break
            item = orjson.loads(line)
            features = list(item.keys())
            print(f"item {item}")
            print(f"features {features}")

def index_collection(input_dir_path=None, convert_pyserini=False, input_filename=None, output_dir_path=None):
    """
    Indexes the collection in the given file path.
    Args:
        file_path (str): The path to the collection file.
    Returns:
        None
    """
    csv.field_size_limit(sys.maxsize)
    if convert_pyserini:
        output_filename = os.path.splitext(os.path.basename(input_filename))[0]
        with open(f"{input_dir_path}/{input_filename}", 'r', newline='') as tsvfile, open(f"{output_dir_path}/{output_filename}.jsonl", 'w') as jsonlfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                data = {
                    "id": row[0],
                    "contents": row[1]
                }
                jsonlfile.write(json.dumps(data) + '\n')
            
    indexing_cmd = f"""python -m pyserini.index.lucene \\
    --collection JsonCollection \\
    --input {output_dir_path} \\
    --index {output_dir_path}/pyserini-index \\
    --generator DefaultLuceneDocumentGenerator \\
    --threads 1 \\
    --storePositions --storeDocvectors --storeRaw"""
    print(f"Running {indexing_cmd}")
    os.system(indexing_cmd)
    

            
def count_json_gz_files(directory_path):
    """
    Counts the number of .json.gz files in the given directory and its subdirectories.
    Args:
        directory_path (str): The path to the directory where the search for .json.gz files will be performed.
    Returns:
        int: The number of .json.gz files found in the directory and its subdirectories.
    """
    
    json_gz_files = glob.glob(os.path.join(directory_path, '**', '*.json.gz'), recursive=True)
    return len(json_gz_files)

def count_json_gz_files_records(directory_path):
    json_gz_files = glob.glob(os.path.join(directory_path, '**', '*.json.gz'), recursive=True)
    # file_count = len(json_gz_files)
    
    records_count = {}
    
    for file_path in json_gz_files:
        with gzip.open(file_path, 'rt') as f:
            records_count[file_path] = sum(1 for line in f)
        records_count[file_path] = records_count
    return records_count

def extract_feature_list(jsonl_file_path=None, id_doc=False, content_doc=False):
    """
    Extracts the 'content' field from a .jsonl file and returns a list of content values.
    It can only be one of the two fields: 'id' or 'content'.
    Args:
        jsonl_file_path (str): Path to the .jsonl file.
        id_doc (bool): If True, the 'id' field will be extracted.
        content_doc (bool): If True, the 'content' field will be extracted.
    Returns:
        list: A list of values from either id or content field
    """
    if content_doc:
        contents = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)  # Parse each line as a JSON object
                if 'contents' in data:  # Check if the 'contents' field exists
                    contents.append(data['contents'])
        return contents
    elif id_doc:
        ids = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                if 'id' in data:
                    ids.append(data['id'])
        return ids
    else:
        raise ValueError("Either id_doc or content_doc must be True")
    
def load_model():
    """
    Loads a pre-trained model for causal language modeling.

    Returns:
        model (AutoModelForCausalLM): The loaded pre-trained model.
    """

    model_name = "Meta-Llama-3-8B-Instruct"
    model_path = "/ivi/ilps/datasets/models/LLMs/Llama3.1/Meta-Llama-3.1-8B"
    float16 = True
    batch_size = 16
    device_map= "auto"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    quantization_config = bnb_config
    model = AutoModel.from_pretrained(model_path, device_map=device_map, quantization_config=quantization_config)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return model

def load_tokenizer():
    """
    Load and configure a tokenizer for the HF model.

    Returns:
        tokenizer (AutoTokenizer): The loaded and configured tokenizer.
    """
    
    truncation= True
    padding= True
    padding_side= "left"
    maximum_length = 60000
    model_max_length = 60000
    model_path = "/ivi/ilps/datasets/models/LLMs/Llama3.1/Meta-Llama-3.1-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=truncation, padding=padding, padding_side=padding_side, maximum_length=maximum_length, model_max_length=model_max_length)
    # tokenizer.chat_template = cfg.model.tokenizer_chat_template_HF   
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
def clustering_and_topic_extraction():
    # index_path = "/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-docs/jsonl/pyserini-index"  
    # searcher = SimpleSearcher(index_path)    
    print("start clustering and topic extraction")
    #working with senteceTransformer is incredibly slow and almost no progress, hence changing to AutoTokenizer
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # print("done loading model")
    
    #using AutoTokenizer

    # tokenizer = load_tokenizer()
    # model = load_model()
    # index_reader = IndexReader(index_path)
    #extracting doc embeddings
    # doc_ids = searcher.docids()
    # documents = [searcher.doc(doc_id).raw() for doc_id in doc_ids]
    
    # doc_ids = index_reader.object.get_document_ids()
    # documents = [searcher.doc(doc_id).raw() for doc_id in doc_ids if searcher.doc(doc_id) is not None]

    #replace the documents with raw string from dataset
    documents = extract_feature_list(content_doc="True", jsonl_file_path="/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-docs/jsonl/msmarco-docs-id-contents.jsonl")
    print("documents length", len(documents))
    embeddings = embedding_model.encode(documents[0:2])
    
    print("done obtaining embeddings")
    #clustering with dbscan
    for eps_ in [0.4, 0.6, 0.8, 1.0]:
        print('eps', eps_)
        
        #trying with dbscan
        # dbscan = DBSCAN(eps=eps_, min_samples=2, metric='cosine') 
        # clusters = dbscan.fit_predict(embeddings)
        
        #with kmeans
        clusters = KMeans(n_clusters=2, random_state=0).fit_predict(embeddings)
        
        clustered_docs = {cluster_id: [] for cluster_id in set(clusters) if cluster_id != -1}
        print("cluster length", len(clusters))
        
        #with faiss k-means and hierarchical clustering with faiss
        # embeddings = np.array(embeddings)
    
        #retrieve the document_ids 
        ids = extract_feature_list(id_doc="True", jsonl_file_path="/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-docs/jsonl/msmarco-docs-id-contents.jsonl")

        ids = ids[0:2] #we want to test with only 2 documents for now
        print("ids length", len(ids))
        for doc, cluster_id in zip(ids, clusters):
            print("cluster_id:", cluster_id)
            if cluster_id != -1:  # Ignore noise (-1)
                clustered_docs[cluster_id].append(ids)
        print("done clustering")
        print("length of clustered docs", len(clustered_docs))
        #display cluster result
        for cluster_id, docs in clustered_docs.items():
            print("docs in clustered_docs", docs)
            print(f"\nCluster {cluster_id}:")
            print(f" - Representative Keywords: {extract_keywords(docs)}")
            for doc in docs[:3]:  
                print(f"   - {doc[:200]}...")  
        
def extract_keywords(docs, top_n=5):
    #using tfidf weights to extract topic for each cluster
    vectorizer = TfidfVectorizer(max_features=5000)  
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = tfidf_scores.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in top_indices]
    
# def process_json_item(item):
#     features = list(item.keys())
#     print(f"item {item}")
#     print(f"features {features}")


#trying out Dask but it seems complicated
# def read_file_dask(file_path):
#     df = dd.read_json(file_path, lines=True, compression='gzip', blocksize='5MB')
    
#     #checking the number of partitions
#     # Get the number of partitions
#     num_partitions = df.npartitions
#     print(f"The DataFrame has {num_partitions} partitions.")
    
#     #checking the length of each partition
#     row_counts = df.map_partitions(len).compute()
#     print(f"Row counts per partition: {row_counts}")
    
#     specific_partition = df.get_partition(0).map_partitions(lambda df: df.apply(process_json_item, axis=1))
    
#     # Compute the result for that partition
#     result = specific_partition.compute()
#     print(f"result {result}")

def main():
    # api = HfApi()
    # api.login(token="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")
    # file_path = '/ivi/ilps/datasets/clueweb22/disk1/txt/en/en00/en0000/en0000-00.json.gz'
    # read_file(file_path)
    # read_file_dask(file_path)
    
    # file_path = '/ivi/ilps/datasets/clueweb22/disk1/txt/en/en33/en3301/en3301-79.json.gz'
    # read_file(file_path)
    
    #reading the file dynamically
    # read_file_dynamic(file_path, start_line=10000, num_lines=5)
    # directory_path = '/ivi/ilps/datasets/clueweb22/disk1/txt/en'
    # file_count = count_json_gz_files(directory_path)
    # print(f"Total number of .json.gz files: {file_count}")
    # records_count = count_json_gz_files_records(directory_path)
    
    # for file_path, count in records_count.items():
    #     print(f"{file_path}: {count} records")
    # total_json_item = sum(records_count.values())
    # print(f"Total number of records: {total_json_item}")
    
    
    #indexing MSMarco documents and cluster them 
    # input_dir_path = "/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-docs/tsv"
    # output_dir_path = "/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-docs/jsonl"
    # index_collection(input_dir_path=input_dir_path, convert_pyserini=True, input_filename="msmarco-docs-id-contents.tsv", output_dir_path=output_dir_path)
    
    #cluster the pyserini index
    print("clustering and topic extraction starting point")
    clustering_and_topic_extraction()
    
    #testing index reader
    # index_path = "/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-docs/jsonl/pyserini-index"  
    # index_reader = IndexReader(index_path)
    # doc_vector = index_reader.get_document_vector('D1555982')
    # print("doc vector: ", doc_vector)
    
    # term_positions = index_reader.get_term_positions('D1555982')
    # print("term positions: ", term_positions)
    
if __name__ == "__main__":
    print("start here")
    # hf_key = os.environ.get('HF_KEY')
    # if not hf_key:
    #     raise EnvironmentError("HF_KEY environment variable is not set")
    # login(token=hf_key)
    # main()
    
    # from datasets import load_dataset

    # docs = load_dataset('irds/clueweb12', 'docs', trust_remote_code=True)
    # # for record in docs:
    # #     record # {'doc_id': ..., 'url': ..., 'date': ..., 'http_headers': ..., 'body': ..., 'body_content_type': ...}
    from ClueWeb22Api import ClueWeb22Api
    from ClueWeb22Api import AnnotateHtml_pb2
    from ClueWeb22Api import AnnotateHtmlApi
    cw22id = "clueweb22-en0000-00-00004"
    root_path = "home/gpoerwa"
    clueweb_api = ClueWeb22Api(cw22id, root_path)
    clueweb_api.get_node_features_with_text(is_primary=True)
