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

def clustering_and_topic_extraction():
    index_path = "/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-docs/jsonl/pyserini-index"  
    searcher = SimpleSearcher(index_path)    
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v1')
    index_reader = IndexReader(index_path)
    #extracting doc embeddings
    # doc_ids = searcher.docids()
    # documents = [searcher.doc(doc_id).raw() for doc_id in doc_ids]
    
    # doc_ids = index_reader.object.get_document_ids()
    # documents = [searcher.doc(doc_id).raw() for doc_id in doc_ids if searcher.doc(doc_id) is not None]

    #replace the documents with raw string from dataset
    documents = extract_feature_list(content_doc="True", jsonl_file_path="/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-docs/jsonl/msmarco-docs-id-contents.jsonl")
    print("documents length", len(documents))
    embeddings = embedding_model.encode(documents)
    
    print("done obtaining embeddings")
    #clustering with dbscan
    dbscan = DBSCAN(eps=0.6, min_samples=1000, metric='cosine') 
    clusters = dbscan.fit_predict(embeddings)
    clustered_docs = {cluster_id: [] for cluster_id in set(clusters) if cluster_id != -1}
    
    #retrieve the document_ids 
    ids = extract_feature_list(id_doc="True", jsonl_file_path="/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-docs/jsonl/msmarco-docs-id-contents.jsonl")
    print("ids length", len(ids))
    for doc, cluster_id in zip(ids, clusters):
        if cluster_id != -1:  # Ignore noise (-1)
            clustered_docs[cluster_id].append(ids)
    print("done clustering")
            
    #display cluster result
    for cluster_id, docs in clustered_docs.items():
        print(f"\nCluster {cluster_id}:")
        print(f" - Representative Keywords: {extract_keywords(docs)}")
        for doc in docs[:3]:  
            print(f"   - {doc[:200]}...")  
        
def extract_keywords(docs, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
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
    login(token="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")
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
    clustering_and_topic_extraction()
    
    #testing index reader
    # index_path = "/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-docs/jsonl/pyserini-index"  
    # index_reader = IndexReader(index_path)
    # doc_vector = index_reader.get_document_vector('D1555982')
    # print("doc vector: ", doc_vector)
    
    # term_positions = index_reader.get_term_positions('D1555982')
    # print("term positions: ", term_positions)
    
if __name__ == "__main__":
    main()