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

def detect_topic_tags(directory_path):
    """
    Detect if there is a topic tag in a json.gz file. 
    """
    json_gz_files = glob.glob(os.path.join(directory_path, '**', '*.json.gz'), recursive=True)
    
    for file_path in json_gz_files:
        with gzip.open(file_path, 'rt') as f:
            first_line = f.readline()
            if first_line:
                json_item = json.loads(first_line)
                feature_names = list(json_item.keys())
                print(f"File: {file_path}")
                print(f"Feature names: {feature_names}")

    
def extract_feature_list_jsonl(jsonl_file_path=None, feature_name=None):
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
    extracted_feature = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)  # Parse each line as a JSON object
            if feature_name in data:  # Check if the 'contents' field exists
                extracted_feature.append(data[feature_name])
    return extracted_feature

def extract_feature_list_jsonl_gz(jsonl_gz_file_path=None, feature_name=None):
    extracted_feature = []
    with gzip.open(jsonl_gz_file_path, 'rt') as f:
        for line in f:
            data = json.loads(line)
            if feature_name in data:
                extracted_feature.append(data[feature_name])
    return extracted_feature

def extract_feature_list_tsv(tsv_file_path=None, col_number=None):
    """
    Extracts the content from the tsv file. Where the tsv file is usually dont have a header, we select the feature based on the column number. 
    
    """
    extracted_feature = []
    with open(tsv_file_path, 'r', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            if col_number < len(row):
                extracted_feature.append(row[col_number])
    return extracted_feature

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

def faiss_clustering_streaming():
    import faiss
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.cluster.hierarchy import fcluster
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    print("in a faiss_clustering_streaming function")
    d = 128
    n_clusters = 100
    chunk_size = 100000  # number of documents to process at a time, in clueweb22 per json.gz file will contain 100000 documents records
    directory_path = "/ivi/ilps/datasets/clueweb22/disk1/txt/en/"
    
    # initialized FAISS index with reduced dimensionality
    index = faiss.IndexFlatL2(d)
    
    # initialize PCA
    pca = PCA(n_components=d)
    
    # intialize the embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # initialize lists to store document IDs and cluster assignments
    all_ids = []
    all_cluster_assignments = []
    # List all .json.gz files in the directory
    json_gz_files = glob.glob(os.path.join(directory_path, '**', '*.json.gz'), recursive=True)
    print("length of json_gz_files", len(json_gz_files))
    for file_path in json_gz_files:
        documents = extract_feature_list_jsonl_gz(jsonl_gz_file_path=file_path, feature_name="Clean-Text")
        ids = extract_feature_list_jsonl_gz(jsonl_gz_file_path=file_path, feature_name="ClueWeb22-ID")
        
        if not documents:
            break
        print(f"Processing {file_path} and number of documents: {len(documents)}")
        embeddings = embedding_model.encode(documents)
        reduced_embeddings = pca.fit_transform(embeddings)
        index.add(reduced_embeddings)
        all_ids.extend(ids)
    
    # Perform clustering using KMeans
    kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=True)
    
    # Retrieve all vectors stored in the index
    vectors = index.reconstruct_n(0, index.ntotal)
    
    # Train the KMeans model using the retrieved vectors
    kmeans.train(vectors)
    
    # Get the cluster assignments for each vector
    cluster_assignments = kmeans.index.search(vectors, 1)[1].flatten()
    
    # Initialize a dictionary to hold documents per cluster
    clustered_docs = {i: [] for i in range(n_clusters)}
    
    # Assign documents to clusters
    for doc_id, cluster_id in zip(all_ids, cluster_assignments):
        clustered_docs[cluster_id].append(doc_id)
    
    print("Clustered documents:", clustered_docs)

    
def faiss_clustering(input_file_path=None, file_type=None):
    import faiss
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.cluster.hierarchy import fcluster
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    print("in a faiss_clustering function")
    d = 128
    documents, ids = [], []
    
    if file_type == "jsonl":
        documents = extract_feature_list_jsonl(jsonl_file_path=input_file_path, feature_name="contents")
        ids = extract_feature_list_jsonl(jsonl_file_path=input_file_path, feature_name="id")
    elif file_type=="tsv":
        documents = extract_feature_list_tsv(tsv_file_path=input_file_path, col_number=1)
        ids = extract_feature_list_tsv(tsv_file_path=input_file_path, col_number=0)
    print(f"Number of documents: {len(documents)}")
    print(f"Number of ids: {len(ids)}")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(documents).astype(np.float16)
    # embeddings = embedding_model.encode(documents[0:4000])
    
    # reduce the dimensionality with PCA
    d_original = embeddings.shape[1]
    d_reduced = 128
    pca = PCA(n_components=d_reduced)
    reduced_embeddings = pca.fit_transform(embeddings)
    print("Reduced embeddings shape:", reduced_embeddings.shape)
    
    # index = faiss.IndexFlatL2(d_reduced)
    # index.add(reduced_embeddings)
    
    n_clusters = 10
    kmeans = faiss.Kmeans(d_reduced, n_clusters, niter=20, verbose=True)
    kmeans.train(reduced_embeddings)
    cluster_assignments = kmeans.index.search(reduced_embeddings, 1)[1].flatten()
    
    clustered_docs = {i: [] for i in range(n_clusters)}
    
    # for doc_id, cluster_id in zip(ids[:100], cluster_assignments):
    for doc_id, cluster_id in zip(ids, cluster_assignments):
        clustered_docs[cluster_id].append(doc_id)
        
    print("Clustered documents:", clustered_docs)
    # cluster_similarity_distribution_pairwise(reduced_embeddings= reduced_embeddings, cluster_id=0, cluster_assignments=cluster_assignments)
    
def compute_pairwise_similarities_in_batches(embeddings, batch_size=1000):
    import numpy as np
    from scipy.spatial.distance import cosine
    n = len(embeddings)
    similarity_scores = []
    
    for i in range(0, n, batch_size):
        for j in range(i+1, n, batch_size):
            batch1 = embeddings[i:i+batch_size]
            batch2 = embeddings[j:j+batch_size]
            similarities = 1 - np.array([[cosine(a,b)for b in batch2] for a in batch1])
            similarity_scores.extend(similarities.flatten())
            
    return np.array(similarity_scores)

def cluster_similarity_distribution_pairwise(reduced_embeddings=None, cluster_id=None, cluster_assignments=None):
    cluster_embeddings = reduced_embeddings[cluster_id == cluster_assignments]
    
    if len(cluster_embeddings) > 1:
        similarity_scores = compute_pairwise_similarities_in_batches(cluster_embeddings)
        print(f"Cluster {cluster_id} - Similarity Distribution:")
        print(f" - Mean: {similarity_scores.mean()}")
        print(f" - Median: {np.median(similarity_scores)}")
        print(f" - Max: {similarity_scores.max()}")
        print(f" - Min: {similarity_scores.min()}")
        print(f" - 75th Percentile: {np.percentile(similarity_scores, 75)}")
        print(f" - 90th Percentile: {np.percentile(similarity_scores, 90)}")
        print(f" - 95th Percentile: {np.percentile(similarity_scores, 95)}")
        print(f" - 99th Percentile: {np.percentile(similarity_scores, 99)}")
    else:
        print(f"Cluster {cluster_id} - Similarity Distribution: Not enough documents to compute pairwise similarities.")

def cluster_similarity_distribution(reduced_embeddings=None, cluster_id=None, cluster_assignments=None):
    #this similarity distribution computation is using full similarity matrix
    reduced_embeddings = reduced_embeddings.astype(np.float16)
    cluster_embeddings = reduced_embeddings[cluster_id == cluster_assignments]

    if len(cluster_embeddings) > 1:
        similarity_matrix = np.dot(cluster_embeddings, cluster_embeddings.T)
        similarity_scores = similarity_matrix[np.triu_indices(len(cluster_embeddings), k=1)]
        print(f"Cluster {cluster_id} - Similarity Distribution:")
        print(f" - Mean: {similarity_scores.mean()}")
        print(f" - Median: {np.median(similarity_scores)}")
        print(f" - Max: {similarity_scores.max()}")
        print(f" - Min: {similarity_scores.min()}")
        print(f" - 1th Percentile: {np.percentile(similarity_scores,1)}")
        print(f" - 25th Percentile: {np.percentile(similarity_scores,25)}")
        print(f" - 75th Percentile: {np.percentile(similarity_scores, 75)}")
        print(f" - 90th Percentile: {np.percentile(similarity_scores, 90)}")
        print(f" - 95th Percentile: {np.percentile(similarity_scores, 95)}")
        print(f" - 99th Percentile: {np.percentile(similarity_scores, 99)}")
    # Compute pairwise cosine similar    
    

    

def hierarchical_faiss_clustering_and_topic_extraction():
    """
    This is another idea of clustering using hierarchical approach. 
    """
    print("before loading library for hierarchical clustering")
    import faiss
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.cluster.hierarchy import fcluster
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    
    #retrieving document list
    #retrieving doc_ids list
    print("i am here at the hierarchical clustering")
    d = 128  # Reduced embedding dimension
    n_clusters = 100000  # Number of clusters
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, n_clusters, 16, 8)
    documents = extract_feature_list_jsonl(content_doc="True", jsonl_file_path="/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-docs/jsonl/msmarco-docs-id-contents.jsonl")
    ids = extract_feature_list_jsonl(id_doc="True", jsonl_file_path="/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-docs/jsonl/msmarco-docs-id-contents.jsonl")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(documents[0:4000])
    print("the embeddings shape", embeddings.shape)
    print("number of embeddings", len(embeddings))
    
    #reduce dimensionality by PCA
    d_original = embeddings.shape[1]
    d_reduced = 128
    pca = PCA(n_components=d_reduced)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    print("Reduced embeddings shape:", reduced_embeddings.shape)

    #initialize FAISS with the reduced dimensionality
    n_clusters = 10
    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(d_reduced), d_reduced, n_clusters, 16, 8)
    #we use the first 1500 documents then add the rest afterwards
    index.use_direct_map = True
    index.train(reduced_embeddings[:1500])    
    index.add(reduced_embeddings)
    
    #extract cluster centroids
    # centroids = faiss.vector_to_array(index.quantizer.xb).reshape(n_clusters, d_reduced)
    # print(f"Centroids shape: {centroids.shape}")  # (n_clusters, d)
    # for i in range(n_clusters):
    #     centroids[i] = index.reconstruct(i)
    # print(f"Centroids shape: {centroids.shape}")
    centroids = np.zeros((n_clusters, d_reduced), dtype=np.float32)
    for i in range(n_clusters):
        centroids[i] = index.reconstruct(i)
    print(f"Centroids shape: {centroids.shape}")
    
    # Perform hierarchical clustering on the centroids
    linkage_matrix = linkage(centroids, method='ward')
    k_hierarchical_clusters = 10
    cluster_assignments = fcluster(linkage_matrix, t=k_hierarchical_clusters, criterion='maxclust')
    
    # Initialize a dictionary to hold documents per cluster
    clustered_docs = {i: [] for i in range(1, k_hierarchical_clusters + 1)}

    # Assign documents to clusters
    for doc_id, cluster_id in zip(ids[:100], cluster_assignments):
        clustered_docs[cluster_id].append(doc_id)
    print("Clustered documents:", clustered_docs)
    
    
    # Display the result
    # for cluster_id, docs in clustered_docs.items():
    #     print(f"Cluster {cluster_id}:")
    #     print(f" - Representative Keywords: {extract_keywords(docs)}")
    #     for doc in docs[:3]:  # Display top 3 docs per cluster
    #         print(f"   - {doc[:10]}...")  # Document preview

def hashing_clustering():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from datasketch import MinHash, MinHashLSH
    import numpy as np

    # Sample document collection
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps over a sleepy dog.",
        "The history of natural language processing is fascinating.",
        "Artificial intelligence is a rapidly evolving field.",
        "Machine learning powers many applications in AI.",
        "Foxes are quick and dogs are usually lazy."
    ]

    # Step 1: Convert documents to TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Step 2: Create MinHash signatures for each document
    def create_minhash_vector(vector, num_perm=128):
        """Convert a sparse TF-IDF vector to a MinHash signature."""
        minhash = MinHash(num_perm=num_perm)
        for idx in vector.nonzero()[1]:  # Extract non-zero indices
            minhash.update(str(idx).encode('utf-8'))
        return minhash

    minhashes = [create_minhash_vector(tfidf_matrix[i]) for i in range(tfidf_matrix.shape[0])]

    # Step 3: Create an LSH index
    lsh = MinHashLSH(threshold=0.5, num_perm=128)  # Set threshold for similarity
    for i, minhash in enumerate(minhashes):
        lsh.insert(f"doc_{i}", minhash)

    # Step 4: Query similar documents
    clusters = {}
    for i, minhash in enumerate(minhashes):
        similar_docs = lsh.query(minhash)
        clusters[f"doc_{i}"] = similar_docs

    # Display clustering results
    print("Document Clusters:")
    for doc, cluster in clusters.items():
        print(f"{doc}: {cluster}")

    
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
    # clustering_and_topic_extraction()
    # hierarchical_faiss_clustering_and_topic_extraction()
    # directory_path = '/ivi/ilps/datasets/clueweb22/disk1/txt/en'
    # detect_topic_tags(directory_path)
    
    # faiss_clustering(input_file_path = "/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-docs/jsonl/msmarco-docs-id-contents.jsonl", file_type="jsonl")
    # faiss_clustering(input_file_path = "/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-passage/collection.tsv", file_type="tsv")
    # faiss_clustering_streaming()
    hashing_clustering()
    #testing index reader
    # index_path = "/ivi/ilps/personal/gpoerwa/msmarco_ir2/collections/msmarco-docs/jsonl/pyserini-index"  
    # index_reader = IndexReader(index_path)
    # doc_vector = index_reader.get_document_vector('D1555982')
    # print("doc vector: ", doc_vector)
    
    # term_positions = index_reader.get_term_positions('D1555982')
    # print("term positions: ", term_positions)
    
if __name__ == "__main__":
    print("start here")
    hf_key = os.environ.get('HF_KEY')
    if not hf_key:
        raise EnvironmentError("HF_KEY environment variable is not set")
    login(token=hf_key)
    # main()
    
    import ir_datasets
    dataset = ir_datasets.load('clueweb09')
    first_doc = next(dataset.docs_iter())
    
    print(first_doc)
        
