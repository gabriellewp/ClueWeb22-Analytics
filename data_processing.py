import sys
from bs4 import BeautifulSoup, builder
from numba import jit
import csv, gzip
    
def extract_id_content_cw09():
    import gzip
    from warcio.archiveiterator import ArchiveIterator
    
    results = []
    with gzip.open('/ivi/ilps/datasets/clueweb09/ClueWeb09_English_1/en0000/00.warc.gz') as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == 'response':
                record_id = record.rec_headers.get_header('WARC-Record-ID')
                content = record.content_stream().read().decode('utf-8')
                print('id', id)
                print('content', content)
                results.append({'id': record_id, 'content': content})
    return results

def parse_html_with_fallback(html_content):
    parsers = ['html.parser', 'lxml', 'html5lib']
    for parser in parsers:
        try:
            print(f"Trying parser: {parser}")
            soup = BeautifulSoup(html_content, parser)
            
            return soup
        except Exception as e:
            print(f"Parser {parser} failed: {e}")

    return None


def extract_id_content_cw09_anotherway(warc_file_path):
    import gzip
    extracted_data = []
    warc_trec_id, content_div_text = None, None
    i = 0
    with gzip.open(warc_file_path, 'rt', encoding='utf-8', errors='ignore') as warc_file:
        # Read the WARC file line by line
        buffer = []
        # counter=0
        for line in warc_file:
            #print('line', line)
            if line.startswith("WARC/") and buffer:
                # print(counter)
                # counter+=1
                # # Parse WARC headers
                # if line.startswith("WARC-"):
                #     key, value = line.split(":", 1)
                #     print("key", key)
                #     print("value", value)
                # elif line.strip():  # Add non-header lines to the buffer
                #     buffer.append(line)
                # Process the current HTML content
                
                html_content = "\n".join(buffer)
                soup = parse_html_with_fallback(html_content)
                #print('soup is', soup)
                # if soup is None:
                #     print('soup is none')
                #     continue
                # Extract specific fields
                title = soup.title.string if soup.title else None
                meta_category, meta_category_content = None, None
                meta_category = soup.find('meta', attrs={'name': 'category'})
                print("meta_category", meta_category)
                if meta_category is not None and 'content' in meta_category.attrs:
                    meta_category_content = meta_category['content']
                
                headers = {
                    'h1': [h.get_text(strip=True) for h in soup.find_all('h1')],
                    'h2': [h.get_text(strip=True) for h in soup.find_all('h2')],
                    'h3': [h.get_text(strip=True) for h in soup.find_all('h3')],
                    'h4': [h.get_text(strip=True) for h in soup.find_all('h4')],
                }
                content_div = soup.find('div', {'id': 'content'})

                if content_div:
                    # Extract text from the div and all its descendants
                    content_div_text = content_div.get_text(separator="\n", strip=True)
                    content_div_text = content_div_text.replace("'", "").replace('"', "").replace('\n', " ")

                else:
                    print("No <div id='content'> found.")
                # temp['id'] = warc_trec_id
                # temp['title'] = title
                # temp['meta_category'] = meta_category_content
                # temp['headers'] = headers
                content = f"{title or ''} {' '.join(headers['h1'])} {' '.join(headers['h2'])} {' '.join(headers['h3'])} {' '.join(headers['h4'])}"
                # print('content after filling headers', content)
                if content_div_text:
                    content += f" {content_div_text}"
                if meta_category_content is not None:
                    content += f" {meta_category_content}"
                if (warc_trec_id is not None):
                    # print('warc_trec_id 2', warc_trec_id)
                    # print('content 2', content)
                    new_data = {
                        'id': warc_trec_id,
                        'contents': content
                    }
                    extracted_data.append(new_data)
                    # print('new_data', new_data)
                    warc_trec_id = None
                        
                    
                # extracted_data.append({
                #     'id': warc_trec_id,
                #     'title': title,
                #     'meta_category': meta_category_content,
                #     'headers': headers,
                # })
                
                #combine all the extracted data into a single string
                # print('id is ',warc_trec_id, 'contnet is', content)
                
                # Reset the buffer for the next WARC record

                buffer = []
                # content = None
            #happens with current buffer
            if line.startswith("WARC-TREC-ID:"):
                warc_trec_id = line.split(":", 1)[1].strip()
                # print('content here', content)
                # if (warc_trec_id is not None) and (content is not None):
                #     print('warc_trec_id', warc_trec_id)
                #     print('content', content)
                #     new_data = {
                #         'id': warc_trec_id,
                #         'content': content
                #     }
                #     extracted_data.append(new_data)
                #     print('new_data', new_data)
                #     content = None
                #     buffer = []
                # recorded = False
            
            if line.strip():  # Ignore blank lines
                buffer.append(line)  # Add non-blank lines to the buffer

    return extracted_data

def process_warc_files(warc_dir_path):
    import os
    import gzip
    import json

    jsonl_dir_path = warc_dir_path.replace('/datasets/clueweb09/', '/datasets/clueweb09/jsonl/')

    for root, _, files in os.walk(warc_dir_path):
        for file in files:
            if file.endswith('.warc.gz'):
                warc_file_path = os.path.join(root, file)

                # Transform the output path
                relative_path = os.path.relpath(root, warc_dir_path)
                jsonl_filename = file.replace("warc.gz", "jsonl")
                jsonl_output_dir = os.path.join(jsonl_dir_path, relative_path)
                os.makedirs(jsonl_output_dir, exist_ok=True)
                jsonl_output_path = os.path.join(jsonl_output_dir, f"{jsonl_filename}")

                # Process the WARC file
                print(f"Processing {warc_file_path}...")
                extracted_data = extract_id_content_cw09_anotherway(warc_file_path)

                with open(jsonl_output_path, 'w', encoding='utf-8') as jsonl_file:
                    for record in extracted_data:
                        jsonl_file.write(json.dumps(record) + '\n')

                print(f"Saved JSONL to {jsonl_output_path}")

    
    
def unsupervised_clustering_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import NMF

    # Step 1: Combine documents and tags
    corpus = documents + tags  # Treat documents and tags as part of a single dataset

    # Step 2: Extract TF-IDF features
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Step 3: Topic modeling (e.g., NMF)
    num_topics = 5  # Choose the number of latent topics
    nmf_model = NMF(n_components=num_topics, random_state=42)
    topic_distributions = nmf_model.fit_transform(tfidf_matrix)

    # Step 4: Separate document and tag representations
    doc_topics = topic_distributions[:len(documents)]
    tag_topics = topic_distributions[len(documents):]

    # Step 5: Cluster documents based on topics
    clustering_model = KMeans(n_clusters=5, random_state=42)
    labels = clustering_model.fit_predict(doc_topics)

    print("Cluster labels:", labels)
    
def load_mmws_hashid_to_list(tsv_file_path):
    hashid_mmws_list = []
    with open(tsv_file_path, 'r', newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            if len(row) >= 1:  # Ensure there is at least one column
                hashid_mmws_list.append(row[0])
    
    return hashid_mmws_list
       
def write_list_to_file(file_path, data_list):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(f"{item}\n")     
            
def translate_hash_cwid(path_to_file, cw22_hash_files_list, output_path_file):
    hashid_mmws_list = load_mmws_hashid_to_list('/ivi/ilps/datasets/MSMarco-Web-Search/100M/doc_hash_mapping.tsv')
    result_dict = []
    print("got into the translate hash cwid")
    for cw22_hash_file in cw22_hash_files_list:
        print(f"checking file {cw22_hash_file}")
        with gzip.open(f"{path_to_file}{cw22_hash_file}",  'rt', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                # parts = line.strip().split(',')
                if row[0] in hashid_mmws_list:
                    result_dict.append(row[1])
    
    write_list_to_file(output_path_file, result_dict)
    #write back to file
    
     
if __name__ == "__main__":
    # data = extract_id_content_cw09()
    # for item in data: 
    #     print(f"ID: {item['id']}\nContent: {item['content']}\n")
    print("i am here")
    args = sys.argv
    print(f"args: {args}")
    # Path to your WARC file
    # warc_file_path = '/ivi/ilps/datasets/clueweb09/ClueWeb09_English_1/en0000/'
    # warc_dir_path = str(args[1])
    # # Extract data
    # data = extract_id_content_cw09_anotherway(warc_file_path)
    # jsonl_dir_path = '/ivi/ilps/datasets/clueweb09/jsonl/ClueWeb09_English_1/en0000/'
    # process_warc_files(warc_dir_path)
    # Print results
    # for i, entry in enumerate(data, 1):
    #     # print(f"Record {i}:")
    #     # print(f"Title: {entry['title']}")
    #     # print(f"Meta Category: {entry['meta_category']}")
    #     # print("Headers:")
    #     # for level, texts in entry['headers'].items():
    #     #     print(f"  {level.upper()}: {', '.join(texts) if texts else 'None'}")
    #     # print("\n" + "-"*50 + "\n")
        
    #     print('data item', entry)
    # Save to JSONL file
    files_group = str(args[1])
    print(f"files_group: {files_group}")
    files_group_dict = {"g1": ['ClueWeb22-ID_URL-Hash_map_00.csv.gz', 'ClueWeb22-ID_URL-Hash_map_01.csv.gz', 'ClueWeb22-ID_URL-Hash_map_02.csv.gz', 'ClueWeb22-ID_URL-Hash_map_03.csv.gz', 'ClueWeb22-ID_URL-Hash_map_04.csv.gz', 'ClueWeb22-ID_URL-Hash_map_05.csv.gz', 'ClueWeb22-ID_URL-Hash_map_06.csv.gz', 'ClueWeb22-ID_URL-Hash_map_07.csv.gz', 'ClueWeb22-ID_URL-Hash_map_08.csv.gz', 'ClueWeb22-ID_URL-Hash_map_09.csv.gz'],
                        "g2": ['ClueWeb22-ID_URL-Hash_map_10.csv.gz', 'ClueWeb22-ID_URL-Hash_map_11.csv.gz', 'ClueWeb22-ID_URL-Hash_map_12.csv.gz', 'ClueWeb22-ID_URL-Hash_map_13.csv.gz', 'ClueWeb22-ID_URL-Hash_map_14.csv.gz', 'ClueWeb22-ID_URL-Hash_map_15.csv.gz', 'ClueWeb22-ID_URL-Hash_map_16.csv.gz', 'ClueWeb22-ID_URL-Hash_map_17.csv.gz', 'ClueWeb22-ID_URL-Hash_map_18.csv.gz', 'ClueWeb22-ID_URL-Hash_map_19.csv.gz'],
                        "g3": ['ClueWeb22-ID_URL-Hash_map_20.csv.gz', 'ClueWeb22-ID_URL-Hash_map_21.csv.gz', 'ClueWeb22-ID_URL-Hash_map_22.csv.gz', 'ClueWeb22-ID_URL-Hash_map_23.csv.gz', 'ClueWeb22-ID_URL-Hash_map_24.csv.gz', 'ClueWeb22-ID_URL-Hash_map_25.csv.gz', 'ClueWeb22-ID_URL-Hash_map_26.csv.gz', 'ClueWeb22-ID_URL-Hash_map_27.csv.gz', 'ClueWeb22-ID_URL-Hash_map_28.csv.gz', 'ClueWeb22-ID_URL-Hash_map_29.csv.gz'],
                        "g4": ['ClueWeb22-ID_URL-Hash_map_30.csv.gz', 'ClueWeb22-ID_URL-Hash_map_31.csv.gz', 'ClueWeb22-ID_URL-Hash_map_32.csv.gz', 'ClueWeb22-ID_URL-Hash_map_33.csv.gz', 'ClueWeb22-ID_URL-Hash_map_34.csv.gz', 'ClueWeb22-ID_URL-Hash_map_35.csv.gz', 'ClueWeb22-ID_URL-Hash_map_36.csv.gz', 'ClueWeb22-ID_URL-Hash_map_37.csv.gz', 'ClueWeb22-ID_URL-Hash_map_38.csv.gz', 'ClueWeb22-ID_URL-Hash_map_39.csv.gz'],
                        "g5": ['ClueWeb22-ID_URL-Hash_map_40.csv.gz', 'ClueWeb22-ID_URL-Hash_map_41.csv.gz', 'ClueWeb22-ID_URL-Hash_map_42.csv.gz', 'ClueWeb22-ID_URL-Hash_map_43.csv.gz', 'ClueWeb22-ID_URL-Hash_map_44.csv.gz', 'ClueWeb22-ID_URL-Hash_map_45.csv.gz', 'ClueWeb22-ID_URL-Hash_map_46.csv.gz', 'ClueWeb22-ID_URL-Hash_map_47.csv.gz', 'ClueWeb22-ID_URL-Hash_map_48.csv.gz', 'ClueWeb22-ID_URL-Hash_map_49.csv.gz']}
    # translate_hash_cwid('/ivi/ilps/datasets/clueweb22/disk1/ClueWeb22-ID_URL-Hash_maps/', files_group_dict[files_group], f'/ivi/ilps/datasets/MSMarco-Web-Search/100M/cwid_list_{files_group}.txt')