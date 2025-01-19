import sys
from bs4 import BeautifulSoup, builder
from numba import jit
import csv, gzip
import os
import json

def save_to_file_list_mode(path2file, output_file, new_data):

    #the data is in list format
    """    Saves new data to a JSON file in list format. If the file does not exist, it creates a new one.
        If the file exists but is empty or not in valid JSON format, it initializes it with an empty list.
        The function appends the new data to the list and writes it back to the file.
        path2file (str): The directory path where the file is located.
        output_file (str): The name of the output file.
        new_data (dict): The new data to be appended to the file.

    Args:
        path2file (_type_): _description_
        output_file (_type_): _description_
        new_data (_type_): _description_
    """
    filename = path2file+output_file
    list_obj = []
    print("new_data", new_data)
    if not os.path.isfile(filename):
        print("The file does not exist. Creating a new file.")
        with open(filename, 'w') as fp:
            json.dump(list_obj, fp)
    else:
        try:
            with open(filename) as fp:
                list_obj = json.load(fp)
                if not list_obj:
                    print("The JSON file is empty.")
                else:
                    print("The JSON file is not empty.")
        except JSONDecodeError:
            print("The file is empty or not in valid JSON format.")
    
    modified_dict = {key: value.replace("'", '\"') if isinstance(value, str) else value for key, value in new_data.items()}
    # list_obj.append(modified_dict)
    list_obj.append(new_data)
    print("new_data in the save to file list mode", new_data)
    with open(filename, 'w') as file:
        # json_line = json.dumps(list_obj)
        json.dump(list_obj, file, separators=(',\n',':'))

def load_mmws_hashid_to_list(tsv_file_path):
    hashid_mmws_list = []
    with open(tsv_file_path, 'r', newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            if len(row) >= 1:  # Ensure there is at least one column
                hashid_mmws_list.append(row[0])
    
    return hashid_mmws_list
       
def write_list_to_file(file_path, data_list):
    # print('writing file to a list')
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(f"{item}\n")     

def write_list_to_file_jsonl(file_path, data_list):
    # print('writing file to a list')
    file_mode = 'a' if os.path.exists(file_path) and os.path.getsize(file_path) > 0 else 'w'
    with open(file_path, file_mode, encoding='utf-8') as f:
        for item in data_list:
            json_line = json.dumps(item)
            f.write(f"{json_line}\n")
            
def translate_hash_cwid(path_to_file, cw22_hash_files_list, output_path_file):
    hashid_mmws_list = load_mmws_hashid_to_list('/ivi/ilps/datasets/MSMarco-Web-Search/100M/doc_hash_mapping.tsv')
    hashid_mmws_list = set(hashid_mmws_list)
    result_dict = []
    hash_list = []
    print("got into the translate hash cwid")

    for cw22_hash_file in cw22_hash_files_list:
        print(f"checking file {cw22_hash_file}")
        with gzip.open(f"{path_to_file}{cw22_hash_file}",  'rt', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                # parts = line.strip().split(',')
                if row[0] in hashid_mmws_list:
                    result_dict.append(row[1])
                    hash_list.append(row[0])
        output_filename = cw22_hash_file.split('.')[0]
    output_hash_list = f'/ivi/ilps/datasets/MSMarco-Web-Search/100M/result-sets/cwid_disk2_hashlist_{output_filename}.txt'
    output_path_file = f'/ivi/ilps/datasets/MSMarco-Web-Search/100M/result-sets/cwid_disk2_list_{output_filename}.txt'
    write_list_to_file(output_path_file, result_dict)
    write_list_to_file(output_hash_list, hash_list)
    
    
def check_matches():
    hashid_mmws_list = set(load_mmws_hashid_to_list('/ivi/ilps/datasets/MSMarco-Web-Search/100M/result-sets/merged_disk1_hashlist.txt'))
    hashid_mmws_checklist = set(load_mmws_hashid_to_list('/ivi/ilps/datasets/MSMarco-Web-Search/100M/result-sets/merged_disk2_hashlist.txt'))
    check_result = {}
    check_result[0] = 0
    check_result[1] = 0
    for item in hashid_mmws_list:
        if item in hashid_mmws_checklist:
            check_result[1] +=1
        else:
            check_result[0] +=1
    print("check result", check_result)


def count_record_per_language():
    from collections import defaultdict
    record_num = {}
    
    cluewebid_list = set(load_mmws_hashid_to_list('/ivi/ilps/datasets/MSMarco-Web-Search/100M/result-sets/merged_disk1_cluewebidlist.txt'))
    # Create a dictionary to count languages dynamically
    language_counts = defaultdict(int)

    # Extract and count language codes
    for record in cluewebid_list:
        # print(record)
        # Extract the language code from the record
        lang_code = record.split("-")[1][:2]  # Get the language part (e.g., 'de', 'en', 'fr')
        language_counts[lang_code] += 1

    # Convert defaultdict to a normal dictionary (optional)
    language_counts = dict(language_counts)

    # Output the result
    print(language_counts)  

def extract_english_ids():
    cluewebid_list = set(load_mmws_hashid_to_list('/ivi/ilps/datasets/MSMarco-Web-Search/100M/result-sets/merged_disk1_cluewebidlist.txt'))
    english_ids = []
    
    for record in cluewebid_list:
        lang_code = record.split("-")[1][:2]
        if lang_code == "en":
            english_ids.append(record)
    write_list_to_file('/ivi/ilps/datasets/MSMarco-Web-Search/100M/result-sets/merged_disk1_cluewebidlist_en.txt', english_ids)
    
def extract_cw22_docs(base_dir, slice_begin, slice_end):
    import re
    #slice_end is exclusive
    print("i am here")
    cluewebid_en_list = load_mmws_hashid_to_list('/ivi/ilps/datasets/MSMarco-Web-Search/100M/result-sets/merged_disk1_cluewebidlist_en.txt')
    #the folder structure is from the disk1 /ivi/ilps/datasets/clueweb22/disk1/txt/txt/en, then inside we have folder en00 to en49, and then inside of each this folder we have en0000 to en0046
    # the base_dir for english dataset should be /ivi/ilps/datasets/clueweb22/disk1/txt/txt/en/en<xy>
    # for dir_level_1 in os.listdir(base_dir):
    #     dir_level_1_path = os.path.join(base_dir, dir_level_1)
    
    # walk through all the possible paths
    # if os.path.isdir(base_dir):
    #     for dir_level_2 in os.listdir(base_dir):
    #         dir_level_2_path = os.path.join(base_dir, dir_level_2)
    #         # print(dir_level_2_path)
    #         for root, _, files in os.walk(dir_level_2_path):
    #             for filename in files:
    #                 #only looking  for a single file for each branch for now
    #                 file_path = os.path.join(root, filename)
    #                 if ".json.gz" in file_path and ".checksum" not in file_path:
    #                     print("file_path", file_path)
    #                     output_filename = filename.replace(".json.gz", ".jsonl")
    #                     output_path_file = f'/ivi/ilps/datasets/MSMarco-Web-Search/100M/cw22_en_53M/{output_filename}'
    #                     print("output_path_file", output_path_file)
    #                     data_list = read_extract_english_docs(file_path, cluewebid_en_list)
    #                     if len(data_list) > 0:
    #                         write_list_to_file_jsonl(output_path_file, data_list)
    
    #we look int the merged_disk1_cluewebidlist_en.txt first
    for cw_id in cluewebid_en_list[slice_begin:slice_end]:
        parts = cw_id.split('-')
        dir_level_1_id = parts[1][0:4]
        dir_level_2_id = parts[1][0:len(parts[1])]
        file_id = parts[2]
        complete_path = os.path.join(base_dir,f"{dir_level_1_id}/{dir_level_2_id}")
        # print("complete_path", complete_path)
        if os.path.isdir(complete_path):
            complete_file_path = f"{complete_path}/{dir_level_2_id}-{file_id}.json.gz"
            # print("complete file path", complete_file_path)
            if os.path.isfile(complete_file_path):
                data_list = read_extract_english_docs(complete_file_path, cw_id)
                write_list_to_file_jsonl(f"/ivi/ilps/datasets/MSMarco-Web-Search/100M/cw22_en_53M/{dir_level_1_id}-{dir_level_2_id}-{file_id}.jsonl", data_list)

            
                                
                        


                        
def read_extract_english_docs(path2file_input, cwid):
    """
    Read a JSONL file and return the parsed data. The parsed data is a list of dictionaries where it follows the feature name of each record in the jsonl. 
    I am avoiding using Pandas, but using array instead. 
    """
    parsed_data = []
    # print("in reading extract english docs")
    try:
        with gzip.open(path2file_input, 'rt', encoding='utf-8') as f:
            # Process each line as a separate JSON object
            # print("length of lines", len(lines))
            # data = json.load(f)
            for line in f:
                # print("line before strip", line)
                if line.strip():  # Check if the line is not empty
                    # print("the line", line)
                    json_item = json.loads(line.strip())
                    if json_item['ClueWeb22-ID'] == cwid:  
                        json_item['id'] = json_item.pop('ClueWeb22-ID')
                        json_item['contents'] = json_item.pop('Clean-Text')
                
                        # print("json data after replacement", line)
                        parsed_data.append(json_item)
                    else:
                        continue
                    # print(line)
                    # Process the JSON data as needed
                    # print(json_data)  # Example: Print the JSON data
                    # parsed_data.append(json_data)
                    # print("current length of the parsed data", len(parsed_data))
                else:
                    continue
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    return parsed_data

if __name__ == "__main__":
    print("i am here")
    args = sys.argv
    print(f"args: {args}")
    # files_group = args[1]
    # print(f"files_group: {files_group}")
    #job number 138664
    #job number with set 138970
    # files_group_dict = {"g1": ['ClueWeb22-ID_URL-Hash_map_00.csv.gz', 'ClueWeb22-ID_URL-Hash_map_01.csv.gz'], 
    #                     "g2": ['ClueWeb22-ID_URL-Hash_map_10.csv.gz', 'ClueWeb22-ID_URL-Hash_map_11.csv.gz'], 
    #                     "g3": ['ClueWeb22-ID_URL-Hash_map_20.csv.gz', 'ClueWeb22-ID_URL-Hash_map_21.csv.gz'], 
    #                     "g4": ['ClueWeb22-ID_URL-Hash_map_30.csv.gz', 'ClueWeb22-ID_URL-Hash_map_31.csv.gz'], 
    #                     "g5": ['ClueWeb22-ID_URL-Hash_map_40.csv.gz', 'ClueWeb22-ID_URL-Hash_map_41.csv.gz']}
    
    #job number 138665
    #job number with set 138975
    # files_group_dict = {"g1": ['ClueWeb22-ID_URL-Hash_map_02.csv.gz', 'ClueWeb22-ID_URL-Hash_map_03.csv.gz'], 
    #                     "g2": ['ClueWeb22-ID_URL-Hash_map_12.csv.gz', 'ClueWeb22-ID_URL-Hash_map_13.csv.gz'], 
    #                     "g3": ['ClueWeb22-ID_URL-Hash_map_22.csv.gz', 'ClueWeb22-ID_URL-Hash_map_23.csv.gz'], 
    #                     "g4": ['ClueWeb22-ID_URL-Hash_map_32.csv.gz', 'ClueWeb22-ID_URL-Hash_map_33.csv.gz'], 
    #                     "g5": ['ClueWeb22-ID_URL-Hash_map_42.csv.gz', 'ClueWeb22-ID_URL-Hash_map_43.csv.gz']}
    
    # #job number 138679
    #job number with set 138980
    # files_group_dict = {"g1": ['ClueWeb22-ID_URL-Hash_map_04.csv.gz', 'ClueWeb22-ID_URL-Hash_map_05.csv.gz'], 
    #                     "g2": ['ClueWeb22-ID_URL-Hash_map_14.csv.gz', 'ClueWeb22-ID_URL-Hash_map_15.csv.gz'], 
    #                     "g3": ['ClueWeb22-ID_URL-Hash_map_24.csv.gz', 'ClueWeb22-ID_URL-Hash_map_25.csv.gz'], 
    #                     "g4": ['ClueWeb22-ID_URL-Hash_map_34.csv.gz', 'ClueWeb22-ID_URL-Hash_map_35.csv.gz'], 
    #                     "g5": ['ClueWeb22-ID_URL-Hash_map_44.csv.gz', 'ClueWeb22-ID_URL-Hash_map_45.csv.gz']}
    
    # #job number 138700
    #job number with set 138985
    # files_group_dict = {"g1": ['ClueWeb22-ID_URL-Hash_map_06.csv.gz', 'ClueWeb22-ID_URL-Hash_map_07.csv.gz'], 
    #                 "g2": ['ClueWeb22-ID_URL-Hash_map_16.csv.gz', 'ClueWeb22-ID_URL-Hash_map_17.csv.gz'], 
    #                 "g3": ['ClueWeb22-ID_URL-Hash_map_26.csv.gz', 'ClueWeb22-ID_URL-Hash_map_27.csv.gz'], 
    #                 "g4": ['ClueWeb22-ID_URL-Hash_map_36.csv.gz', 'ClueWeb22-ID_URL-Hash_map_37.csv.gz'], 
    #                 "g5": ['ClueWeb22-ID_URL-Hash_map_46.csv.gz', 'ClueWeb22-ID_URL-Hash_map_47.csv.gz']}
    
    #job number 138695
    #job number with set 138990
    # files_group_dict = {"g1": ['ClueWeb22-ID_URL-Hash_map_08.csv.gz', 'ClueWeb22-ID_URL-Hash_map_09.csv.gz'], 
    #             "g2": ['ClueWeb22-ID_URL-Hash_map_18.csv.gz', 'ClueWeb22-ID_URL-Hash_map_19.csv.gz'], 
    #             "g3": ['ClueWeb22-ID_URL-Hash_map_28.csv.gz', 'ClueWeb22-ID_URL-Hash_map_29.csv.gz'], 
    #             "g4": ['ClueWeb22-ID_URL-Hash_map_38.csv.gz', 'ClueWeb22-ID_URL-Hash_map_39.csv.gz'], 
    #             "g5": ['ClueWeb22-ID_URL-Hash_map_48.csv.gz', 'ClueWeb22-ID_URL-Hash_map_49.csv.gz']}
    
    
    # files_group_dict = {"g1": ['ClueWeb22-ID_URL-Hash_map_00.csv.gz', 'ClueWeb22-ID_URL-Hash_map_01.csv.gz', 'ClueWeb22-ID_URL-Hash_map_02.csv.gz', 'ClueWeb22-ID_URL-Hash_map_03.csv.gz', 'ClueWeb22-ID_URL-Hash_map_04.csv.gz', 'ClueWeb22-ID_URL-Hash_map_05.csv.gz', 'ClueWeb22-ID_URL-Hash_map_06.csv.gz', 'ClueWeb22-ID_URL-Hash_map_07.csv.gz', 'ClueWeb22-ID_URL-Hash_map_08.csv.gz', 'ClueWeb22-ID_URL-Hash_map_09.csv.gz'],
    #                     "g2": ['ClueWeb22-ID_URL-Hash_map_10.csv.gz', 'ClueWeb22-ID_URL-Hash_map_11.csv.gz', 'ClueWeb22-ID_URL-Hash_map_12.csv.gz', 'ClueWeb22-ID_URL-Hash_map_13.csv.gz', 'ClueWeb22-ID_URL-Hash_map_14.csv.gz', 'ClueWeb22-ID_URL-Hash_map_15.csv.gz', 'ClueWeb22-ID_URL-Hash_map_16.csv.gz', 'ClueWeb22-ID_URL-Hash_map_17.csv.gz', 'ClueWeb22-ID_URL-Hash_map_18.csv.gz', 'ClueWeb22-ID_URL-Hash_map_19.csv.gz'],
    #                     "g3": ['ClueWeb22-ID_URL-Hash_map_20.csv.gz', 'ClueWeb22-ID_URL-Hash_map_21.csv.gz', 'ClueWeb22-ID_URL-Hash_map_22.csv.gz', 'ClueWeb22-ID_URL-Hash_map_23.csv.gz', 'ClueWeb22-ID_URL-Hash_map_24.csv.gz', 'ClueWeb22-ID_URL-Hash_map_25.csv.gz', 'ClueWeb22-ID_URL-Hash_map_26.csv.gz', 'ClueWeb22-ID_URL-Hash_map_27.csv.gz', 'ClueWeb22-ID_URL-Hash_map_28.csv.gz', 'ClueWeb22-ID_URL-Hash_map_29.csv.gz'],
    #                     "g4": ['ClueWeb22-ID_URL-Hash_map_30.csv.gz', 'ClueWeb22-ID_URL-Hash_map_31.csv.gz', 'ClueWeb22-ID_URL-Hash_map_32.csv.gz', 'ClueWeb22-ID_URL-Hash_map_33.csv.gz', 'ClueWeb22-ID_URL-Hash_map_34.csv.gz', 'ClueWeb22-ID_URL-Hash_map_35.csv.gz', 'ClueWeb22-ID_URL-Hash_map_36.csv.gz', 'ClueWeb22-ID_URL-Hash_map_37.csv.gz', 'ClueWeb22-ID_URL-Hash_map_38.csv.gz', 'ClueWeb22-ID_URL-Hash_map_39.csv.gz'],
    #                     "g5": ['ClueWeb22-ID_URL-Hash_map_40.csv.gz', 'ClueWeb22-ID_URL-Hash_map_41.csv.gz', 'ClueWeb22-ID_URL-Hash_map_42.csv.gz', 'ClueWeb22-ID_URL-Hash_map_43.csv.gz', 'ClueWeb22-ID_URL-Hash_map_44.csv.gz', 'ClueWeb22-ID_URL-Hash_map_45.csv.gz', 'ClueWeb22-ID_URL-Hash_map_46.csv.gz', 'ClueWeb22-ID_URL-Hash_map_47.csv.gz', 'ClueWeb22-ID_URL-Hash_map_48.csv.gz', 'ClueWeb22-ID_URL-Hash_map_49.csv.gz']}
    
    # print(files_group_dict[files_group])
    # translate_hash_cwid('/ivi/ilps/datasets/clueweb22/disk2/ClueWeb22-ID_URL-Hash_maps/', files_group_dict[files_group], f'/ivi/ilps/datasets/MSMarco-Web-Search/100M/result-sets/cwid_list_{files_group}.txt')
    # list_t = [1,2,3]
    # files_group = "g11"
    # write_list_to_file(f'/ivi/ilps/datasets/MSMarco-Web-Search/100M/cwid_list_{files_group}.txt', list_t)
    # check_matches()
    # count_record_per_language()
    # extract_english_ids()
    base_dir = '/ivi/ilps/datasets/clueweb22/disk1/txt/en/'
    output_path_file = '/ivi/ilps/datasets/MSMarco-Web-Search/100M/cw22_en_53M.jsonl' 
    
    start_idx = int(args[1])
    end_idx = int(args[2])
    extract_cw22_docs(base_dir, start_idx, end_idx)