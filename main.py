import sys
import orjson
import gzip
import dask.dataframe as dd
import glob, os

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
    # file_path = '/ivi/ilps/datasets/clueweb22/disk1/txt/en/en00/en0000/en0000-00.json.gz'
    # read_file(file_path)
    # read_file_dask(file_path)
    
    # file_path = '/ivi/ilps/datasets/clueweb22/disk1/txt/en/en33/en3301/en3301-79.json.gz'
    # read_file(file_path)
    
    #reading the file dynamically
    # read_file_dynamic(file_path, start_line=10000, num_lines=5)
    directory_path = '/ivi/ilps/datasets/clueweb22/disk1/txt/en'
    file_count = count_json_gz_files(directory_path)
    print(f"Total number of .json.gz files: {file_count}")
    records_count = count_json_gz_files_records(directory_path)
    
    for file_path, count in records_count.items():
        print(f"{file_path}: {count} records")
    total_json_item = sum(records_count.values())
    print(f"Total number of records: {total_json_item}")
        
    
if __name__ == "__main__":
    main()