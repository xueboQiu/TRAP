import gzip
import io
import json
import os
import re

from tqdm import tqdm

pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_dst1 = re.compile(r'objectID\":\"(.*?)\"')
# pattern_src_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestamp\":\"(.*?)\"')
pattern_dst_type = re.compile(r'object\":\"(.*?)\"')
pattern_src_id = re.compile(r'actorID\":"(.*?)\"')
pattern_edge_type = re.compile(r"\"action\":\"(.*?)\"")
pattern_dst_file_name = re.compile(r'file_path\":\"(.*?)\"')
pattern_dst_module_name = re.compile(r'module_path\":\"(.*?)\"')
pattern_dst_flow_in_ip_name = re.compile(r'src_ip\":\"(.*?)\"')
pattern_dst_flow_out_ip_name = re.compile(r'dest_ip\":\"(.*?)\"')
pattern_src_process_name = re.compile(r'"image_path":"(.*?)"')

def preprocess_optc_dataset(dataset, metadata):
    id_nodetype_map = {}
    id_nodename_map = {}

    valid_objects = {'PROCESS', 'FILE', 'FLOW', 'MODULE'}
    # valid_objects = {'PROCESS', 'FILE', 'FLOW', 'MODULE', 'SHELL'}
    # valid_objects = {'PROCESS', 'FILE', 'FLOW', 'MODULE', 'SHELL'}
    # invalid_actions = {'START', 'TERMINATE'}
    invalid_actions = {}
    for file in os.listdir('./data/{}/'.format(dataset)):
        if 'json' in file and ('systemia.com.' in file or 'benign_' in file) and not '.txt' in file and not 'names' in file and not 'types' in file and not 'metadata' in file:
            print('reading {} ...'.format(file))
            f = open('./data/{}/'.format(dataset) + file, 'r', encoding='utf-8')
            for line in tqdm(f):
                line = line.replace(r"\\", "\\").replace("\\", "/").replace("/\"", "").replace("/??/", "")

                uuid = pattern_src_id.findall(line)
                if uuid is None or len(uuid) == 0: continue
                uuid = uuid[0]

                id_nodetype_map[uuid] = 'PROCESS'
                # src_uuid, src_name, dst_uuid, dst_name = parse_node_name(line)
    for key in metadata[dataset]:
        for file in metadata[dataset][key]:
            if os.path.exists('./data_cl/{}/'.format(dataset) + file + '.txt'):
                continue
            f = open('./data/{}/'.format(dataset) + file, 'r', encoding='utf-8')
            fw = open('./data_cl/{}/'.format(dataset) + file + '.txt', 'w', encoding='utf-8')
            print('processing {} ...'.format(file))
            for line in tqdm(f):
                line = line.replace(r"\\", "\\").replace("\\", "/").replace("/\"", "").replace("/??/", "")
                edgeType = pattern_edge_type.findall(line)[0]

                if edgeType in invalid_actions:
                    print('edgeType not valid in line: {}'.format(line))
                    continue


                timestamp = pattern_time.findall(line)[0]
                srcId = pattern_src_id.findall(line)

                if len(srcId) == 0: continue
                srcId = srcId[0]
                if not srcId in id_nodetype_map:
                    print('srcId not found in line: {}'.format(line))
                    continue

                srcType = id_nodetype_map[srcId]
                dstId1 = pattern_dst1.findall(line)
                if len(dstId1) > 0 and dstId1[0] != 'null':
                    dstId1 = dstId1[0]
                    # if not dstId1 in id_nodetype_map:
                    #     continue
                else:
                    print('dstId1 not found in line: {}'.format(line))
                    exit(0)

                dstType1 = pattern_dst_type.findall(line)
                if len(dstType1) > 0 and dstType1[0] != 'null':
                    dstType1 = dstType1[0]

                # if dstType1 not in valid_objects:
                #     # print('dstType1 not valid in line: {}'.format(line))
                #     continue
                if dstId1 not in id_nodetype_map:
                    id_nodetype_map[dstId1] = dstType1
                this_edge1 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId1) + '\t' + str(
                    dstType1) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                fw.write(this_edge1)

            fw.close()
            f.close()
    if len(id_nodename_map) != 0:
        fw = open('./data_cl/{}/'.format(dataset) + 'names.json', 'w', encoding='utf-8')
        json.dump(id_nodename_map, fw)
    if len(id_nodetype_map) != 0:
        fw = open('./data_cl/{}/'.format(dataset) + 'types.json', 'w', encoding='utf-8')
        json.dump(id_nodetype_map, fw)

def extract_logs(filepath, hostid, is_train = False):

    in_dir= './data/optc/'
    if '0201' in hostid:
        out_dir= './data/optc_1/'
    elif '0501' in hostid:
        out_dir= './data/optc_2/'
    elif '0051' in hostid:
        out_dir= './data/optc_3/'

    search_pattern = f'SysClient{hostid}'

    if not is_train:
        output_filename = f'SysClient{hostid}.systemia.com.json'
    else:
        id_name = filepath.split('ecar-')[1].replace('.json.gz', '.json')
        output_filename = f'{hostid}_benign_{id_name}'
        # output_filename = f'SysClient{hostid}_benign.systemia.com.json'

    with gzip.open(in_dir + filepath, 'rt', encoding='utf-8') as fin:
        with open(out_dir + output_filename, 'ab') as f:
            out = io.BufferedWriter(f)
            for line in fin:
                if search_pattern in line:
                    out.write(line.encode('utf-8'))
            out.flush()

def prepare_test_set():

    mal_log_files = [
        # ("AIA-201-225.ecar-2019-12-08T11-05-10.046.json.gz", "0201"),
        # ("AIA-201-225.ecar-last.json.gz", "0201"),
        ("AIA-501-525.ecar-2019-11-17T04-01-58.625.json.gz", "0501"),
        ("AIA-501-525.ecar-last.json.gz", "0501"),
        # ("AIA-51-75.ecar-last.json.gz", "0051")
    ]

    # os.system("rm SysClient0201.com.txt")
    # os.system("rm SysClient0501.com.txt")
    # os.system("rm SysClient0051.com.txt")

    for file, code in tqdm(mal_log_files, desc="Extracting logs", unit="file"):
        extract_logs(file, code)

def prepare_train_set():
    benign_log_files = [
        # ("AIA-201-225.ecar-2019-12-07T22-06-33.589.json.gz", "0201"),
        # ("AIA-201-225.ecar-2019-12-08T01-57-30.012.json.gz", "0201"),
        # ("AIA-201-225.ecar-2019-12-08T05-46-21.658.json.gz", "0201"),
        # ("AIA-201-225.ecar-last (1).json.gz", "0201"),
        #
        # ("AIA-501-525.ecar-2019-11-15T13-29-59.064.json.gz", "0501"),
        # ("AIA-501-525.ecar-2019-11-15T17-22-42.923.json.gz", "0501"),
        # ("AIA-501-525.ecar-last (1).json.gz", "0501"),

        ("AIA-51-75.ecar-2019-12-08T00-56-58.175.json.gz", "0051"),
        ("AIA-51-75.ecar-2019-12-08T04-30-36.852.json.gz", "0051"),
        ("AIA-51-75.ecar-last (1).json.gz", "0051")
    ]

    for file, code in tqdm(benign_log_files, desc="Extracting logs", unit="file"):
        extract_logs(file, code, is_train = True)

if __name__ == "__main__":

    prepare_train_set()
    # prepare_test_set()