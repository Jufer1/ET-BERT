#!/usr/bin/python3
#-*- coding:utf-8 -*-

import scapy.all as scapy
import binascii
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import json
import os
import csv
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from flowcontainer.extractor import extract
import tqdm
import random
from multiprocessing import Pool

random.seed(40)

# pcap_dir = "D:\\desktop\\workcache\\aiLabSpace\\ET-BERT\\datasets\\origin-datasets\\Supplementary_Dataset_from_completePCAPs\\"
pcap_dir = "datasets\\origin-datasets\\fine-tuning_dataset\\cstnet-tls 1.3\\"

# 预料存放路径
word_dir = "corpora\\"
word_name = "encrypted_tls13_burst_yxy_4from18.txt"

# 词表存放路径
vocab_dir = "models\\"
vocab_name = "encryptd_vocab_tls13_yxy_4from18.txt"

def preprocess(files_dir, proc_id, start, end):
    print("Worker %d is building dataset ... start %d end %d " %(proc_id, start, end))
    
    packet_num = 0
    n = 0

    
    for index in range(start, end):
        tmp_dir = files_dir[index] + "\\"
        print("Worker %d now pre-process pcap_dir is %s"%(proc_id, tmp_dir))
        for parent,dirs,files in os.walk(tmp_dir):
            for file in files:
                if "pcapng" not in file and "pcap" in file:
                    n += 1
                    pcap_name = parent + "\\" + file
                    print("Worker %d No.%d pacp is processed ... %s ..."%(proc_id, n,file))
                    packets = scapy.rdpcap(pcap_name)
                    #word_packet = b''
                    words_txt = []

                    # 对于每个packet
                    # 1. 将packet转换为16进制字符串
                    # 2. 将数据包切分为两部分
                    # 3. 对每部分进行bigram处理
                    # 4. 将两部分写入words_txt
                    # 5. 每个数据包处理完后，写入换行符

                    for p in packets:
                        packet_num += 1
                        word_packet = p.copy()
                        words = (binascii.hexlify(bytes(word_packet))) # 两个16进制字符表示一个字节
                        # 以太网帧头部长度为14-18字节，IPv4头部长度为20-60字节，IPv6头部长度为40字节
                        words_string = words.decode()[76:] # 从第76个字节开始处理,去掉以太网帧头部和IPv4头部 （20 + 18） * 2 = 76
                        # print(words_string)
                        length = len(words_string)
                        if length < 10: # TODO 看看这个阈值是否合适
                            continue
                        for string_txt in cut(words_string, int(length / 2)): # 分成两部分的猜测：BERT的输入需要两个句子，这里的两部分就是两个句子
                            token_count = 0 
                            sentence = cut(string_txt,1)  
                            for sub_string_index in range(len(sentence)):
                                if sub_string_index != (len(sentence) - 1):
                                    token_count += 1
                                    if token_count > 256: # 限制最大token数
                                        break
                                    else:
                                        merge_word_bigram = sentence[sub_string_index] + sentence[
                                                                    sub_string_index + 1]  
                                else:
                                    break  
                                words_txt.append(merge_word_bigram)
                                words_txt.append(' ')
                            words_txt.append("\n")
                        words_txt.append("\n")

                    
                    result_file = open(word_dir + str(proc_id) + "-" + word_name, 'a')
                    for words in words_txt:
                        result_file.write(words)
                    result_file.close()
    print("Worker %d finish preprocessing %d pcaps"%(proc_id, n))
    return packet_num

def cut(obj, sec):
    result = [obj[i:i+sec] for i in range(0,len(obj),sec)]
    remanent_count = len(result[0])%4
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i+sec+remanent_count] for i in range(0,len(obj),sec+remanent_count)]
    return result

def build_BPE(): # 制作Byte Pair Encoding (BPE)词表
    # generate source dictionary,0-65535
    num_count = 65536
    not_change_string_count = 5
    i = 0
    source_dictionary = {} 
    tuple_sep = ()
    tuple_cls = ()
    #'PAD':0,'UNK':1,'CLS':2,'SEP':3,'MASK':4
    while i < num_count:
        temp_string = '{:04x}'.format(i) 
        source_dictionary[temp_string] = i
        i += 1
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.WordPiece(vocab=source_dictionary,unk_token="[UNK]",max_input_chars_per_word=4))

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece()
    tokenizer.post_processor = processors.BertProcessing(sep=("[SEP]",1),cls=('[CLS]',2))

    # And then train
    trainer = trainers.WordPieceTrainer(vocab_size=65536, min_frequency=2)
    tokenizer.train([word_dir+word_name], trainer=trainer) # 根据词频训练BPE

    # And Save it
    tokenizer.save("wordpiece_tls13_yxy_4from18.tokenizer.json", pretty=True)
    return 0

def build_vocab(): # 制作词表
    json_file = open("wordpiece_tls13_yxy_4from18.tokenizer.json",'r')
    json_content = json_file.read()
    json_file.close()
    vocab_json = json.loads(json_content)
    vocab_txt = ["[PAD]","[SEP]","[CLS]","[UNK]","[MASK]"]
    for item in vocab_json['model']['vocab']:
        vocab_txt.append(item) # append key of vocab_json
    with open(vocab_dir+vocab_name,'w') as f:
        for word in vocab_txt:
            f.write(word+"\n")
    return 0

def bigram_generation(packet_string,flag=False):
    result = ''
    sentence = cut(packet_string,1)
    token_count = 0
    for sub_string_index in range(len(sentence)):
        if sub_string_index != (len(sentence) - 1):
            token_count += 1
            if token_count > 256: 
                break
            else:
                merge_word_bigram = sentence[sub_string_index] + sentence[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '
    if flag == True:
        result = result.rstrip()

    return result

def read_pcap_feature(pcap_file):
    packet_length_feature = []
    feature_result = extract(pcap_file, filter='tcp')
    for key in feature_result.keys():
        value = feature_result[key]
        packet_length_feature.append(value.ip_lengths)
    return packet_length_feature[0]

def read_pcap_flow(pcap_file):
    packets = scapy.rdpcap(pcap_file)

    packet_count = 0  
    flow_data_string = '' 

    if len(packets) < 5:
        print("preprocess flow %s but this flow has less than 5 packets."%pcap_file)
        return -1

    print("preprocess flow %s" % pcap_file)
    for packet in packets:
        packet_count += 1
        if packet_count == 5:
            packet_data = packet.copy()
            data = (binascii.hexlify(bytes(packet_data)))
            packet_string = data.decode()
            flow_data_string += bigram_generation(packet_string,flag = True)
            break
        else:
            packet_data = packet.copy()
            data = (binascii.hexlify(bytes(packet_data)))
            packet_string = data.decode()
            flow_data_string += bigram_generation(packet_string)
    return flow_data_string

def split_cap(pcap_file,pcap_name):
    cmd = "I:\\SplitCap.exe -r %s -s session -o I:\\split_pcaps\\" + pcap_name
    command = cmd%pcap_file
    os.system(command)
    return 0

def preprocess_helper():
    '''
    多进程处理数据集
    '''
    workers_num = 6
    lines_num = len(os.listdir("datasets\\origin-datasets\\fine-tuning_dataset\\cstnet-tls 1.3\\"))

    current_directory = "datasets\\origin-datasets\\fine-tuning_dataset\\cstnet-tls 1.3\\"
    files_dir = []
    entries = os.listdir(current_directory)
    for entry in entries:
        full_path = os.path.join(current_directory, entry)
        if os.path.isdir(full_path):
            files_dir.append(full_path)
    files_dir_array = np.array(files_dir)
    print(files_dir_array)
    np.savetxt("corpora//files_dir.txt", files_dir_array, delimiter=",", fmt="%s")

    if debug_flag == True:
        preprocess(files_dir, 0, 0, 1)
        return 0

    pool = Pool(workers_num)
    for i in range(workers_num):
        start = i * lines_num // workers_num
        end = (i + 1) * lines_num // workers_num
        pool.apply_async(func=preprocess, args=[files_dir, i, start, end])
        # print("preprocess worker %d start %d end %d"%(i, start, end))
    pool.close()
    pool.join()

if __name__ == '__main__':
    debug_flag = True
    preprocess_helper()
    # preprocess(pcap_dir)

    # build vocab
    # build_BPE()
    # build_vocab()
