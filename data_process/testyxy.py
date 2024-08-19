import os

for _parent,_dirs,files in os.walk('D:\\desktop\\workcache\\aiLabSpace\\ET-BERT\\datasets\\'):
    print(__file__)
    print(_parent)
    print(_dirs)
    print(files)
    break