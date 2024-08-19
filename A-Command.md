1. fine tuning command

```
python fine-tuning\run_classifier.py --pretrained_model_path models\pretrained_model.bin --vocab_path models\encryptd_vocab.txt --train_path datasets\fine-tuning_dataset\cstnet-tls1.3_packet\packet\train_dataset.tsv --dev_path datasets\fine-tuning_dataset\cstnet-tls1.3_packet\packet\valid_dataset.tsv --test_path datasets\fine-tuning_dataset\cstnet-tls1.3_packet\packet\test_dataset.tsv --epochs_num 10 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 --learning_rate 2e-5

微调flow
python fine-tuning/run_classifier.py --pretrained_model_path models/pretrained_model.bin --vocab_path models/encryptd_vocab.txt --train_path datasets/CSTNET-TLS1.3/flow/train_dataset.tsv --dev_path datasets/CSTNET-TLS1.3/flow/valid_dataset.tsv --test_path datasets/CSTNET-TLS1.3/flow/test_dataset.tsv --epochs_num 10 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 --learning_rate 2e-5


python fine-tuning/run_classifier.py --pretrained_model_path models/pretrained_model.bin --vocab_path models/encryptd_vocab.txt --train_path datasets/fine-tuning_dataset/packet/train_dataset.tsv --dev_path datasets/fine-tuning_dataset/packet/valid_dataset.tsv --test_path datasets/fine-tuning_dataset/packet/test_dataset.tsv --epochs_num 10 --batch_size 32 --embedding word_pos_seg --encoder transformer --mask fully_visible --seq_length 128 --learning_rate 2e-5

```

2. infer command
```
# 推理package流量
python inference/run_classifier_infer.py --load_model_path models/finetuned_model_tls13_flow.bin --vocab_path models/encryptd_vocab.txt --test_path datasets/CSTNET-TLS1.3/package/nolabel_test_dataset.tsv --prediction_path datasets/CSTNET-TLS1.3/package/prediction_fine-tuned-model.tsv --labels_num 120 --embedding word_pos_seg --encoder transformer --mask fully_visible

# 推理flow流量
python inference/run_classifier_infer.py --load_model_path models/finetuned_model_tls13_flow.bin --vocab_path models/encryptd_vocab.txt --test_path datasets/CSTNET-TLS1.3/flow/nolabel_test_dataset.tsv --prediction_path datasets/CSTNET-TLS1.3/flow/prediction_fine-tuned-model.tsv --labels_num 120 --embedding word_pos_seg --encoder transformer --mask fully_visible
```

3. create pratrain data command
```
python preprocess.py --corpus_path corpora/encrypted_tls13_burst_yxy_8from9.txt --vocab_path models/encryptd_vocab_tls13_yxy_8from9.txt --dataset_path dataset_tls13_yxy_8from9.pt --processes_num 8 --target bert
```


4. docker
```
docker exec -it yxy /bin/bash

docker run -it --rm -v /home/gpu2/yxy:/workspace/yxy --name yxy 910458da4f87 /bin/bash

3090  jupyter
docker run -it --rm -v /home/gpu2/yxy:/code/yxy -p 13002:13002 --name yxy -e JUPYTER_TOKEN=my_secret_token 910458da4f87 /bin/bash

docker run -it -v /home/gpu2/yxy/ET-BERT:/workspace/ET-BERT --name yxy 67c1f0ac9023 /bin/bash

docker run -d -it -v /home/gpu2/yxy/ET-BERT:/workspace/ET-BERT --name yxy 67c1f0ac9023 /bin/bash

docker run -it --rm -v /home/emnets/Downloads/youxinyun/Datasets:/code/Datasets  --name youxinyun-netLlm d51cc2d50eee /bin/bash  #下载数据用的镜像

docker run --gpus all --ipc=host -it -p 8101:8101 -v /home/emnets/Downloads/youxinyun:/workspace/yxy --restart=always --name youxy ab2c674da5c6 /bin/bash  #训练用的
# 8001 用于vllm
```



