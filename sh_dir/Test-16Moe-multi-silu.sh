# -*- coding: utf-8 -*-

DEVICES=$1
lg=$2
dataset=test

cd  /userhome/fairseq/fairseq
CUDA_VISIBLE_DEVICES=${DEVICES} fairseq-generate  /userhome/fairseq/fairseq/SiLuData/bin_dir --batch-size 16 --path /userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_16moe/checkpoint_last.pt  --fixed-dictionary dict_dir/model_dict.128k.txt -s 'zh' -t ${lg} --remove-bpe 'sentencepiece' --beam 5 --task translation_multi_simple_epoch --lang-pairs dict_dir/language_pairs_small_models.txt --decoder-langtok --encoder-langtok src --gen-subset ${dataset} --skip-invalid-size-inputs-valid-test --fp16 --user-dir /userhome/fairseq/fairseq/user_dir --max-tokens-valid 1024 --max-source-positions 1024 --max-target-positions 1024 --model-overrides '{"max-source-positions": "1024","max-target-positions": "1024"}' > /userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_16moe/moe16_test_logs/${dataset}_zh-${lg}.log

CUDA_VISIBLE_DEVICES=${DEVICES} fairseq-generate  /userhome/fairseq/fairseq/SiLuData/bin_dir --batch-size 16 --path /userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_16moe/checkpoint_last.pt --fixed-dictionary dict_dir/model_dict.128k.txt -s ${lg} -t 'zh' --remove-bpe 'sentencepiece' --beam 5 --task translation_multi_simple_epoch --lang-pairs dict_dir/language_pairs_small_models.txt --decoder-langtok --encoder-langtok src --gen-subset ${dataset} --skip-invalid-size-inputs-valid-test --fp16 --user-dir /userhome/fairseq/fairseq/user_dir --max-tokens-valid 1024 --max-source-positions 1024 --max-target-positions 1024 --model-overrides '{"max-source-positions": "1024", "max-target-positions": "1024"}' > /userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_16moe/moe16_test_logs/${dataset}_${lg}-zh.log 


cp -r /userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_16moe/moe16_test_logs/ examples/m2m_100/
cd examples/m2m_100/
bash Compute_bleu.sh zh    test.${lg}-zh.zh moe16_test_logs/test_${lg}-zh.log    > bleu_16moe/moe_16_${lg}-zh.bleu
bash Compute_bleu.sh ${lg} test.${lg}-zh.${lg} moe16_test_logs/test_zh-${lg}.log > bleu_16moe/moe_16_zh-${lg}.bleu
cat bleu_16moe/moe_16_${lg}-zh.bleu
cat bleu_16moe/moe_16_zh-${lg}.bleu