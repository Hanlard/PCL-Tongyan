#!/bin/bash                                                                                                                                                                                                       


dataset=$1
lang=${dataset#*.}
src=${lang%-*}
tgt=${lang#*-}

test_path=/userhome/fairseq/fairseq/SiLuData/TXT
log_path=/userhome/fairseq/fairseq/checkpoint/checkpoint_1.2B_multi_Silu/Test_BELU_1.2B_without_BPE_${lang}.log
SCRIPTS=/userhome/fairseq/mosesdecoder-master/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
gen=${log_path}
testset=${test_path}/${dataset}/valid.zh
detokenizer=$SCRIPTS/tokenizer/detokenizer.perl
 
 
 
grep ^H ${gen} \
    | sed 's/^H\-//' \
    | sort -n -k 1 \
    | cut -f 3 \
    | perl ${detokenizer} -l ${tgt} \
    > ${gen}.sorted.detok
 
cat ${testset} \
    | sed -r 's/(@@ )|(@@ ?$)//g' \
    | perl ${detokenizer} -l ${tgt} \
    > ${gen}.detok
 
cat ${gen}.sorted.detok | sacrebleu -w 2 -l ${lang} ${gen}.detok
