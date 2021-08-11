#!/usr/bin/env bash
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

src=$1
tgt=$2
pair=${src}-${tgt}
subset=$3

if [ "$src" = "zh" ]; then
pair=${tgt}-zh
fi

# fairseq=/userhome/fairseq/fairseq
# cd ${fairseq}/examples/m2m_100
# ## prediction
# pred_path=Pangu/translate_output/val.translation_${src}_${tgt}.txt
# ## labels 
# # label_path=Pangu/val_data/${subset}.${pair}.${tgt}
# label_path=/userhome/fairseq/fairseq/SiLuData/Detokenized/${subset}.${pair}.${tgt}

# cat ${pred_path} | sh tok.sh ${tgt} > Pangu/res/${src}-${tgt}_predcitions
# cat ${label_path} | sh tok.sh ${tgt} > Pangu/res/${src}-${tgt}_labels


# if [ "$src" = "zh" ];then
# # echo "Not detokenized"
# sacrebleu -tok 'none' Pangu/res/${src}-${tgt}_labels < Pangu/res/${src}-${tgt}_predcitions
# fi


# ## chinese Detokenized
# if [ "$tgt" = "zh" ];then
# # echo "Detokenized"
# sed -r 's/( )//g'  Pangu/res/${src}-${tgt}_predcitions  > Pangu/res/${src}-${tgt}_predcitions_D
# sed -r 's/( )//g' Pangu/res/${src}-${tgt}_labels  > Pangu/res/${src}-${tgt}_labels_D
# cat Pangu/res/${src}-${tgt}_predcitions_D | sacrebleu -w 2 -l ${src}-${tgt} Pangu/res/${src}-${tgt}_labels_D
# fi



# |sed '/^$/d'
# sed -r 's/(@@ )|(@@ ?$)//g'




fairseq=/userhome/fairseq/fairseq
cd ${fairseq}/examples/m2m_100
## prediction
pred_path=Pangu/translate_test_output/test.translation_${src}_2${tgt}.txt
## labels 
# label_path=Pangu/val_data/${subset}.${pair}.${tgt}
label_path=/userhome/fairseq/fairseq/SiLuData/Detokenized/${subset}.${pair}.${tgt}

if [ "$src" = "zh" ];then
cat ${pred_path} | sh tok.sh ${tgt} > Pangu/res/${src}-${tgt}_predcitions
cat ${label_path} | sh tok.sh ${tgt} > Pangu/res/${src}-${tgt}_labels
# echo "Not detokenized"
sacrebleu -tok 'none' Pangu/res/${src}-${tgt}_labels < Pangu/res/${src}-${tgt}_predcitions
fi

## chinese Detokenized
if [ "$tgt" = "zh" ];then
cat ${pred_path} > Pangu/res/${src}-${tgt}_predcitions
cat ${label_path}  > Pangu/res/${src}-${tgt}_labels
# echo "Detokenized"
sed -r 's/( )//g'  Pangu/res/${src}-${tgt}_predcitions  > Pangu/res/${src}-${tgt}_predcitions_D
sed -r 's/( )//g' Pangu/res/${src}-${tgt}_labels  > Pangu/res/${src}-${tgt}_labels_D
cat Pangu/res/${src}-${tgt}_predcitions_D | sacrebleu -w 2 -l ${src}-${tgt} Pangu/res/${src}-${tgt}_labels_D
fi
