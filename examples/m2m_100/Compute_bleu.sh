#!/usr/bin/env bash
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
lg=$1
data=$2
log_path=$3
lg_pair=${data#*.}
lg_pair=${lg_pair%.*}
fairseq=/userhome/fairseq/fairseq
datadir=/userhome/fairseq/fairseq/SiLuData/Detokenized/${data}
cd ${fairseq}/examples/m2m_100
cat ${log_path} | grep -P "^H" | sort -V | cut -f 3- | sh tok.sh ${lg} > res/${data}_hyp_predcitions
cat ${datadir} | sh tok.sh ${lg} > res/${data}_ref_labels
# sed '/^$/d'  res/${data}_hyp_predcitions  > res/${data}_hyp_predcitions_0
# sed '/^$/d'  res/${data}_ref_labels  > res/${data}_ref_labels_0
echo "Not detokenized"
sacrebleu -tok 'none' res/${data}_ref_labels < res/${data}_hyp_predcitions

## chinese Detokenized
if [ "$lg" = "zh" ];then
echo "Detokenized"
sed -r 's/( )//g'  res/${data}_hyp_predcitions  > res/${data}_hyp_predcitions_D
sed -r 's/( )//g' res/${data}_ref_labels  > res/${data}_ref_labels_D
cat res/${data}_hyp_predcitions_D | sacrebleu -w 2 -l ${lg_pair} res/${data}_ref_labels_D
fi



#|sed '/^$/d'
# sed -r 's/(@@ )|(@@ ?$)//g'