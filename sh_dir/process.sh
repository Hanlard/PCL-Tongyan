#!/bin/bash
## 解压
cd /userhome/fairseq/fairseq/SiLuData
# for file in nc.id-zh.tar.gz  nc.pt-zh.tar.gz  os.bs-zh.tar.gz  os.de-zh.tar.gz  os.et-zh.tar.gz  os.he-zh.tar.gz  os.hu-zh.tar.gz  os.pl-zh.tar.gz  os.tr-zh.tar.gz  un.ar-zh.tar.gz  un.es-zh.tar.gz  un.ru-zh.tar.gz   nc.it-zh.tar.gz  os.bg-zh.tar.gz  os.cs-zh.tar.gz  os.el-zh.tar.gz  os.fa-zh.tar.gz  os.hr-zh.tar.gz  os.nl-zh.tar.gz  os.sl-zh.tar.gz  os.ur-zh.tar.gz  un.en-zh.tar.gz  un.fr-zh.tar.gz  ; do
for file in  nc.it-zh.tar.gz    ; do
    array=(${file//./ })
    predix=${array[0]}
    lg_pair=${array[1]}
    dirname=${predix}.${lg_pair}
    array2=(${lg_pair//-/ })
    sou=${array2[0]}
    tar=${array2[1]}
    ## jieya
    tar -zxvf download/datasets/${lg_pair}/${file} -C TXT
    ## clean
    sed -r 's/(@@ )|(@@ ?$)|(@-@ )//g' TXT/${dirname}/train.${sou} >Detokenized/train.${lg_pair}.${sou}
    sed -r 's/(@@ )|(@@ ?$)|(@-@ )//g' TXT/${dirname}/valid.${sou} >Detokenized/valid.${lg_pair}.${sou}
    sed -r 's/(@@ )|(@@ ?$)|(@-@ )//g' TXT/${dirname}/test.${sou}  >Detokenized/test.${lg_pair}.${sou}
    sed -r 's/(@@ )|(@@ ?$)|(@-@ )//g' TXT/${dirname}/train.${tar} >Detokenized/train.${lg_pair}.${tar}
    sed -r 's/(@@ )|(@@ ?$)|(@-@ )//g' TXT/${dirname}/valid.${tar} >Detokenized/valid.${lg_pair}.${tar}   
    sed -r 's/(@@ )|(@@ ?$)|(@-@ )//g' TXT/${dirname}/test.${tar}  >Detokenized/test.${lg_pair}.${tar}  
    ## ->spm
    for mode in train valid test ; do
        python /userhome/fairseq/fairseq/scripts/spm_encode.py \
            --model /userhome/fairseq/fairseq/dict_dir/spm.128k.model \
            --output_format=piece \
            --inputs=/userhome/fairseq/fairseq/SiLuData/Detokenized/${mode}.${lg_pair}.${sou} \
            --outputs=/userhome/fairseq/fairseq/SiLuData/spm_dir/spm.${mode}.${lg_pair}.${sou}
            
        python /userhome/fairseq/fairseq/scripts/spm_encode.py \
            --model /userhome/fairseq/fairseq/dict_dir/spm.128k.model \
            --output_format=piece \
            --inputs=/userhome/fairseq/fairseq/SiLuData/Detokenized/${mode}.${lg_pair}.${tar} \
            --outputs=/userhome/fairseq/fairseq/SiLuData/spm_dir/spm.${mode}.${lg_pair}.${tar}
    done
    ## ->bin
    fairseq-preprocess \
        --source-lang ${sou} --target-lang ${tar} \
        --testpref /userhome/fairseq/fairseq/SiLuData/spm_dir/spm.test.${lg_pair} \
        --thresholdsrc 0 --thresholdtgt 0 \
        --destdir /userhome/fairseq/fairseq/SiLuData/bin_dir \
        --srcdict /userhome/fairseq/fairseq/dict_dir/1.2B_model_dict.128k.txt --tgtdict /userhome/fairseq/fairseq/dict_dir/1.2B_model_dict.128k.txt
    fairseq-preprocess \
        --source-lang ${sou} --target-lang ${tar} \
        --validpref /userhome/fairseq/fairseq/SiLuData/spm_dir/spm.valid.${lg_pair} \
        --thresholdsrc 0 --thresholdtgt 0 \
        --destdir /userhome/fairseq/fairseq/SiLuData/bin_dir \
        --srcdict /userhome/fairseq/fairseq/dict_dir/1.2B_model_dict.128k.txt --tgtdict /userhome/fairseq/fairseq/dict_dir/1.2B_model_dict.128k.txt
    fairseq-preprocess \
        --source-lang ${sou} --target-lang ${tar} \
        --trainpref /userhome/fairseq/fairseq/SiLuData/spm_dir/spm.train.${lg_pair} \
        --thresholdsrc 0 --thresholdtgt 0 \
        --destdir /userhome/fairseq/fairseq/SiLuData/bin_dir \
        --srcdict /userhome/fairseq/fairseq/dict_dir/1.2B_model_dict.128k.txt --tgtdict /userhome/fairseq/fairseq/dict_dir/1.2B_model_dict.128k.txt
done

