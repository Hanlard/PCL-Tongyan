cd /userhome/fairseq/fairseq
# pip install --editable ./
# pip install fairscale
# python setup.py build_ext --inplace
# cd  /userhome/fairseq/fastmoe-master
# USE_NCCL=1 pip install --editable ./
# cd /userhome/fairseq/fairseq


Time=$(date "+%Y%m%d-%H%M%S")
N_world=$1
GateType=$2
moeTopK=$3
N_moe_Encoder=1
N_moe_Decoder=1
lg_pair="bg-zh,zh-bg,bs-zh,zh-bs,cs-zh,zh-cs,de-zh,zh-de,el-zh,zh-el,et-zh,zh-et,fa-zh,zh-fa,he-zh,zh-he,hr-zh,zh-hr,hu-zh,zh-hu,zh-id,id-zh,nl-zh,zh-nl,pl-zh,zh-pl,pt-zh,zh-pt,sl-zh,zh-sl,tr-zh,zh-tr,ur-zh,zh-ur,it-zh,zh-it"
# lg_pair="bg-zh"

# python  /userhome/fairseq/fairseq/user_dir/Change_1.2B_To_16Moe_Version.py
# rm -r /userhome/fairseq/fairseq/checkpoint/Moe-fairseq-DDP
fairseq-train \
/userhome/fairseq/fairseq/SiLuData/bin_dir \
--fixed-dictionary /userhome/fairseq/fairseq/dict_dir/418M_model_dict.128k.txt \
--save-dir /userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_16moe \
--task translation_multi_simple_epoch \
--encoder-normalize-before \
--langs 'af,am,ar,ast,az,ba,be,bg,bn,br,bs,ca,ceb,cs,cy,da,de,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kk,km,kn,ko,lb,lg,ln,lo,lt,lv,mg,mk,ml,mn,mr,ms,my,ne,nl,no,ns,oc,or,pa,pl,ps,pt,ro,ru,sd,si,sk,sl,so,sq,sr,ss,su,sv,sw,ta,th,tl,tn,tr,uk,ur,uz,vi,wo,xh,yi,yo,zh,zu' \
--lang-pairs ${lg_pair}  \
--max-tokens 384 \
--decoder-normalize-before \
--sampling-method temperature \
--sampling-temperature 1.5 \
--encoder-langtok src \
--decoder-langtok  \
--label-smoothing 0.1 \
--optimizer adam \
--adam-eps 1e-06 \
--adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt \
--lr 3e-05 \
--warmup-updates 5000 \
--max-update 5000000 \
--attention-dropout 0.1 \
--weight-decay 0.0 \
--update-freq 8 \
--save-interval 1 \
--save-interval-updates 5000 \
--keep-interval-updates 10 \
--no-epoch-checkpoints \
--seed 666 \
--log-format simple \
--log-interval 10 \
--patience 10 \
--arch transformer_moe_wmt_en_de_big \
--encoder-layers 24 --decoder-layers 24 --encoder-ffn-embed-dim 8192 --decoder-ffn-embed-dim 8192 --decoder-embed-dim 1024 --encoder-embed-dim 1024 \
--encoder-layerdrop 0.05 \
--decoder-layerdrop 0.05 \
--share-decoder-input-output-embed \
--share-all-embeddings \
--skip-invalid-size-inputs-valid-test \
--ddp-backend  legacy_ddp \
--user-dir /userhome/fairseq/fairseq/user_dir \
--gate-type ${GateType} \
--num-encoder-expert ${N_moe_Encoder} \
--moe-world-size ${N_world} \
--num-decoder-expert ${N_moe_Decoder} \
--clip-norm 1.0 \
--activation-dropout 0.4 \
--dropout 0.1 \
--moeTopK ${moeTopK} \
--min-loss-scale 1e-4 \
--fp16  \
--fp16-no-flatten-grads \
--max-source-positions 256 \
--max-target-positions 256 \
--criterion label_smoothed_cross_entropy_gate >log_dir/SiLuMoe/SiLu_1.2B_${N_world}Moe_${GateType}_Inhert_${Time}.log 2>&1
# --max-tokens 320 \
# --fp16-init-scale 8 \

# --encoder-ffn-embed-dim 8192 --decoder-ffn-embed-dim 8192 --decoder-embed-dim 2048 --encoder-embed-dim 2048
# --encoder-layers 24 --decoder-layers 24 --encoder-attention-heads 16 --decoder-attention-heads 16
# --encoder-layers 6 --decoder-layers 6 --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 --decoder-embed-dim 512 --encoder-embed-dim 512 \
