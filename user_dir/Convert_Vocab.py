import torch
from collections import OrderedDict


vocabs=["<s>","<pad>","</s>","<unk>"]

Vocab_path=f"/userhome/fairseq/fairseq/dict_dir/1.2B_model_dict.128k.txt"
ModelPath=f"/userhome/fairseq/fairseq/model_dir/1.2B_last_checkpoint.pt"
with open(Vocab_path,encoding="utf-8") as f:
    vocabs_=f.read().splitlines()
    vocabs = vocabs + vocabs_

chk=torch.load(ModelPath)
embs=chk['model']['encoder.embed_tokens.weight']
del chk

from tqdm import tqdm
with open(f"/userhome/fairseq/fairseq/model_dir/emb_128112_1024.txt","w",encoding="utf-8") as w:
    to_wright="128112 1024\n"
    for i in tqdm(range(len(vocabs))):
        word = vocabs[i]
        if " " in word:
            continue
        emb=" ".join([str(x) for x in embs[i].cpu().tolist()])
        
        line = f"{word} {emb}\n"
        to_wright = to_wright+line
        if len(to_wright)>5000000:
            w.write(to_wright)
            to_wright=""
    if to_wright:
        w.write(to_wright)
        to_wright=""