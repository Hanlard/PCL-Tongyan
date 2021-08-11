import torch
from collections import OrderedDict

Gate='SwitchGate'

## Ori-M2M or Silu-M2M
# ModelPath="/userhome/fairseq/fairseq/model_dir/1.2B_last_checkpoint.pt"
ModelPath="/userhome/fairseq/fairseq/checkpoint/checkpoint_1.2B_multi_Silu/checkpoint_last.pt"

chk=torch.load(ModelPath)
print(chk.keys())
Encoder_layers=chk['cfg']['model'].encoder_layers
Decoder_layers=chk['cfg']['model'].decoder_layers

Num_experts=8

# dict_keys(['args', 'model', 'optimizer_history', 'extra_state', 'last_optimizer_state'])

model_ori=chk['model']

Hidden=model_ori['encoder.embed_tokens.weight'].shape[1]

model = OrderedDict()

chk0=torch.load("/userhome/fairseq/fairseq/checkpoint/Moe-fairseq-DDP/Silu_Moe_1.2B_SwitchGate/checkpoint_best_expert_0.pt")
para_names=chk0['model'].keys()
del chk0

for name in para_names:    
    if name in model_ori:
        model[name]=model_ori[name]
    elif "MoeMLP.gate.gate.weight" in name:
        model[name]=torch.rand([Num_experts,Hidden])*0.01+torch.ones([Num_experts,Hidden])
    elif "MoeMLP.gate.gate.bias" in name:
        model[name]=torch.zeros([Num_experts])
    elif "MoeMLP.experts.htoh4" in name:
        name_ori = name.replace("MoeMLP.experts.htoh4","fc1")
        model[name]=model_ori[name_ori].unsqueeze(dim=0)
    elif "MoeMLP.experts.h4toh" in name:
        name_ori = name.replace("MoeMLP.experts.h4toh","fc2")
        model[name]=model_ori[name_ori].unsqueeze(dim=0)
    elif "decoder.output_projection.weight"==name:
        model[name]=model_ori['encoder.embed_tokens.weight']
    else:
        print(f"No that name:{name}")


chk['last_optimizer_state']=None
chk['model'] = model
# chk['args'].arch='transformer_moe_wmt_en_de_big'

for i in range(Num_experts):
    save_dir=f"/userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_8moe_{Gate}/checkpoint_last_expert_{i}.pt"
    torch.save(chk, save_dir)
    print(f"saved on {save_dir}")

print(f"saved all!")

