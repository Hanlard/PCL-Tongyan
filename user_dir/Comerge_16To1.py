import torch
from collections import OrderedDict


Num_experts=16

save_dir=f"/userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_16moe/checkpoint_last.pt"

# chk_history=torch.load(f"/userhome/fairseq/fairseq/checkpoint/SiLuMoe/Silu_Moe_1.2B_Inhert_GShardGate/checkpoint_81_95000_expert_0.pt")

i=0
chk=torch.load(f"/userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_16moe/checkpoint_last_expert_{i}.pt")
# chk['cfg']=chk_history['cfg']
# del chk_history

print(chk['cfg'])
chk['args']=None
chk['cfg']['model'].nprocs_per_node=1
chk['cfg']['model'].distributed_rank=0
chk['cfg']['model'].distributed_world_size=1
chk['cfg']['model'].num_encoder_expert=Num_experts
chk['cfg']['model'].num_decoder_expert=Num_experts
chk['cfg']['model'].moe_world_size=1
chk['cfg']['model'].max_source_positions=1024
chk['cfg']['model'].max_target_positions=1024

chk['cfg']['distributed_training']['distributed_world_size']=1
chk['cfg']['distributed_training']['nprocs_per_node']=1
chk['cfg']['distributed_training']['distributed_num_procs']=1
chk['cfg']['task'].moe_world_size=1
chk['cfg']['task'].nprocs_per_node=1
chk['cfg']['task'].num_encoder_expert=Num_experts
chk['cfg']['task'].num_decoder_expert=Num_experts
chk['cfg']['task'].distributed_world_size=1
chk['cfg']['task'].max_source_positions=1024
chk['cfg']['task'].max_target_positions=1024

chk['cfg']['bmuf']['distributed_world_size']=1

for i in range(1,Num_experts):
    ModelPath=f"/userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_16moe/checkpoint_last_expert_{i}.pt"
    chk_tmp=torch.load(ModelPath)
    print(f"Load : {ModelPath}")

    for name in chk['model']:
        if "htoh4" in name or "h4toh" in name:
            chk['model'][name]=torch.cat([chk['model'][name],chk_tmp['model'][name]],dim=0)
            print(name,chk['model'][name].shape)
    del chk_tmp
torch.save(chk, save_dir)
print(f"Saved to {save_dir}")