import torch
from collections import OrderedDict


Num_experts=8

# save_dir=f"/userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_8moe_SwitchGate/checkpoint_last.pt"

# chks=[]
# for i in range(Num_experts):
#     ModelPath=f"/userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_8moe_SwitchGate/checkpoint_1_92000_expert_{i}.pt"
    
#     chks.append(torch.load(ModelPath))
    
#     print(f"Load : {ModelPath}")
# print(f"Loaded all!")

# chk=chks[0]
# print(chk['cfg'])
# chk['cfg']['model'].nprocs_per_node=1
# chk['cfg']['model'].distributed_rank=0
# chk['cfg']['model'].distributed_world_size=1
# chk['cfg']['model'].num_encoder_expert=8
# chk['cfg']['model'].num_decoder_expert=8
# chk['cfg']['model'].moe_world_size=1
# chk['cfg']['distributed_training']['distributed_world_size']=1
# chk['cfg']['distributed_training']['nprocs_per_node']=1
# chk['cfg']['distributed_training']['distributed_num_procs']=1
# chk['cfg']['task'].moe_world_size=1
# chk['cfg']['task'].nprocs_per_node=1
# chk['cfg']['task'].num_encoder_expert=8
# chk['cfg']['task'].num_decoder_expert=8
# chk['cfg']['task'].distributed_world_size=1
# chk['cfg']['bmuf'].distributed_world_size=1


# for name in chk['model']:
#     if "htoh4" in name or "h4toh" in name:
#         chk['model'][name]=torch.cat([chk_i['model'][name] for chk_i in chks],dim=0)
#     print(name,chk['model'][name].shape)


# torch.save(chk, save_dir)


save_dir=f"/userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_8moe_SwitchGate/checkpoint_8to1_best.pt"
i=0
chk=torch.load(f"/userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_8moe_SwitchGate/checkpoint_best_expert_{i}.pt")
chk['args']=None
chk['cfg']['model'].nprocs_per_node=1
chk['cfg']['model'].distributed_rank=0
chk['cfg']['model'].distributed_world_size=1
chk['cfg']['model'].num_encoder_expert=Num_experts
chk['cfg']['model'].num_decoder_expert=Num_experts
chk['cfg']['model'].moe_world_size=1
chk['cfg']['distributed_training']['distributed_world_size']=1
chk['cfg']['distributed_training']['nprocs_per_node']=1
chk['cfg']['distributed_training']['distributed_num_procs']=1
chk['cfg']['task'].moe_world_size=1
chk['cfg']['task'].nprocs_per_node=1
chk['cfg']['task'].num_encoder_expert=Num_experts
chk['cfg']['task'].num_decoder_expert=Num_experts
chk['cfg']['task'].distributed_world_size=1
chk['cfg']['bmuf']['distributed_world_size']=1

for i in range(1,Num_experts):
    ModelPath=f"/userhome/fairseq/fairseq/checkpoint/SiLuMoe/1.2B_8moe_SwitchGate/checkpoint_best_expert_{i}.pt"
    chk_tmp=torch.load(ModelPath)
    print(f"Load : {ModelPath}")

    for name in chk['model']:
        if "htoh4" in name or "h4toh" in name:
            chk['model'][name]=torch.cat([chk['model'][name],chk_tmp['model'][name]],dim=0)
            print(name,chk['model'][name].shape)
    del chk_tmp
torch.save(chk, save_dir)
