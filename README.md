
# PCL-Tongyan
PCL-tonglian is a multi-language machine translation model. The single model supports 17  minority languages translation with Chinese, it also supports translation between any two languages. PCL-Tongyan is a multilingual machine translation model improved on the structure of M2M-100 model. Through parameter reusing and incremental training, the model parameters are increased from 1.2B to 13.2B, which greatly improves the translation performance of multiple minority languages. We use a lifelong learning approach based on dynamic playback, PCL-Tongyan can continuously learn new language translation without forgetting old languages. More details are given in the PPT.


## Features

1. The M2M-1.2B model was incrementally improved to a MOE version
2. Incremental training is used to reduce computational consumption, in line with the concept of "green-AI"
3. By resuing M2M parameters + random noise - > Multiple experts
4. Support lifelong learning of new languages by the sustainable learning solutions based on dynamic playback
5. Using distributed MOE strategy to improve resource utilization efficiency, only 16 V100 are enough to data and expert hybrid parallel
6. Based on FairseQ and FastMOe, fast to train and easy to deploy
7. Using a single V100 graphics card for inference, without inter-card communication, greatly improving the inference speed.

## Model structure
![add image](https://github.com/Hanlard/PCL-Tongyan/blob/main/model_strcture.png)

## Data source
 https://git.pcl.ac.cn/PCMachineTranslation/PCMT/src/branch/master/datasets
 
-- See Excel for data statistics
    
## Incremental training principle
![add image](https://github.com/Hanlard/PCL-Tongyan/blob/main/incre_training_en.png)


## Training steps
    
1. According to the official instructions install fairseq and fastmoe (We use NVCR. IO/nvidia/pytorch: 21.06 py3 docker environment installation, also can bash sh_dir/Install_fair.sh for installation)
2. Copy all files from pcl-tongyan directory to fairseq installation directory (overwrite)
3. Download the "silk road" dataset: https://git.pcl.ac.cn/PCMachineTranslation/PCMT/src/branch/master/datasets 
4. Process data (change the directory)
    bash sh_dir/process.sh
5. Download m2m - 100 dict and checkpoint files from https://github.com/pytorch/fairseq/tree/master/examples/m2m_100
6. Convert M2M-100 into MOE model
    Python change_1.2b_to_16moe_version.py
7. Start incremental MOE training 
    sh_dir/train-16MOe-silu-inhert.sh

## Single V100 inference step (32G video memory is required)
1. Convert the distributed MOE model to a single-card deployment
    uer_dir/Comerge_16To1.py
2.  Test bleu on xx->zh/h->xx
    sh_dir/Test-16Moe-multi-silu.sh

## Function command
1. Switch from normal model to MOE model
    python Change_1.2B_To_16Moe_Version.py

2. Convert distributed MOE model to single card deployment
    python Comerge_16To1.py

3. Fine-tuning multilingual translation task 
    bash sh_dir/Train-16moe-SiLu-Inhert.sh 16 GShardGate 2

4. Test bleu on xx->zh and zh->xx direction 
    bash sh_dir/Test-16Moe-multi-silu.sh 0 xx

5. Data processing
    bash sh_dir/process.sh
    
## Service Invocation API
    import requests
    def Tongyan_Translate(sentences=None,direction=None,PyTorch_REST_API_URL = 'http://192.168.202.124:5000/predict'):
        c_lgs=['中文(zh)','意大利语(it)','德语(de)','捷克语(cs)','荷兰语(nl)','葡萄牙语(pt)','印尼语(id)','保加利亚语(bg)','波斯尼亚(bs)',
               '波斯尼亚(bs)','希腊语(el)','波斯语(fa)','克罗地亚语(hr)','匈牙利语(hu)','爱沙尼亚语(et)','希伯来语(he)',
              '斯洛文尼亚(sl)','波兰语(pl)','土耳其语(tr)','乌尔都语(ur)']
        lgs=['zh','it','de','cs','nl','pt','id','bg','bs','bs','el','fa','hr','hu','et','he','sl','pl','tr','ur']
        src,tgt=direction.split("-")
        if src not in lgs or tgt not in lgs:
            print(f"参数<direction>请在下面集合中的语言按照xx-xx的格式输入: \n{','.join(c_lgs)}")
            return None
        else:
            payload = {'data': [direction,sentences]}
            # Submit the request.
            r = requests.post(PyTorch_REST_API_URL, data=payload).json()
            if r['success']:
                translations=[sent for sent in enumerate(r['predictions'])]
                return translations
            else:
                return None
    if __name__ == '__main__':
        sentences = [
        "I want to eat an apple ",
        "Today is a fine day! ",
        "Hello, I am THE senior engineer OF PCL XXX, please give me your advice!"
        ]
        direction = "zh-pt"
        res=Tongyan_Translate(sentences=sentences,direction=direction)
        print(res)   
        
## Environments
    fairseq                       1.0.0a0+2fd9d8a     
    fastmoe                       0.2.0               
    

## Model performance
### Multilingual machine translation performance
![add image](https://github.com/Hanlard/PCL-Tongyan/blob/main/bleus_en.png)
### Lifelong learning performance
![add image](https://github.com/Hanlard/PCL-Tongyan/blob/main/lll_pic_en.png)






