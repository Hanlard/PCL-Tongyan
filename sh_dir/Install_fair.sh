# -*- coding: utf-8 -*-
cd /userhome/fairseq/fairseq
pip install --editable ./

pip install fairscale

python setup.py build_ext --inplace

cd  /userhome/fairseq/fastmoe-master
# USE_NCCL=1 python setup.py install
USE_NCCL=1 pip install --editable ./