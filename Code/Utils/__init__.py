'''
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
=================================================
@Project -> File   ：GraphSum -> __init__.py
@Author ：MollyShuu
@Date   ：2021/4/22 14:57
@IDE    ：PyCharm
==================================================
'''

import torch.cuda

from Utils.metric import calculate_bleu, file_bleu
from Utils.saver import Saver
from Utils.file_utils import *


def get_device():
    if torch.cuda.is_available():
        print('Using GPU!')
        return torch.device('cuda:0')
    else:
        print('Using CPU!')
        return torch.device('cpu')


def printing_opt(opt):
    return "\n".join(["%15s | %s" % (e[0], e[1]) for e in sorted(vars(opt).items(), key=lambda x: x[0])])
