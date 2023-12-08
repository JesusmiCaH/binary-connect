import torch
import torch.nn as nn
import os
import copy


if __name__ == '__main__':
    sample_array = (torch.rand([64,4,4]) - 0.5)>0

    with open('./saved_weight.txt', "w", encoding='utf-8') as f:
        for arr_2d in sample_array:
            f.write('{\n')
            for arr_1d in arr_2d:
                f.write('\t{')
                for value in arr_1d:
                    string = '%s, '% value.item()
                    f.write(string.lower())
                f.write('},\n')
            f.write('},\n')