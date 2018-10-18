#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Ruizhi Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# torch related
import torch

# espnet related
from asr_utils import get_model_conf
from asr_utils import torch_load
from e2e_asr_th import E2E
from e2e_asr_th import Loss


import argparse
import logging


from torch.nn.modules.module import _addindent

def summary(model):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        total_params = sum(p.numel() for p in model.parameters())
        main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    main_str, count = repr(model)
    return main_str, count

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Model file parameters to read')
    parser.add_argument('--model-conf', type=str, default=None,
                        help='Model config file')
    parser.add_argument('--outfile', type=str, default=None,
                        help='Output file to write the informations about the model')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # read training config
    logging.info("reading model config %s", args.model_conf)
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    e2e = E2E(idim, odim, train_args)
    model = Loss(e2e, train_args.mtlalpha)
    torch_load(args.model, model)

    main_str, count = summary(model)

    with open(args.outfile, 'w') as f:
        f.write(main_str)