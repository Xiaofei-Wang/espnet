#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import argparse
import logging
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, help='json files', default='./data.json')
    parser.add_argument('--avgl2att', type=str, help='average l2 (stream) level attention (weight for first stream)', default='./avgl2att')
    args = parser.parse_args()
    
    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    logging.info("reading %s", args.json)
    with open(args.json, 'r') as f:
        j = json.load(f)

    l2att_ws_total = 0
    l2att_ws_cnts = 0

    for uttname in j['utts'].keys():
        ws = np.array([float(i) for i in j['utts'][uttname]['output'][0]['ref_l2att'].split()])
        l2att_ws_total += np.sum(ws)
        l2att_ws_cnts += len(ws)
    avgl2att = l2att_ws_total/l2att_ws_cnts
    print(avgl2att)

    logging.info("writing average l2 attention weight to %s", args.avgl2att)
    with open(args.avgl2att, 'w') as avgl2att_file:
        avgl2att_file.write("{}".format(avgl2att))
