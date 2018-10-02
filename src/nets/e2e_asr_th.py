#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from __future__ import division

import logging
import math
import sys

from argparse import Namespace

import chainer
import numpy as np
import six
import torch
import torch.nn.functional as F
import torch.distributions as tdist
import warpctc_pytorch as warp_ctc
import asr_utils

from chainer import reporter
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from ctc_prefix_score import CTCPrefixScore
from e2e_asr_common import end_detect
from e2e_asr_common import get_vgg2l_odim, get_maxpooling2_odim, get_RCNN_ilens, get_RCNN_odim
from e2e_asr_common import label_smoothing_dist


CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5
MAX_DECODER_OUTPUT = 5


# ------------- Utility functions --------------------------------------------------------------------------------------
def to_cuda(m, x):
    """Function to send tensor into corresponding device

    :param torch.nn.Module m: torch module
    :param torch.Tensor x: torch tensor
    :return: torch tensor located in the same place as torch module
    :rtype: torch.Tensor
    """
    assert isinstance(m, torch.nn.Module)
    device = next(m.parameters()).device
    return x.to(device)


def pad_list(xs, pad_value):
    """Function to pad values

    :param list xs: list of torch.Tensor [(L_1, D), (L_2, D), ..., (L_B, D)]
    :param float pad_value: value for padding
    :return: padded tensor (B, Lmax, D)
    :rtype: torch.Tensor
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


def make_pad_mask(lengths):
    """Function to make mask tensor containing indices of padded part

    e.g.: lengths = [5, 3, 2]
          mask = [[0, 0, 0, 0 ,0],
                  [0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 1]]

    :param list lengths: list of lengths (B)
    :return: mask tensor containing indices of padded part (B, Tmax)
    :rtype: torch.Tensor
    """
    bs = int(len(lengths))
    maxlen = int(max(lengths))
    mask = torch.zeros(bs, maxlen, dtype=torch.uint8)
    for i, l in enumerate(lengths):
        mask[i, l:] = 1

    return mask

def fill_padded_part(xs, ilens, fill_value):
    """Fucntion to fill padded part with selected value

    :param torch.Tensor xs: tensor (B, Tmax, ...)
    :param torch.Tensor ilens:  list of lengths (B)
    :param float fill_value:  value to fill padded part
    :return: xs whose padded parts are filled by fill_value
    """
    assert xs.size(0) == len(ilens)
    new_xs = xs.new(*xs.size()).fill_(fill_value)
    for idx, l in enumerate(ilens):
        new_xs[idx, :l] = xs[idx, :l]

    return new_xs


def th_accuracy(pad_outputs, pad_targets, ignore_label):
    """Function to calculate accuracy

    :param torch.Tensor pad_outputs: prediction tensors (B*Lmax, D)
    :param torch.Tensor pad_targets: target tensors (B, Lmax, D)
    :param int ignore_label: ignore label id
    :retrun: accuracy value (0.0 - 1.0)
    :rtype: float
    """
    pad_pred = pad_outputs.view(
        pad_targets.size(0),
        pad_targets.size(1),
        pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)


class Reporter(chainer.Chain):
    def report(self, loss_ctc, loss_att, acc, mtl_loss):
        reporter.report({'loss_ctc': loss_ctc}, self)
        reporter.report({'loss_att': loss_att}, self)
        reporter.report({'acc': acc}, self)
        logging.info('mtl loss:' + str(mtl_loss))
        reporter.report({'loss': mtl_loss}, self)


# TODO(watanabe) merge Loss and E2E: there is no need to make these separately
class Loss(torch.nn.Module):
    """Multi-task learning loss module

    :param torch.nn.Module predictor: E2E model instance
    :param float mtlalpha: mtl coefficient value (0.0 ~ 1.0)
    """

    def __init__(self, predictor, mtlalpha):
        super(Loss, self).__init__()
        assert 0.0 <= mtlalpha <= 1.0, "mtlalpha shoule be [0.0, 1.0]"
        self.mtlalpha = mtlalpha
        self.loss = None
        self.accuracy = None
        self.predictor = predictor
        self.reporter = Reporter()

    def forward(self, xs_pad, ilens, ys_pad):
        '''Multi-task learning loss forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        '''
        self.loss = None
        loss_ctc, loss_att, acc = self.predictor(xs_pad, ilens, ys_pad)
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_ctc_data, loss_att_data, acc, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        return self.loss


class E2E(torch.nn.Module):
    """E2E module

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param namespace args: argument namespace containing options
    """

    def __init__(self, idim, odim, args):
        super(E2E, self).__init__()
        self.etype = args.etype
        self.verbose = args.verbose
        self.char_list = args.char_list
        self.outdir = args.outdir
        self.mtlalpha = args.mtlalpha
        self.num_enc = args.num_enc
        self.share_ctc = args.share_ctc

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1

        # subsample info
        # +1 means input (+1) and layers outputs (args.elayer)
        subsample = np.ones(args.elayers + 1, dtype=np.int)
        if args.etype in ['blstmp', 'blstmpbn', 'multiVggblstmBlstmp', 'multiVggdil2blstmBlstmp',
                          'multiBandBlstmpBlstmp', 'highBandBlstmp', 'lowBandBlstmp', 'multiVgg8blstmBlstmp',
                          'multiVggblstmpBlstmp', 'multiVggblstmpBlstmpFixed4', 'multiVggblstmBlstmpFixed4',
                          'amiCH1BlstmpCH2Blstmp', 'amiCH1Blstmp', 'amiCH2Blstmp']:
            ss = args.subsample.split("_")
            for j in range(min(args.elayers + 1, len(ss))):
                subsample[j] = int(ss[j])
        else:
            logging.warning(
                'Subsampling is not performed for vgg*. It is performed in max pooling layers at CNN.')
        logging.info('subsample: ' + ' '.join([str(x) for x in subsample]))
        self.subsample = subsample

        # label smoothing info
        if args.lsm_type:
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(odim, args.lsm_type, transcript=args.train_json)
        else:
            labeldist = None

        # encoder
        if isinstance(args, asr_utils.AttributeDict): # decoding stage
            # training from scratch: args is namespace;
            # training from model: args is namespace
            # decoding: args is asr_utils.AttributeDict object
            self.enc = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs,
                               self.subsample, args.dropout_rate, addgauss=args.addgauss, addgauss_mean=args.addgauss_mean, addgauss_std=args.addgauss_std,
                               addgauss_type=args.addgauss_type)
        else:
            self.enc = Encoder(args.etype, idim, args.elayers, args.eunits, args.eprojs,
                           self.subsample, args.dropout_rate)
        # ctc
        self.ctc = CTC(odim, args.eprojs, args.dropout_rate, num_enc=args.num_enc, share_ctc=self.share_ctc)
        # attention
        if args.atype == 'noatt':
            self.att = NoAtt()
        elif args.atype == 'dot':
            self.att = AttDot(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'add':
            self.att = AttAdd(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'location':
            self.att = AttLoc(args.eprojs, args.dunits,
                              args.adim, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'location2d':
            self.att = AttLoc2D(args.eprojs, args.dunits,
                                args.adim, args.awin, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'location_recurrent':
            self.att = AttLocRec(args.eprojs, args.dunits,
                                 args.adim, args.aconv_chans, args.aconv_filts)
        elif args.atype == 'coverage':
            self.att = AttCov(args.eprojs, args.dunits, args.adim)
        elif args.atype == 'coverage_location':
            self.att = AttCovLoc(args.eprojs, args.dunits, args.adim,
                                 args.aconv_chans, args.aconv_filts)
        elif args.atype == 'multi_head_dot':
            self.att = AttMultiHeadDot(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim)
        elif args.atype == 'multi_head_add':
            self.att = AttMultiHeadAdd(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim)
        elif args.atype == 'multi_head_loc':
            self.att = AttMultiHeadLoc(args.eprojs, args.dunits,
                                       args.aheads, args.adim, args.adim,
                                       args.aconv_chans, args.aconv_filts)
        elif args.atype == 'multi_head_multi_res_loc':
            self.att = AttMultiHeadMultiResLoc(args.eprojs, args.dunits,
                                               args.aheads, args.adim, args.adim,
                                               args.aconv_chans, args.aconv_filts)
        elif args.atype == 'enc2_add':
            self.att = Enc2AttAdd(args.eprojs, args.dunits,
                                  args.adim, num_enc=args.num_enc)
        elif args.atype == 'enc2_add_l2w0.5':
            if 'l2_weight' not in vars(args):  # training stage
                self.att = Enc2AttAdd(args.eprojs, args.dunits,
                                      args.adim, num_enc=args.num_enc, l2_weight=0.5, l2_dropout=False)
            else:  # decoding stage
                self.att = Enc2AttAdd(args.eprojs, args.dunits,
                                      args.adim, num_enc=args.num_enc, l2_weight=args.l2_weight, l2_dropout=False)
        elif args.atype == 'enc2_add_l2dp':
            self.att = Enc2AttAdd(args.eprojs, args.dunits,
                                  args.adim, num_enc=args.num_enc, l2_dropout=args.l2_dropout)

        elif args.atype == 'enc2_add_linproj':
            self.att = Enc2AttAddLinProj(args.eprojs, args.dunits,
                                         args.adim, num_enc=args.num_enc, l2_dropout=False, l2_weight=0.5,
                                         apply_tanh=False)
        elif args.atype == 'enc2_add_linprojtanh':
            self.att = Enc2AttAddLinProj(args.eprojs, args.dunits,
                                         args.adim, num_enc=args.num_enc, l2_dropout=False, l2_weight=0.5,
                                         apply_tanh=True)
        elif args.atype == 'enc2_add_addlinproj':
            self.att = Enc2AttAddLinProj(args.eprojs, args.dunits,
                                         args.adim, num_enc=args.num_enc, l2_dropout=False, l2_weight=None,
                                         apply_tanh=False)
        elif args.atype == 'enc2_add_addlinprojtanh':
            self.att = Enc2AttAddLinProj(args.eprojs, args.dunits,
                                         args.adim, num_enc=args.num_enc, l2_dropout=False, l2_weight=None,
                                         apply_tanh=True)
        elif args.atype == 'enc2_none_frmaddlinproj':
            self.att = Enc2AttAddFrmLinProj(args.eprojs, args.dunits,
                                            args.adim, num_enc=args.num_enc, l2_dropout=False, l1_weight=1.0,
                                            l2_stream_weight=1.0, apply_tanh=False)
        elif args.atype == 'enc2_none_frmaddlinprojtanh':
            self.att = Enc2AttAddFrmLinProj(args.eprojs, args.dunits,
                                            args.adim, num_enc=args.num_enc, l2_dropout=False, l1_weight=1.0,
                                            l2_stream_weight=1.0, apply_tanh=True)
        elif args.atype == 'enc2_none_frmaddaddlinprojtanh':
            self.att = Enc2AttAddFrmLinProj(args.eprojs, args.dunits,
                                            args.adim, num_enc=args.num_enc, l2_dropout=False, l1_weight=1.0,
                                            l2_stream_weight=None, apply_tanh=True)
        elif args.atype == 'enc2_add_frmaddlinproj':
            self.att = Enc2AttAddFrmLinProj(args.eprojs, args.dunits,
                                            args.adim, num_enc=args.num_enc, l2_dropout=False, l1_weight=None,
                                            l2_stream_weight=1.0, apply_tanh=False)
        elif args.atype == 'enc2_add_frmaddlinprojtanh':
            self.att = Enc2AttAddFrmLinProj(args.eprojs, args.dunits,
                                            args.adim, num_enc=args.num_enc, l2_dropout=False, l1_weight=None,
                                            l2_stream_weight=1.0, apply_tanh=True)
        elif args.atype == 'enc2_add_frmaddaddlinprojtanh':
            self.att = Enc2AttAddFrmLinProj(args.eprojs, args.dunits,
                                            args.adim, num_enc=args.num_enc, l2_dropout=False, l1_weight=None,
                                            l2_stream_weight=None, apply_tanh=True)
        else:
            logging.error(
                "Error: need to specify an appropriate attention archtecture")
            sys.exit()
        # decoder
        self.dec = Decoder(args.eprojs, odim, args.dlayers, args.dunits,
                           self.sos, self.eos, self.att, self.verbose, self.char_list,
                           labeldist, args.lsm_weight, num_enc=args.num_enc)

        # weight initialization
        self.init_like_chainer()

    def init_like_chainer(self):
        """Initialize weight like chainer

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)

        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """

        def lecun_normal_init_parameters(module):
            # for p in module.parameters():
            for name, p in module.named_parameters():
                data = p.data
                if 'batchnorm' in name and 'weight' in name:
                    data.uniform_()
                elif data.dim() == 1:
                    # bias
                    data.zero_()
                elif data.dim() == 2:
                    # linear weight
                    n = data.size(1)
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                elif data.dim() == 4:
                    # conv weight
                    n = data.size(1)
                    for k in data.size()[2:]:
                        n *= k
                    stdv = 1. / math.sqrt(n)
                    data.normal_(0, stdv)
                else:
                    raise NotImplementedError

        def set_forget_bias_to_one(bias):
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(1.)

        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        self.dec.embed.weight.data.normal_(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
        for l in six.moves.range(len(self.dec.decoder)):
            set_forget_bias_to_one(self.dec.decoder[l].bias_ih)

    def forward(self, xs_pad, ilens, ys_pad):
        '''E2E forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        '''
        # 1. encoder
        hs_pad, hlens = self.enc(xs_pad, ilens)

        # 2. CTC loss
        if self.mtlalpha == 0:
            loss_ctc = None
        else:
            loss_ctc = self.ctc(hs_pad, hlens, ys_pad)

        # 3. attention loss
        if self.mtlalpha == 1:
            loss_att = None
            acc = None
        else:
            loss_att, acc = self.dec(hs_pad, hlens, ys_pad)

        return loss_ctc, loss_att, acc

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        '''E2E beam search

        :param ndarray x: input acouctic feature (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        '''
        prev = self.training
        self.eval()
        # subsample frame
        x = x[::self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_cuda(self, torch.from_numpy(
            np.array(x, dtype=np.float32)))

        # 1. encoder
        # make a utt list (1) to use the same interface for encoder
        h, _ = self.enc(h.unsqueeze(0), ilen)

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            if self.num_enc == 1:
                lpz = self.ctc.log_softmax(h)[0]
            else:
                lpz = [self.ctc.log_softmax(h)[idx][0] for idx in range(self.num_enc)]
        else:
            lpz = None

        # 2. decoder
        # decode the first utterance
        y = self.dec.recognize_beam(h, lpz, recog_args, char_list, rnnlm)

        if prev:
            self.train()
        return y

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        '''E2E attention calculation

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        '''
        with torch.no_grad():
            # encoder
            hpad, hlens = self.enc(xs_pad, ilens)

            # decoder
            att_ws = self.dec.calculate_all_attentions(hpad, hlens, ys_pad)

        return att_ws


# ------------- CTC Network --------------------------------------------------------------------------------------------
class CTC(torch.nn.Module):
    """CTC module

    :param int odim: dimension of outputs
    :param int eprojs: number of encoder projection units
    :param float dropout_rate: dropout rate (0.0 ~ 1.0)
    """

    def __init__(self, odim, eprojs, dropout_rate, num_enc, share_ctc=True):
        super(CTC, self).__init__()
        self.dropout_rate = dropout_rate
        self.loss = None
        if share_ctc:
            self.ctc_lo = torch.nn.Linear(eprojs, odim)
        else:
            for idx in range(num_enc):
                setattr(self, "ctc_lo_%d" % idx, torch.nn.Linear(eprojs, odim))
        self.loss_fn = warp_ctc.CTCLoss(size_average=True)
        self.ignore_id = -1
        self.num_enc = num_enc
        self.share_ctc = share_ctc

    def forward(self, hs_pad, hlens, ys_pad):
        '''CTC forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        '''
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

        self.loss = None
        if self.num_enc == 1:
            hlens = torch.from_numpy(np.fromiter(hlens, dtype=np.int32))
            olens = torch.from_numpy(np.fromiter(
                (x.size(0) for x in ys), dtype=np.int32))

            # zero padding for hs
            ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))

            # zero padding for ys
            ys_true = torch.cat(ys).cpu().int()  # batch x olen

            # get length info
            logging.info(self.__class__.__name__ + ' input lengths:  ' + ''.join(str(hlens).split('\n')))
            logging.info(self.__class__.__name__ + ' output lengths: ' + ''.join(str(olens).split('\n')))

            # get ctc loss
            # expected shape of seqLength x batchSize x alphabet_size
            ys_hat = ys_hat.transpose(0, 1)
            self.loss = to_cuda(self, self.loss_fn(ys_hat, ys_true, hlens, olens))
            logging.info('ctc loss:' + str(float(self.loss)))

        else:  # multi-encoder case
            assert len(hs_pad) == len(hlens)
            hlens = [torch.from_numpy(np.fromiter(hlens[idx], dtype=np.int32)) for idx in range(self.num_enc)]
            olens = torch.from_numpy(np.fromiter(
                (x.size(0) for x in ys), dtype=np.int32))

            # zero padding for hs
            if self.share_ctc:
                ys_hat = [self.ctc_lo(F.dropout(hs_pad[idx], p=self.dropout_rate)) for idx
                          in range(self.num_enc)]
            else:
                ys_hat = [getattr(self, "ctc_lo_%d" % idx)(F.dropout(hs_pad[idx], p=self.dropout_rate)) for idx
                          in range(self.num_enc)]

            # zero padding for ys
            ys_true = torch.cat(ys).cpu().int()  # batch x olen

            # get length info
            [logging.info(self.__class__.__name__ + ' input lengths:  ' + ''.join(str(hlens[idx]).split('\n'))) for idx
             in range(self.num_enc)]
            logging.info(self.__class__.__name__ + ' output lengths: ' + ''.join(str(olens).split('\n')))

            # get ctc loss
            # expected shape of seqLength x batchSize x alphabet_size
            ys_hat = [ys_hat[idx].transpose(0, 1) for idx in range(self.num_enc)]
            self.loss = [to_cuda(self, self.loss_fn(ys_hat[idx], ys_true, hlens[idx], olens)) for idx in
                         range(self.num_enc)]
            self.loss = torch.mean(torch.cat(self.loss, dim=0))
            # TODO: Implement different ctc weight for each stream
            logging.info('ctc loss:' + str(float(self.loss)))

        return self.loss

    def log_softmax(self, hs_pad):
        '''log_softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        '''
        if self.num_enc == 1:
            return F.log_softmax(self.ctc_lo(hs_pad), dim=2)
        if self.share_ctc:
            return [F.log_softmax(self.ctc_lo(hs_pad[idx]), dim=2) for idx in range(self.num_enc)]
        else:
            return [F.log_softmax(getattr(self, "ctc_lo_%d" % idx)(hs_pad[idx]), dim=2) for idx in range(self.num_enc)]


# ------------- Attention Network --------------------------------------------------------------------------------------
class NoAtt(torch.nn.Module):
    '''No attention'''

    def __init__(self):
        super(NoAtt, self).__init__()
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.c = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''NoAtt forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B, T_max, D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weights
        :rtype: torch.Tensor
        '''
        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [enc_hs_pad.new(l).fill_(1.0 / l) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev = pad_list(att_prev, 0)
            self.c = torch.sum(self.enc_h * att_prev.view(batch, self.h_length, 1), dim=1)

        return self.c, att_prev


class AttDot(torch.nn.Module):
    '''Dot product attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    '''

    def __init__(self, eprojs, dunits, att_dim):
        super(AttDot, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: dummy (does not use)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weight (B x T_max)
        :rtype: torch.Tensor
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = torch.tanh(self.mlp_enc(self.enc_h))

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        e = torch.sum(self.pre_compute_enc_h * torch.tanh(self.mlp_dec(dec_z)).view(batch, 1, self.att_dim),
                      dim=2)  # utt x frame

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
        return c, w


class AttAdd(torch.nn.Module):
    '''Additive attention

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    '''

    def __init__(self, eprojs, dunits, att_dim):
        super(AttAdd, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)
        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weights (B x T_max)
        :rtype: torch.Tensor
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        # c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
        c = torch.matmul(w.unsqueeze(1), self.enc_h).squeeze(1)
        return c, w


class AttLoc(torch.nn.Module):
    '''location-aware attention

    Reference: Attention-Based Models for Speech Recognition
        (https://arxiv.org/pdf/1506.07503.pdf)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.aconv_chans = aconv_chans

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attetion weight (B x T_max)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weights (B x T_max)
        :rtype: torch.Tensor
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev = [enc_hs_pad.new(l).fill_(1.0 / l) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev = pad_list(att_prev, 0)

        # att_prev: utt x frame -> utt x 1 x 1 x frame -> utt x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE equivalent to c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
        c = torch.matmul(w.unsqueeze(1), self.enc_h).squeeze(1)

        return c, w


class Enc2AttAdd(torch.nn.Module):
    '''add attention with 2 encoder streams

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int k : attention dimension
    :param int num_enc: # of encoder stream
    :param int l2_dropout: flag to apply level-2 dropout
    :param int l2_weight: fix the first encoder with l2_weight, second one has 1-l2_weight. default: None, do adaptive l2_weight learning.
    '''

    # __init__ will define the architecture of the attention model
    # chnage the l2Weight when recog
    def __init__(self, eprojs, dunits, att_dim, num_enc=2, l2_dropout=None, l2_weight=None):
        super(Enc2AttAdd, self).__init__()

        # level 1 attention: one attention mechanism for each stream

        for idx in range(num_enc):
            setattr(self, "mlp_enc%d_l1" % idx, torch.nn.Linear(eprojs, att_dim))
            setattr(self, "mlp_dec%d_l1" % idx, torch.nn.Linear(dunits, att_dim, bias=False))
            setattr(self, "gvec%d_l1" % idx, torch.nn.Linear(att_dim, 1))

        if l2_weight is None:  # fixed l2 weight
            # level 2 attention: one attention mechanism for stream selection
            self.mlp_enc_l2 = torch.nn.Linear(eprojs, att_dim)
            self.mlp_dec_l2 = torch.nn.Linear(dunits, att_dim, bias=False)
            self.gvec_l2 = torch.nn.Linear(att_dim, 1)

        if l2_dropout is not None:
            self.dropout_l2 = torch.nn.Dropout2d(p=l2_dropout)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length_l1 = None
        self.h_length_l2 = None
        self.enc_h_l1 = None
        self.enc_h_l2 = None
        self.pre_compute_enc_h_l1 = None
        self.pre_compute_enc_h_l2 = None
        self.num_enc = num_enc
        self.l2_weight = l2_weight
        self.l2_dropout = l2_dropout
        self.l2_mask = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length_l1 = None
        self.h_length_l2 = None
        self.enc_h_l1 = None
        self.enc_h_l2 = None
        self.pre_compute_enc_h_l1 = None
        self.pre_compute_enc_h_l2 = None
        self.l2_mask =None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttLoc forward

        :param Variable enc_hs_pad: list of padded encoder hidden state (list(B x T_max x D_enc))
        :param list enc_h_len: list of padded encoder hidden state lenght list((B))
        :param Variable dec_z: docoder hidden state (B x D_dec)
        :param Variable att_prev: list of previous attetion weight list(B x T_max)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: Variable
        :return: list of previous attentioin weights [list(B x T_max), B x T_max] --> l1 and l2
        :rtype: Variable
        '''

        # level 1 attention
        batch = len(enc_hs_pad[0])
        # pre-compute all h outside the decoder loop

        if self.pre_compute_enc_h_l1 is None:
            self.enc_h_l1 = enc_hs_pad  # list (utt x frame x hdim)
            self.h_length_l1 = [self.enc_h_l1[idx].size(1) for idx in range(self.num_enc)]
            # utt x frame x att_dim
            self.pre_compute_enc_h_l1 = [getattr(self, "mlp_enc%d_l1" % idx)(self.enc_h_l1[idx]) for idx in
                                         range(self.num_enc)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev_l1 = None

            if self.l2_weight is None:
                att_prev_l2 = [enc_hs_pad[0].new(self.num_enc).fill_(1.0 / self.num_enc) for _ in range(batch)]

            else:  # fixed l2 weight
                att_prev_l2 = []
                w_np = np.array([self.l2_weight, 1 - self.l2_weight])  # hard coded for two encoders
                for _ in range(batch):
                    w = torch.from_numpy(w_np).type(enc_hs_pad[0].type())
                    att_prev_l2 += [w]

            att_prev_l2 = pad_list(att_prev_l2, 0)  # utt x frame_max
            att_prev = [att_prev_l1, att_prev_l2]  # [[att_l1_1, att_l1_2, ...], att_l2_1]

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled_l1 = [getattr(self, "mlp_dec%d_l1" % idx)(dec_z).view(batch, 1, self.att_dim) for idx in
                          range(self.num_enc)]

        # dot with gvec
        # list(utt x frame x att_dim) -> list(utt x frame)
        # NOTE consider zero padding when compute w.
        e_l1 = [getattr(self, "gvec%d_l1" % idx)(torch.tanh(
            self.pre_compute_enc_h_l1[idx] + dec_z_tiled_l1[idx])).squeeze(2) for idx in range(self.num_enc)]

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = [to_cuda(self, make_pad_mask(enc_hs_len[idx]))  for idx in range(self.num_enc)]
        e_l1 = [e_l1[idx].masked_fill_(self.mask[idx], -float('inf')) for idx in range(self.num_enc)]


        w_l1 = [F.softmax(scaling * e_l1[idx], dim=1) for idx in range(self.num_enc)]

        # weighted sum over flames
        # list (utt x hdim)
        # NOTE equivalent to c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
        # self.enc_h_l2 = [torch.sum(self.enc_h_l1[idx] * w_l1[idx].view(batch, self.h_length_l1[idx], 1), dim=1) for idx in
        #                  range(self.num_enc)]
        self.enc_h_l2 = [torch.matmul(w_l1[idx].unsqueeze(1), self.enc_h_l1[idx]).squeeze(1) for idx in
                         range(self.num_enc)]
        self.enc_h_l2 = torch.stack(self.enc_h_l2, dim=1)  # utt x numEncStream x hdim

        if self.l2_dropout is not None:  # TODO: (0,0) situation
            self.enc_h_l2 = self.dropout_l2(self.enc_h_l2.unsqueeze(3)).squeeze(3)

        # level 2 attention
        if self.l2_weight is None:
            self.h_length_l2 = self.num_enc
            self.pre_compute_enc_h_l2 = self.mlp_enc_l2(self.enc_h_l2)

            # dec_z_tiled: utt x frame x att_dim
            dec_z_tiled_l2 = self.mlp_dec_l2(dec_z).view(batch, 1, self.att_dim)

            # dot with gvec
            # utt x frame x att_dim -> utt x frame
            # NOTE consider zero padding when compute w.
            e_l2 = self.gvec_l2(torch.tanh(
                self.pre_compute_enc_h_l2 + dec_z_tiled_l2)).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.l2_mask is None:
                self.l2_mask = to_cuda(self, make_pad_mask([self.num_enc]))
            e_l2.masked_fill_(self.l2_mask, -float('inf'))

            w_l2 = F.softmax(scaling * e_l2, dim=1)
        else:  # fixed l2 weight
            w_l2 = att_prev[1]

        # weighted sum over flames
        # utt x hdim
        # NOTE equivalent to c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
        # c_l2 = torch.sum(self.enc_h_l2 * w_l2.view(batch, self.h_length_l2, 1), dim=1)
        c_l2 = torch.matmul(w_l2.unsqueeze(1), self.enc_h_l2).squeeze(1)
        logging.info(self.__class__.__name__ + ' level two attention weight: ')
        logging.warning(w_l2[0][0])  # print the l2-weight for the first stream

        return c_l2, [w_l1, w_l2]


class Enc2AttAddLinProj(torch.nn.Module): # TODO: need to be updated with the new espnet
    '''context vector level fusion with 2 encoder streams
    level-1 (stream level): H1=AddAttention1, H2=AddAttention2
    level-2 (fusion level): H=[RELU*](W*concat([a1]*H1+[a2]*H2))
        RELU is optional, [a1,a2] is optional

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int num_enc: # of encoder stream
    :param int l2_dropout: flag to apply level-2 dropout
    :param int l2_weight: fix the first encoder with l2_weight, second one has 1-l2_weight. default: None, do adaptive l2_weight learning.
    '''

    # __init__ will define the architecture of the attention model
    # chnage the l2Weight when recog
    def __init__(self, eprojs, dunits, att_dim, num_enc=2, l2_dropout=False, l2_weight=None, apply_tanh=False):
        super(Enc2AttAddLinProj, self).__init__()

        # level 1 attention: one attention mechanism for each stream

        for idx in range(num_enc):
            setattr(self, "mlp_enc%d_l1" % idx, torch.nn.Linear(eprojs, att_dim))
            setattr(self, "mlp_dec%d_l1" % idx, torch.nn.Linear(dunits, att_dim, bias=False))
            setattr(self, "gvec%d_l1" % idx, torch.nn.Linear(att_dim, 1))

        self.proj = torch.nn.Linear(num_enc * eprojs, eprojs)  # TODO: more projection layer

        if l2_weight is None:  # fixed l2 weight
            # level 2 attention: one attention mechanism for stream selection
            self.mlp_enc_l2 = torch.nn.Linear(eprojs, att_dim)
            self.mlp_dec_l2 = torch.nn.Linear(dunits, att_dim, bias=False)
            self.gvec_l2 = torch.nn.Linear(att_dim, 1)

        if l2_dropout:
            self.dropout_l2 = torch.nn.Dropout2d(p=0.5)

        if apply_tanh:
            self.tanh = torch.nn.Tanh()

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length_l1 = None
        self.h_length_l2 = None
        self.enc_h_l1 = None
        self.enc_h_l2 = None
        self.pre_compute_enc_h_l1 = None
        self.pre_compute_enc_h_l2 = None
        self.num_enc = num_enc
        self.l2_weight = l2_weight
        self.l2_dropout = l2_dropout
        self.apply_tanh = apply_tanh

    def reset(self):
        '''reset states'''
        self.h_length_l1 = None
        self.h_length_l2 = None
        self.enc_h_l1 = None
        self.enc_h_l2 = None
        self.pre_compute_enc_h_l1 = None
        self.pre_compute_enc_h_l2 = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''Enc2AttAddLinProj forward

        :param Variable enc_hs_pad: list of padded encoder hidden state (list(B x T_max x D_enc))
        :param list enc_h_len: list of padded encoder hidden state lenght list((B))
        :param Variable dec_z: docoder hidden state (B x D_dec)
        :param Variable att_prev: list of previous attetion weight list(B x T_max)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: Variable
        :return: list of previous attentioin weights [list(B x T_max), B x T_max] --> l1 and l2
        :rtype: Variable
        '''

        # level 1 attention
        batch = len(enc_hs_pad[0])

        # pre-compute all h outside the decoder loop

        if self.pre_compute_enc_h_l1 is None:
            self.enc_h_l1 = enc_hs_pad  # list (utt x frame x hdim)
            self.h_length_l1 = [self.enc_h_l1[idx].size(1) for idx in range(self.num_enc)]
            # utt x frame x att_dim
            self.pre_compute_enc_h_l1 = [getattr(self, "mlp_enc%d_l1" % idx)(self.enc_h_l1[idx]) for idx in
                                         range(self.num_enc)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            att_prev_l1 = None

            if self.l2_weight is None:
                att_prev_l2 = [enc_hs_pad[0].new(self.num_enc).fill_(1.0 / self.num_enc) for _ in range(batch)]
            else:  # fixed l2 weight
                att_prev_l2 = []
                w_np = np.array([self.l2_weight, 1 - self.l2_weight])  # hard coded for two encoders
                for _ in range(batch):
                    w = torch.from_numpy(w_np).type(enc_hs_pad[0].type())
                    att_prev_l2 += [w]

            att_prev_l2 = pad_list(att_prev_l2, 0)  # utt x frame_max
            att_prev = [att_prev_l1, att_prev_l2]  # [[att_l1_1, att_l1_2, ...], att_l2_1]

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled_l1 = [getattr(self, "mlp_dec%d_l1" % idx)(dec_z).view(batch, 1, self.att_dim) for idx in
                          range(self.num_enc)]

        # dot with gvec
        # list(utt x frame x att_dim) -> list(utt x frame)
        # NOTE consider zero padding when compute w.
        e_l1 = [getattr(self, "gvec%d_l1" % idx)(torch.tanh(
            self.pre_compute_enc_h_l1[idx] + dec_z_tiled_l1[idx])).squeeze(2) for idx in range(self.num_enc)]
        w_l1 = [F.softmax(scaling * e_l1[idx], dim=1) for idx in range(self.num_enc)]

        # weighted sum over flames
        # list (utt x hdim)
        # NOTE equivalent to c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)
        # self.enc_h_l2 = [torch.sum(self.enc_h_l1[idx] * w_l1[idx].view(batch, self.h_length_l1[idx], 1), dim=1) for idx in range(self.num_enc)]
        self.enc_h_l2 = [torch.matmul(w_l1[idx].unsqueeze(1), self.enc_h_l1[idx]).squeeze(1) for idx in
                         range(self.num_enc)]
        self.enc_h_l2 = torch.stack(self.enc_h_l2, dim=1)  # utt x numEncStream x hdim

        if self.l2_dropout:  # TODO: (0,0) situation
            self.enc_h_l2 = self.dropout_l2(self.enc_h_l2.unsqueeze(3)).squeeze(3)

        # level 2 attention
        if self.l2_weight is None:
            self.h_length_l2 = self.num_enc
            self.pre_compute_enc_h_l2 = self.mlp_enc_l2(self.enc_h_l2)

            # dec_z_tiled: utt x frame x att_dim
            dec_z_tiled_l2 = self.mlp_dec_l2(dec_z).view(batch, 1, self.att_dim)

            # dot with gvec
            # utt x frame x att_dim -> utt x frame
            # NOTE consider zero padding when compute w.
            e_l2 = self.gvec_l2(torch.tanh(
                self.pre_compute_enc_h_l2 + dec_z_tiled_l2)).squeeze(2)
            w_l2 = F.softmax(scaling * e_l2, dim=1)
        else:  # fixed l2 weight
            w_l2 = att_prev[1]

        # w_l2: utt x numEnc
        # enc_h_l2:  utt x numEnc x hdim
        self.enc_h_l2 = w_l2.unsqueeze(2) * self.enc_h_l2  # utt x num_enc x T_max
        # utt x hdim
        c_l2 = self.proj(self.enc_h_l2.view(batch, -1))

        if self.apply_tanh: c_l2 = self.tanh(c_l2)

        logging.info(self.__class__.__name__ + ' level two attention weight: ')
        logging.warning(w_l2[0][0])  # print the l2-weight for the first stream

        return c_l2, [w_l1, w_l2]


class Enc2AttAddFrmLinProj(torch.nn.Module): # TODO: need to be updated with the new espnet
    '''frame level fusion with 2 encoder streams
    level-1 (stream level): calculate frame level attention vector at1 at2
    level-2 (fusion level):
        ht=[Tanh]*(W*concat([bt1]*[at1]*ht1+[bt2]*[at2]*ht2))
        H=AddAttention(ht)

    Optional: Tanh, [a1], [a2], [b1, b2]
        attention across T [a1], [a2] and attention across streams [b1,b2]

    There are three attention here:
    a1 = AddAttention(h1,z), a2 = AddAtention(h2, z)
    b1,b2=AddAttention(h1,h2,z)
    H=AddAttention(concat(h1,h2),z)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int num_enc: # of encoder stream
    :param int l2_dropout: flag to apply level-2 dropout
    :param int l1_weight: fix the all a1,a2 attention with l1_weight, default: None, do adaptive l1_weight learning.
    :param int l2_stream_weight: fix the all b1,b2 attention with l2_stream_weight, default: None, do adaptive l2_stream_weight learning.
    '''

    # __init__ will define the architecture of the attention model
    # chnage the l2Weight when recog
    def __init__(self, eprojs, dunits, att_dim, num_enc=2, l2_dropout=False, l1_weight=None, l2_stream_weight=None,
                 apply_tanh=False):
        super(Enc2AttAddFrmLinProj, self).__init__()

        # level 1 attention: one attention mechanism for each stream
        if l1_weight is None:
            for idx in range(num_enc):
                setattr(self, "mlp_enc%d_l1" % idx, torch.nn.Linear(eprojs, att_dim))
                setattr(self, "mlp_dec%d_l1" % idx, torch.nn.Linear(dunits, att_dim, bias=False))
                setattr(self, "gvec%d_l1" % idx, torch.nn.Linear(att_dim, 1))

        # level 2 attention
        if l2_stream_weight is None:
            setattr(self, "mlp_enc_l2_stream", torch.nn.Linear(eprojs, att_dim))
            setattr(self, "mlp_dec_l2_stream", torch.nn.Linear(dunits, att_dim, bias=False))
            setattr(self, "gvec_l2_stream", torch.nn.Linear(att_dim, 1))

        setattr(self, "mlp_enc_l2", torch.nn.Linear(eprojs, att_dim))
        setattr(self, "mlp_dec_l2", torch.nn.Linear(dunits, att_dim, bias=False))
        setattr(self, "gvec_l2", torch.nn.Linear(att_dim, 1))

        self.proj = torch.nn.Linear(num_enc * eprojs, eprojs)  # TODO: more projection layer

        if l2_dropout:
            self.dropout_l2 = torch.nn.Dropout2d(p=0.5)

        if apply_tanh:
            self.tanh = torch.nn.Tanh()

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length_l1 = None
        self.h_length_l2 = None
        self.h_length_l2_stream = None
        self.enc_h_l1 = None
        self.enc_h_l2 = None
        self.enc_h_l2_stream = None
        self.pre_compute_enc_h_l1 = None
        self.pre_compute_enc_h_l2 = None
        self.pre_compute_enc_h_l2_stream = None
        self.num_enc = num_enc
        self.l2_stream_weight = l2_stream_weight
        self.l1_weight = l1_weight
        self.l2_dropout = l2_dropout
        self.apply_tanh = apply_tanh

    def reset(self):
        '''reset states'''
        self.h_length_l1 = None
        self.h_length_l2 = None
        self.h_length_l2_stream = None
        self.enc_h_l1 = None
        self.enc_h_l2 = None
        self.enc_h_l2_stream = None
        self.pre_compute_enc_h_l1 = None
        self.pre_compute_enc_h_l2 = None
        self.pre_compute_enc_h_l2_stream = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''Enc2AttAddFrmLinProj forward

        :param Variable enc_hs_pad: list of padded encoder hidden state (list(B x T_max x D_enc))
        :param list enc_h_len: list of padded encoder hidden state lenght list((B))
        :param Variable dec_z: docoder hidden state (B x D_dec)
        :param Variable att_prev: list of previous attetion weight list(B x T_max)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: Variable
        :return: list of attentioin weights [list(B x T_max), list(B x T_max), B x T_max] --> l1 and l2_stream and l2
        :rtype: Variable
        '''

        batch = len(enc_hs_pad[0])
        # logging.warning(batch)
        assert enc_hs_pad[0].size()[1] == enc_hs_pad[1].size()[1]  # enc1_t_max == enc2_t_max
        T_max = enc_hs_pad[1].size()[1]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist if specified
        if att_prev is None:

            # l1
            if self.l1_weight is None:
                att_prev_l1 = None
            else:
                att_prev_l1 = [enc_hs_pad[0].new(batch, T_max).fill_(self.l1_weight) for _ in range(self.num_enc)]
            # l2_stream
            if self.l2_stream_weight is None:
                att_prev_l2_stream = None
            else:
                att_prev_l2_stream = [enc_hs_pad[0].new(batch, T_max).fill_(self.l2_stream_weight) for _ in
                                      range(self.num_enc)]
            # l2
            att_prev_l2 = None
            att_prev = [att_prev_l1, att_prev_l2_stream, att_prev_l2]

        # level 1 attention (enc_hs_pad, dec_z --> att)
        if self.l1_weight is None:
            # pre-compute all h outside the decoder loop
            if self.pre_compute_enc_h_l1 is None:
                self.enc_h_l1 = enc_hs_pad  # list (utt x frame x hdim)
                self.h_length_l1 = [self.enc_h_l1[idx].size(1) for idx in range(self.num_enc)]  # T_max for each encoder
                # utt x frame x att_dim
                self.pre_compute_enc_h_l1 = [getattr(self, "mlp_enc%d_l1" % idx)(self.enc_h_l1[idx]) for idx
                                             in range(self.num_enc)]

            # dec_z_tiled: utt x frame x att_dim
            dec_z_tiled_l1 = [getattr(self, "mlp_dec%d_l1" % idx)(dec_z).view(batch, 1, self.att_dim) for idx in
                              range(self.num_enc)]

            # dot with gvec
            # list(utt x frame x att_dim) -> list(utt x frame)
            # NOTE consider zero padding when compute w.
            e_l1 = [getattr(self, "gvec%d_l1" % idx)(torch.tanh(
                self.pre_compute_enc_h_l1[idx] + dec_z_tiled_l1[idx])).squeeze(2) for idx in range(self.num_enc)]
            w_l1 = [F.softmax(scaling * e_l1[idx], dim=1) for idx in range(self.num_enc)]
        else:
            # fix w_l1: list(utt x frame)
            w_l1 = att_prev[0]

        # level 2 stream attention (enc_hs_pad, dec_z --> att)
        if self.l2_stream_weight is None:
            # pre-compute all h outside the decoder loop
            if self.pre_compute_enc_h_l2_stream is None:
                self.enc_h_l2_stream = enc_hs_pad  # list (utt x frame x hdim)
                self.h_length_l2_stream = [self.enc_h_l2_stream[idx].size(1) for idx in
                                           range(self.num_enc)]  # T_max for each encoder
                # utt x frame x att_dim
                self.pre_compute_enc_h_l2_stream = [
                    getattr(self, "mlp_enc_l2_stream")(self.enc_h_l2_stream[idx]) for idx in
                    range(self.num_enc)]

            # dec_z_tiled: utt x frame x att_dim
            dec_z_tiled_l2_stream = [getattr(self, "mlp_dec_l2_stream")(dec_z).view(batch, 1, self.att_dim) for _ in
                                     range(self.num_enc)]

            # dot with gvec
            # list(utt x frame x att_dim) -> list(utt x frame)
            # NOTE consider zero padding when compute w.
            e_l2_stream = [getattr(self, "gvec_l2_stream")(torch.tanh(
                self.pre_compute_enc_h_l2_stream[idx] + dec_z_tiled_l2_stream[idx])).squeeze(2) for idx in
                           range(self.num_enc)]

            # list(uttxframe) -> utt x numEnc x frame # TODO: organize it after debug
            # w_l2_stream = torch.stack(e_l2_stream, dim=1) # list(uttxframe) -> utt x numEnc x frame
            # w_l2_stream = F.softmax(scaling * w_l2_stream, dim=1)
            # w_l2_stream = [w_l2_stream[:,idx,:] for idx in range(self.num_enc)]
            w_l2_stream = [F.softmax(scaling * torch.stack(e_l2_stream, dim=1), dim=1)[:, idx, :] for idx in
                           range(self.num_enc)]

        else:
            # fix w_l2: list(utt x frame)
            w_l2_stream = att_prev[1]

        # ht = Tanh(W*concat(bt1*at1*ht1,bt2*at2*ht2))
        # enc_hs_pad: list(utt x frame x hdim)
        # w_l2_stream: list(utt x frame)
        # w_l1: list(utt x frame)
        self.enc_h_proj = [w_l1[idx].unsqueeze(2) * w_l2_stream[idx].unsqueeze(2) * enc_hs_pad[idx] for idx in
                           range(self.num_enc)]
        self.enc_h_proj = torch.stack(self.enc_h_proj, dim=1)  # utt x num_enc X frame x hdim
        if self.l2_dropout: self.enc_h_proj = self.dropout_l2(self.enc_h_proj)  # TODO: (0,0) situation
        self.enc_h_proj = self.enc_h_proj.transpose(1, 2).contiguous().view(batch, T_max,
                                                                            -1)  # utt x frame x (hdim*num_enc)
        self.enc_h_proj = self.proj(self.enc_h_proj)
        if self.apply_tanh: self.enc_h_proj = self.tanh(self.enc_h_proj)

        # level 2 attention (enc_hs_pad, dec_z --> att)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h_l2 is None:
            # utt x frame x att_dim
            self.pre_compute_enc_h_l2 = getattr(self, "mlp_enc_l2")(self.enc_h_proj)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled_l2 = getattr(self, "mlp_dec_l2")(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim) -> utt x frame
        # NOTE consider zero padding when compute w.
        e_l2 = getattr(self, "gvec_l2")(torch.tanh(
            self.pre_compute_enc_h_l2 + dec_z_tiled_l2)).squeeze(2)
        w_l2 = F.softmax(scaling * e_l2, dim=1)
        # c_l2 = torch.sum(self.enc_h_proj * w_l2.view(batch, T_max, 1), dim=1)
        c_l2 = torch.matmul(w_l2.unsqueeze(1), self.enc_h_proj).squeeze(1)

        return c_l2, [w_l1, w_l2_stream, w_l2]


class AttCov(torch.nn.Module):
    '''Coverage mechanism attention

    Reference: Get To The Point: Summarization with Pointer-Generator Network
       (https://arxiv.org/abs/1704.04368)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    '''

    def __init__(self, eprojs, dunits, att_dim):
        super(AttCov, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.wvec = torch.nn.Linear(1, att_dim)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_list, scaling=2.0):
        '''AttCov forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param list att_prev_list: list of previous attetion weight
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weights
        :rtype: list
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev_list is None:
            att_prev = [enc_hs_pad.new(l).fill_(1.0 / l) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev_list = [pad_list(att_prev, 0)]

        # att_prev_list: L' * [B x T] => cov_vec B x T
        cov_vec = sum(att_prev_list)
        # cov_vec: B x T => B x T x 1 => B x T x att_dim
        cov_vec = self.wvec(cov_vec.unsqueeze(-1))

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(cov_vec + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)
        att_prev_list += [w]

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, att_prev_list


class AttLoc2D(torch.nn.Module):
    '''2D location-aware attention

    This attention is an extended version of location aware attention.
    It take not only one frame before attention weights, but also earlier frames into account.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param int att_win: attention window size (default=5)
    '''

    def __init__(self, eprojs, dunits, att_dim, att_win, aconv_chans, aconv_filts):
        super(AttLoc2D, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (att_win, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.aconv_chans = aconv_chans
        self.att_win = att_win
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttLoc2D forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: previous attetion weight (B x att_win x T_max)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attentioin weights (B x att_win x T_max)
        :rtype: torch.Tensor
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            # B * [Li x att_win]
            att_prev = [enc_hs_pad.new(l, self.att_win).fill_(1.0 / l) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev = pad_list(att_prev, 0).transpose(1, 2)

        # att_prev: B x att_win x Tmax -> B x 1 x att_win x Tmax -> B x C x 1 x Tmax
        att_conv = self.loc_conv(att_prev.unsqueeze(1))
        # att_conv: B x C x 1 x Tmax -> B x Tmax x C
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        # update att_prev: B x att_win x Tmax -> B x att_win+1 x Tmax -> B x att_win x Tmax
        att_prev = torch.cat([att_prev, w.unsqueeze(1)], dim=1)
        att_prev = att_prev[:, 1:]

        return c, att_prev


class AttLocRec(torch.nn.Module):
    '''location-aware recurrent attention

    This attention is an extended version of location aware attention.
    With the use of RNN, it take the effect of the history of attention weights into account.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttLocRec, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.att_lstm = torch.nn.LSTMCell(aconv_chans, att_dim, bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_states, scaling=2.0):
        '''AttLocRec forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param tuple att_prev_states: previous attetion weight and lstm states
                                      ((B, T_max), ((B, att_dim), (B, att_dim)))
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: previous attention weights and lstm states (w, (hx, cx))
                 ((B, T_max), ((B, att_dim), (B, att_dim)))
        :rtype: tuple
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev_states is None:
            # initialize attention weight with uniform dist.
            att_prev = [enc_hs_pad.new(l).fill_(1.0 / l) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev = pad_list(att_prev, 0)

            # initialize lstm states
            att_h = enc_hs_pad.new_zeros(batch, self.att_dim)
            att_c = enc_hs_pad.new_zeros(batch, self.att_dim)
            att_states = (att_h, att_c)
        else:
            att_prev = att_prev_states[0]
            att_states = att_prev_states[1]

        # B x 1 x 1 x T -> B x C x 1 x T
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # apply non-linear
        att_conv = F.relu(att_conv)
        # B x C x 1 x T -> B x C x 1 x 1 -> B x C
        att_conv = F.max_pool2d(att_conv, (1, att_conv.size(3))).view(batch, -1)

        att_h, att_c = self.att_lstm(att_conv, att_states)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_h.unsqueeze(1) + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, (w, (att_h, att_c))


class AttCovLoc(torch.nn.Module):
    '''Coverage mechanism location aware attention

    This attention is a combination of coverage and location-aware attentions.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int att_dim: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, att_dim, aconv_chans, aconv_filts):
        super(AttCovLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(eprojs, att_dim)
        self.mlp_dec = torch.nn.Linear(dunits, att_dim, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_dim, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1)

        self.dunits = dunits
        self.eprojs = eprojs
        self.att_dim = att_dim
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.aconv_chans = aconv_chans
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev_list, scaling=2.0):
        '''AttCovLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: docoder hidden state (B x D_dec)
        :param list att_prev_list: list of previous attetion weight
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weights
        :rtype: list
        '''

        batch = len(enc_hs_pad)
        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        # initialize attention weight with uniform dist.
        if att_prev_list is None:
            att_prev = [enc_hs_pad.new(l).fill_(1.0 / l) for l in enc_hs_len]
            # if no bias, 0 0-pad goes 0
            att_prev_list = [pad_list(att_prev, 0)]

        # att_prev_list: L' * [B x T] => cov_vec B x T
        cov_vec = sum(att_prev_list)

        # cov_vec: B x T -> B x 1 x 1 x T -> B x C x 1 x T
        att_conv = self.loc_conv(cov_vec.view(batch, 1, 1, self.h_length))
        # att_conv: utt x att_conv_chans x 1 x frame -> utt x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: utt x frame x att_conv_chans -> utt x frame x att_dim
        att_conv = self.mlp_att(att_conv)

        # dec_z_tiled: utt x frame x att_dim
        dec_z_tiled = self.mlp_dec(dec_z).view(batch, 1, self.att_dim)

        # dot with gvec
        # utt x frame x att_dim -> utt x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + dec_z_tiled)).squeeze(2)

        # NOTE consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
        e.masked_fill_(self.mask, -float('inf'))
        w = F.softmax(scaling * e, dim=1)
        att_prev_list += [w]

        # weighted sum over flames
        # utt x hdim
        # NOTE use bmm instead of sum(*)
        c = torch.sum(self.enc_h * w.view(batch, self.h_length, 1), dim=1)

        return c, att_prev_list


class AttMultiHeadDot(torch.nn.Module):
    '''Multi head dot product attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int ahead: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    '''

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v):
        super(AttMultiHeadDot, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        for h in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''AttMultiHeadDot forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weight (B x T_max) * aheads
        :rtype: list
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                torch.tanh(self.mlp_k[h](self.enc_h)) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            e = torch.sum(self.pre_compute_k[h] * torch.tanh(self.mlp_q[h](dec_z)).view(
                batch, 1, self.att_dim_k), dim=2)  # utt x frame

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadAdd(torch.nn.Module):
    '''Multi head additive attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using additive attention for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int ahead: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    '''

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v):
        super(AttMultiHeadAdd, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        for h in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''AttMultiHeadAdd forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: dummy (does not use)
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B, D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weight (B x T_max) * aheads
        :rtype: list
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            e = self.gvec[h](torch.tanh(
                self.pre_compute_k[h] + self.mlp_q[h](dec_z).view(batch, 1, self.att_dim_k))).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadLoc(torch.nn.Module):
    '''Multi head location based attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using location-aware attention for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v, aconv_chans, aconv_filts):
        super(AttMultiHeadLoc, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        self.loc_conv = torch.nn.ModuleList()
        self.mlp_att = torch.nn.ModuleList()
        for h in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
            self.loc_conv += [torch.nn.Conv2d(
                1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)]
            self.mlp_att += [torch.nn.Linear(aconv_chans, att_dim_k, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev, scaling=2.0):
        '''AttMultiHeadLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: list of previous attentioin weight (B x T_max) * aheads
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weight (B x T_max) * aheads
        :rtype: list
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            att_prev = []
            for h in six.moves.range(self.aheads):
                att_prev += [[enc_hs_pad.new(l).fill_(1.0 / l) for l in enc_hs_len]]
                # if no bias, 0 0-pad goes 0
                att_prev[h] = pad_list(att_prev[h], 0)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            att_conv = self.loc_conv[h](att_prev[h].view(batch, 1, 1, self.h_length))
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            att_conv = self.mlp_att[h](att_conv)

            e = self.gvec[h](torch.tanh(
                self.pre_compute_k[h] + att_conv + self.mlp_q[h](dec_z).view(
                    batch, 1, self.att_dim_k))).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


class AttMultiHeadMultiResLoc(torch.nn.Module):
    '''Multi head multi resolution location based attention

    Reference: Attention is all you need
        (https://arxiv.org/abs/1706.03762)

    This attention is multi head attention using location-aware attention for each head.
    Furthermore, it uses different filter size for each head.

    :param int eprojs: # projection-units of encoder
    :param int dunits: # units of decoder
    :param int aheads: # heads of multi head attention
    :param int att_dim_k: dimension k in multi head attention
    :param int att_dim_v: dimension v in multi head attention
    :param int aconv_chans: maximum # channels of attention convolution
        each head use #ch = aconv_chans * (head + 1) / aheads
        e.g. aheads=4, aconv_chans=100 => filter size = 25, 50, 75, 100
    :param int aconv_filts: filter size of attention convolution
    '''

    def __init__(self, eprojs, dunits, aheads, att_dim_k, att_dim_v, aconv_chans, aconv_filts):
        super(AttMultiHeadMultiResLoc, self).__init__()
        self.mlp_q = torch.nn.ModuleList()
        self.mlp_k = torch.nn.ModuleList()
        self.mlp_v = torch.nn.ModuleList()
        self.gvec = torch.nn.ModuleList()
        self.loc_conv = torch.nn.ModuleList()
        self.mlp_att = torch.nn.ModuleList()
        for h in six.moves.range(aheads):
            self.mlp_q += [torch.nn.Linear(dunits, att_dim_k)]
            self.mlp_k += [torch.nn.Linear(eprojs, att_dim_k, bias=False)]
            self.mlp_v += [torch.nn.Linear(eprojs, att_dim_v, bias=False)]
            self.gvec += [torch.nn.Linear(att_dim_k, 1)]
            afilts = aconv_filts * (h + 1) // aheads
            self.loc_conv += [torch.nn.Conv2d(
                1, aconv_chans, (1, 2 * afilts + 1), padding=(0, afilts), bias=False)]
            self.mlp_att += [torch.nn.Linear(aconv_chans, att_dim_k, bias=False)]
        self.mlp_o = torch.nn.Linear(aheads * att_dim_v, eprojs, bias=False)
        self.dunits = dunits
        self.eprojs = eprojs
        self.aheads = aheads
        self.att_dim_k = att_dim_k
        self.att_dim_v = att_dim_v
        self.scaling = 1.0 / math.sqrt(att_dim_k)
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def reset(self):
        '''reset states'''
        self.h_length = None
        self.enc_h = None
        self.pre_compute_k = None
        self.pre_compute_v = None
        self.mask = None

    def forward(self, enc_hs_pad, enc_hs_len, dec_z, att_prev):
        '''AttMultiHeadMultiResLoc forward

        :param torch.Tensor enc_hs_pad: padded encoder hidden state (B x T_max x D_enc)
        :param list enc_h_len: padded encoder hidden state lenght (B)
        :param torch.Tensor dec_z: decoder hidden state (B x D_dec)
        :param torch.Tensor att_prev: list of previous attentioin weight (B x T_max) * aheads
        :param float scaling: scaling parameter before applying softmax
        :return: attentioin weighted encoder state (B x D_enc)
        :rtype: torch.Tensor
        :return: list of previous attentioin weight (B x T_max) * aheads
        :rtype: list
        '''

        batch = enc_hs_pad.size(0)
        # pre-compute all k and v outside the decoder loop
        if self.pre_compute_k is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_k = [
                self.mlp_k[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if self.pre_compute_v is None:
            self.enc_h = enc_hs_pad  # utt x frame x hdim
            self.h_length = self.enc_h.size(1)
            # utt x frame x att_dim
            self.pre_compute_v = [
                self.mlp_v[h](self.enc_h) for h in six.moves.range(self.aheads)]

        if dec_z is None:
            dec_z = enc_hs_pad.new_zeros(batch, self.dunits)
        else:
            dec_z = dec_z.view(batch, self.dunits)

        if att_prev is None:
            att_prev = []
            for h in six.moves.range(self.aheads):
                att_prev += [[enc_hs_pad.new(l).fill_(1.0 / l) for l in enc_hs_len]]
                # if no bias, 0 0-pad goes 0
                att_prev[h] = pad_list(att_prev[h], 0)

        c = []
        w = []
        for h in six.moves.range(self.aheads):
            att_conv = self.loc_conv[h](att_prev[h].view(batch, 1, 1, self.h_length))
            att_conv = att_conv.squeeze(2).transpose(1, 2)
            att_conv = self.mlp_att[h](att_conv)

            e = self.gvec[h](torch.tanh(
                self.pre_compute_k[h] + att_conv + self.mlp_q[h](dec_z).view(
                    batch, 1, self.att_dim_k))).squeeze(2)

            # NOTE consider zero padding when compute w.
            if self.mask is None:
                self.mask = to_cuda(self, make_pad_mask(enc_hs_len))
            e.masked_fill_(self.mask, -float('inf'))
            w += [F.softmax(self.scaling * e, dim=1)]

            # weighted sum over flames
            # utt x hdim
            # NOTE use bmm instead of sum(*)
            c += [torch.sum(self.pre_compute_v[h] * w[h].view(batch, self.h_length, 1), dim=1)]

        # concat all of c
        c = self.mlp_o(torch.cat(c, dim=1))

        return c, w


# ------------- Decoder Network ----------------------------------------------------------------------------------------
class Decoder(torch.nn.Module):
    """Decoder module

    :param int eprojs: # encoder projection units
    :param int odim: dimension of outputs
    :param int dlayers: # decoder layers
    :param int dunits: # decoder units
    :param int sos: start of sequence symbol id
    :param int eos: end of sequence symbol id
    :param torch.nn.Module att: attention module
    :param int verbose: verbose level
    :param list char_list: list of character strings
    :param ndarray labeldist: distribution of label smoothing
    :param float lsm_weight: label smoothing weight
    """

    def __init__(self, eprojs, odim, dlayers, dunits, sos, eos, att, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0., num_enc=1):
        super(Decoder, self).__init__()
        self.dunits = dunits
        self.dlayers = dlayers
        self.embed = torch.nn.Embedding(odim, dunits)
        self.decoder = torch.nn.ModuleList()
        self.decoder += [torch.nn.LSTMCell(dunits + eprojs, dunits)]
        for l in six.moves.range(1, self.dlayers):
            self.decoder += [torch.nn.LSTMCell(dunits, dunits)]
        self.ignore_id = -1
        self.output = torch.nn.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.num_enc = num_enc

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def forward(self, hs_pad, hlens, ys_pad):
        '''Decoder forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        '''
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

        if self.num_enc == 1:
            hlens = list(map(int, hlens))

            # initialization
            c_list = [self.zero_state(hs_pad)]
            z_list = [self.zero_state(hs_pad)]
            for l in six.moves.range(1, self.dlayers):
                c_list.append(self.zero_state(hs_pad))
                z_list.append(self.zero_state(hs_pad))

        else:
            hlens = [list(map(int, hlens[idx])) for idx in range(self.num_enc)]

            # initialization
            c_list = [self.zero_state(hs_pad[0])]
            z_list = [self.zero_state(hs_pad[0])]
            for l in six.moves.range(1, self.dlayers):
                c_list.append(self.zero_state(hs_pad[0]))
                z_list.append(self.zero_state(hs_pad[0]))

        att_w = None
        z_all = []
        self.att.reset()  # reset pre-computation of h

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get dim, length info
        batch = ys_out_pad.size(0)
        olength = ys_out_pad.size(1)
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(hlens))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str([y.size(0) for y in ys_out]))

        # pre-computation of embedding
        eys = self.embed(ys_in_pad)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs_pad, hlens, z_list[0], att_w)
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            z_all.append(z_list[-1])

        z_all = torch.stack(z_all, dim=1).view(batch * olength, self.dunits)
        # compute loss
        y_all = self.output(z_all)
        self.loss = F.cross_entropy(y_all, ys_out_pad.view(-1),
                                    ignore_index=self.ignore_id,
                                    size_average=True)
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        acc = th_accuracy(y_all, ys_out_pad, ignore_label=self.ignore_id)
        logging.info('att loss:' + ''.join(str(self.loss.item()).split('\n')))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            ys_hat = y_all.view(batch, olength, -1)
            ys_true = ys_out_pad
            for (i, y_hat), y_true in zip(enumerate(ys_hat.detach().cpu().numpy()),
                                          ys_true.detach().cpu().numpy()):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = np.argmax(y_hat[y_true != self.ignore_id], axis=1)
                idx_true = y_true[y_true != self.ignore_id]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = to_cuda(self, torch.from_numpy(self.labeldist))
            loss_reg = - torch.sum((F.log_softmax(y_all, dim=1) * self.vlabeldist).view(-1), dim=0) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc

    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None):
        '''beam search implementation

        :param torch.Tensor h: encoder hidden state (1, T, eprojs)
        :param torch.Tensor lpz: ctc log softmax output (T, odim)
        :param Namespace recog_args: argument namespace contraining options
        :param char_list: list of character strings
        :param torch.nn.Module rnnlm: language module
        :return: N-best decoding results
        :rtype: list of dicts
        '''

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # in multi stream case, h is a tuple [utt(1) x frame x hdim], lpz is also a tuple [utt(1) x frame x odim]
        # in one stream case, h is utt(1) x frame x hdim, lpz is utt(1) x frame x odim

        if self.num_enc == 1:
            logging.info('input lengths: ' + str(h.size(1)))

            # initialization
            c_list = [self.zero_state(h)]
            z_list = [self.zero_state(h)]
            for l in six.moves.range(1, self.dlayers):
                c_list.append(self.zero_state(h))
                z_list.append(self.zero_state(h))
            a = None
            self.att.reset()  # reset pre-computation of h

            # preprate sos
            y = self.sos
            vy = h.new_zeros(1).long()

            if recog_args.maxlenratio == 0:
                maxlen = h.shape[1]
            else:
                # maxlen >= 1
                maxlen = max(1, int(recog_args.maxlenratio * h.size(1)))
            minlen = int(recog_args.minlenratio * h.size(0))
            logging.info('max output length: ' + str(maxlen))
            logging.info('min output length: ' + str(minlen))

            # initialize hypothesis
            if rnnlm:
                hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list,
                       'z_prev': z_list, 'a_prev': a, 'rnnlm_prev': None}
            else:
                hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}
            if lpz is not None:
                ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, np)
                hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
                hyp['ctc_score_prev'] = 0.0
                if ctc_weight != 1.0:
                    # pre-pruning based on attention scores
                    ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
                else:
                    ctc_beam = lpz.shape[-1]
            hyps = [hyp]
            ended_hyps = []
            init_hyp = hyp.copy()
        else:
            logging.info('input lengths: ' + str([h[idx].size(0) for idx in range(self.num_enc)]))

            # initialization
            c_list = [self.zero_state(h[0])]
            z_list = [self.zero_state(h[0])]
            for l in six.moves.range(1, self.dlayers):
                c_list.append(self.zero_state(h[0]))
                z_list.append(self.zero_state(h[0]))
            a = None
            self.att.reset()  # reset pre-computation of h

            # preprate sos
            y = self.sos
            vy = h[0].new_zeros(1).long()

            max_frame = np.amin([hh.shape[1] for hh in h])
            if recog_args.maxlenratio == 0:
                maxlen = max_frame
            else:
                # maxlen >= 1
                maxlen = max(1, int(recog_args.maxlenratio * max_frame))
            minlen = int(recog_args.minlenratio * max_frame)
            logging.info('max output length: ' + str(maxlen))
            logging.info('min output length: ' + str(minlen))

            # initialize hypothesis
            if rnnlm:
                hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list,
                       'z_prev': z_list, 'a_prev': a, 'rnnlm_prev': None}
            else:
                hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}

            if lpz is not None:
                ctc_prefix_score = [CTCPrefixScore(lpz[idx].detach().numpy(), 0, self.eos, np) for idx in
                                    range(self.num_enc)]
                hyp['ctc_state_prev'] = [ctc_prefix_score[idx].initial_state() for idx in range(self.num_enc)]
                hyp['ctc_score_prev'] = [0.0 for idx in range(self.num_enc)]
                if ctc_weight != 1.0:
                    # pre-pruning based on attention scores
                    ctc_beam = min(lpz[0].shape[-1], int(beam * CTC_SCORING_RATIO))
                else:
                    ctc_beam = lpz[0].shape[-1]
            hyps = [hyp]
            ended_hyps = []
            init_hyp = hyp.copy()

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                # vy.unsqueeze(1) # not inplace function
                vy[0] = hyp['yseq'][i]
                ey = self.embed(vy)  # utt list (1) x zdim
                # ey.unsqueeze(0) # not inplace function
                # in multiencoder case, att_w = [att1, att2, att3 (stream attention)]

                if self.num_enc == 1:
                    att_c, att_w = self.att(h, [h.size(1)], hyp['z_prev'][0], hyp['a_prev'])
                else:
                    att_c, att_w = self.att(h, [[hh.shape[1]] for hh in h], hyp['z_prev'][0], hyp['a_prev'])

                ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
                z_list[0], c_list[0] = self.decoder[0](ey, (hyp['z_prev'][0], hyp['c_prev'][0]))
                for l in six.moves.range(1, self.dlayers):
                    z_list[l], c_list[l] = self.decoder[l](
                        z_list[l - 1], (hyp['z_prev'][l], hyp['c_prev'][l]))

                # get nbest local scores and their ids
                local_att_scores = F.log_softmax(self.output(z_list[-1]), dim=1)
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    if self.num_enc == 1:
                        ctc_scores, ctc_states = ctc_prefix_score(
                            hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                        local_scores = \
                            (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                            + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    else:
                        ctc_scores, ctc_states = zip(*[ctc_prefix_score[idx](
                            hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'][idx]) for idx in range(
                            self.num_enc)])  # ctc_scores=[score_stream1, score_stream2, ..., score_streamN], ctc_states = [state1, state2, ..., stateN] where stateN is beam x frame
                        local_scores = \
                            (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                            + ctc_weight * torch.from_numpy(
                                np.mean(
                                    np.stack(
                                        [ctc_scores[idx] - hyp['ctc_score_prev'][idx] for idx in range(self.num_enc)]),
                                    axis=0))
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    # [:] is needed!
                    new_hyp['z_prev'] = z_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = att_w[:]
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        if self.num_enc == 1:
                            new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                            new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                        else:
                            new_hyp['ctc_state_prev'] = [ctc_states[idx][joint_best_ids[0, j]] for idx in
                                                         range(self.num_enc)]
                            new_hyp['ctc_score_prev'] = [ctc_scores[idx][joint_best_ids[0, j]] for idx in
                                                         range(self.num_enc)]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            logging.debug(
                'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                logging.debug(
                    'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warn('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize_beam(h, lpz, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        # remove sos
        return nbest_hyps

    def calculate_all_attentions(self, hs_pad, hlen, ys_pad):
        '''Calculate all of attentions

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        '''
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        if self.num_enc == 1:
            # hlen should be list of integer
            hlen = list(map(int, hlen))

            # initialization
            c_list = [self.zero_state(hs_pad)]
            z_list = [self.zero_state(hs_pad)]
            for l in six.moves.range(1, self.dlayers):
                c_list.append(self.zero_state(hs_pad))
                z_list.append(self.zero_state(hs_pad))
        else:
            # hlen should be list of list of integer
            hlen = [list(map(int, hlen[idx])) for idx in range(self.num_enc)]

            # initialization
            c_list = [self.zero_state(hs_pad[0])]
            z_list = [self.zero_state(hs_pad[0])]
            for l in six.moves.range(1, self.dlayers):
                c_list.append(self.zero_state(hs_pad[0]))
                z_list.append(self.zero_state(hs_pad[0]))

        att_w = None
        att_ws = []
        self.att.reset()  # reset pre-computation of h

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get length info
        olength = ys_out_pad.size(1)

        # pre-computation of embedding
        eys = self.embed(ys_in_pad)  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att(hs_pad, hlen, z_list[0], att_w)
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list[0], c_list[0] = self.decoder[0](ey, (z_list[0], c_list[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    z_list[l - 1], (z_list[l], c_list[l]))
            att_ws.append(att_w)

        # convert to numpy array with the shape (B, Lmax, Tmax)
        if isinstance(self.att, AttLoc2D):
            # att_ws => list of previous concate attentions
            att_ws = torch.stack([aw[:, -1] for aw in att_ws], dim=1).cpu().numpy()
        elif isinstance(self.att, (AttCov, AttCovLoc)):
            # att_ws => list of list of previous attentions
            att_ws = torch.stack([aw[-1] for aw in att_ws], dim=1).cpu().numpy()
        elif isinstance(self.att, AttLocRec):
            # att_ws => list of tuple of attention and hidden states
            att_ws = torch.stack([aw[0] for aw in att_ws], dim=1).cpu().numpy()
        elif isinstance(self.att, (AttMultiHeadDot, AttMultiHeadAdd, AttMultiHeadLoc, AttMultiHeadMultiResLoc)):
            # att_ws => list of list of each head attetion
            n_heads = len(att_ws[0])
            att_ws_sorted_by_head = []
            for h in six.moves.range(n_heads):
                att_ws_head = torch.stack([aw[h] for aw in att_ws], dim=1)
                att_ws_sorted_by_head += [att_ws_head]
            att_ws = torch.stack(att_ws_sorted_by_head, dim=1).cpu().numpy()
        # elif isinstance(self.att, (Enc2AttAdd, Enc2AttLoc)):
        elif isinstance(self.att, (Enc2AttAdd, Enc2AttAddLinProj)):
            # att_ws => list(len(odim)) of [[att1_l1, att2_l1, ...,attN_l1], att_l2]; N: numstreams
            # list of [utt x (NumEnc+1) x T_max]
            att_ws = [pad_list([a.transpose(0, 1) for a in aw[0] + [aw[1]]], 0).transpose(0, 2).transpose(1, 2) for aw
                      in att_ws]
            # att_ws => utt x (NumEnc+1) x odim x T_max; T_max is the max length among all attention sets for all utterance
            att_ws = torch.stack(att_ws, dim=2).cpu().numpy()
        elif isinstance(self.att, Enc2AttAddFrmLinProj):
            # att_ws => list(len(odim)) of [[att1_l1, att2_l1, ...,attN_l1], [att1_l2_stream, att2_l2_stream, ...,attN_l2_stream], att_l2]; N: numstreams
            # list of [utt x (2*NumEnc+1) x T_max]
            att_ws = [pad_list([a.transpose(0, 1) for a in aw[0] + aw[1] + [aw[2]]], 0).transpose(0, 2).transpose(1, 2)
                      for aw in att_ws]
            # att_ws => utt x (2*NumEnc+1) x odim x T_max; T_max is the max length among all attention sets for all utterance
            att_ws = torch.stack(att_ws, dim=2).cpu().numpy()
        else:
            # att_ws => list of attetions
            att_ws = torch.stack(att_ws, dim=1).cpu().numpy()
        return att_ws


# ------------- Encoder Network ----------------------------------------------------------------------------------------
class Encoder(torch.nn.Module):
    '''Encoder module

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int epojs: number of projection units of encoder network
    :param list subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
    '''

    def __init__(self, etype, idim, elayers, eunits, eprojs, subsample, dropout, in_channel=1, addgauss=False, addgauss_mean=0, addgauss_std=1, addgauss_type='all'):
        super(Encoder, self).__init__()

        # blstm
        if etype == 'blstm':
            self.enc1 = BLSTM(idim, elayers, eunits, eprojs, dropout)
            logging.info('BLSTM without projection for encoder')
        elif etype == 'blstmp':
            self.enc1 = BLSTMP(idim, elayers, eunits,
                               eprojs, subsample, dropout)
            logging.info('BLSTM with every-layer projection for encoder')
        elif etype == 'blstmss':
            self.enc1 = BLSTMSS(idim, elayers, eunits,
                                eprojs, subsample, dropout)
            logging.info('BLSTM with subsampling without every-layer projection for encoder')
        elif etype == 'blstmpbn':
            self.enc1 = BLSTMPBN(idim, elayers, eunits,
                                 eprojs, subsample, dropout)
            logging.info('BLSTM with every-layer projection for encoder with batchnorm')

        # vgg
        elif etype == 'vgg':
            self.enc1 = VGGOnly(idim=idim, eprojs=eprojs, in_channels=in_channel, )
            logging.info('Use CNN-VGG for encoder')
        elif etype == 'vggblstm':
            self.enc1 = VGG(in_channels=in_channel)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-VGG + BLSTM for encoder')
        elif etype == 'vgg8blstm':
            self.enc1 = VGG8(in_channels=in_channel)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel, mode='vgg8', out_channel=256),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-VGG8 + BLSTM for encoder')
        elif etype == 'vggblstmp':
            self.enc1 = VGG(in_channels=in_channel)
            self.enc2 = BLSTMP(get_maxpooling2_odim(idim, in_channel=in_channel),
                               elayers, eunits, eprojs,
                               subsample, dropout)
            logging.info('Use CNN-VGG + BLSTMP for encoder')
        elif etype == 'vggbnblstm':
            self.enc1 = VGG(batch_norm=True, in_channels=in_channel)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-VGG-BN + BLSTM for encoder')
        elif etype == 'vggbnblstmp':
            self.enc1 = VGG(batch_norm=True, in_channels=in_channel)
            self.enc2 = BLSTMP(get_maxpooling2_odim(idim, in_channel=in_channel),
                               elayers, eunits, eprojs,
                               subsample, dropout)
            logging.info('Use CNN-VGG-BN + BLSTMP for encoder')
        elif etype == 'vggsdpblstm':
            self.enc1 = VGG(spatial_dp=True, in_channels=in_channel)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-VGG-SpatialDP + BLSTM for encoder')
        elif etype == 'vggsdpblstmp':
            self.enc1 = VGG(spatial_dp=True, in_channels=in_channel)
            self.enc2 = BLSTMP(get_maxpooling2_odim(idim, in_channel=in_channel),
                               elayers, eunits, eprojs,
                               subsample, dropout)
            logging.info('Use CNN-VGG-SpatialDP + BLSTMP for encoder')
        elif etype == 'vggnbblstm':
            self.enc1 = VGG(bias=False, in_channels=in_channel)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-VGG-nobias + BLSTM for encoder')
        elif etype == 'vggnbblstmp':
            self.enc1 = VGG(bias=False, in_channels=in_channel)
            self.enc2 = BLSTMP(get_maxpooling2_odim(idim, in_channel=in_channel),
                               elayers, eunits, eprojs,
                               subsample, dropout)
            logging.info('Use CNN-VGG-nobias + BLSTMP for encoder')
        elif etype == 'vggceilblstm':
            self.enc1 = VGG(ceil_mode=True, in_channels=in_channel)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel, ceil_mode=True),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-VGG-ceil + BLSTM for encoder')
        elif etype == 'vggceilblstmp':
            self.enc1 = VGG(ceil_mode=True, in_channels=in_channel)
            self.enc2 = BLSTMP(get_maxpooling2_odim(idim, in_channel=in_channel, ceil_mode=True),
                               elayers, eunits, eprojs,
                               subsample, dropout)
            logging.info('Use CNN-VGG-ceil + BLSTMP for encoder')
        elif etype == 'vggsjblstm':
            self.enc1 = VGG2L(in_channel=in_channel)
            self.enc2 = BLSTM(get_vgg2l_odim(idim, in_channel=in_channel),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-VGG + BLSTM for encoder')
        elif etype == 'vggdil2blstm':
            self.enc1 = VGG(in_channels=in_channel, dilation=2)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel, dilation=2),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-VGG with dilation 2 + BLSTM for encoder')
        elif etype == 'vggresblstm':
            self.enc1 = VGGRes(in_channels=in_channel)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-VGG followed by one resnet + BLSTM for encoder')
        elif etype == 'vggresdil2blstm':
            self.enc1 = VGGRes(in_channels=in_channel, dilation=2)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel, dilation=2),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-VGG followed by one reset with dilation 2 + BLSTM for encoder')

        # resnet
        elif etype == 'resblstm':
            self.enc1 = VGG(resnet=True, in_channels=in_channel)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-Res + BLSTM for encoder')
        elif etype == 'resblstmp':
            self.enc1 = VGG(resnet=True, in_channels=in_channel)
            self.enc2 = BLSTMP(get_maxpooling2_odim(idim, in_channel=in_channel),
                               elayers, eunits, eprojs,
                               subsample, dropout)
            logging.info('Use CNN-Res + BLSTMP for encoder')
        elif etype == 'resbnblstm':
            self.enc1 = VGG(resnet=True, batch_norm=True, in_channels=in_channel)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-Res-BN + BLSTM for encoder')
        elif etype == 'resbnblstmp':
            self.enc1 = VGG(resnet=True, batch_norm=True, in_channels=in_channel)
            self.enc2 = BLSTMP(get_maxpooling2_odim(idim, in_channel=in_channel),
                               elayers, eunits, eprojs,
                               subsample, dropout)
            logging.info('Use CNN-Res-BN + BLSTMP for encoder')
        elif etype == 'ressdpblstm':
            self.enc1 = VGG(resnet=True, spatial_dp=True, in_channels=in_channel)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-Res-SpatialDP + BLSTM for encoder')
        elif etype == 'ressdpblstmp':
            self.enc1 = VGG(resnet=True, spatial_dp=True, in_channels=in_channel)
            self.enc2 = BLSTMP(get_maxpooling2_odim(idim, in_channel=in_channel),
                               elayers, eunits, eprojs,
                               subsample, dropout)
            logging.info('Use CNN-Res-SpatialDP + BLSTMP for encoder')
        elif etype == 'resnbblstm':
            self.enc1 = VGG(resnet=True, bias=False, in_channels=in_channel)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-Res-nobias + BLSTM for encoder')
        elif etype == 'resnbblstmp':
            self.enc1 = VGG(resnet=True, bias=False, in_channels=in_channel)
            self.enc2 = BLSTMP(get_maxpooling2_odim(idim, in_channel=in_channel),
                               elayers, eunits, eprojs,
                               subsample, dropout)
            logging.info('Use CNN-Res-nobias + BLSTMP for encoder')
        elif etype == 'resceilblstm':
            self.enc1 = VGG(resnet=True, ceil_mode=True, in_channels=in_channel)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel, ceil_mode=True),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-Res-ceil + BLSTM for encoder')
        elif etype == 'resceilblstmp':
            self.enc1 = VGG(resnet=True, ceil_mode=True, in_channels=in_channel, )
            self.enc2 = BLSTMP(get_maxpooling2_odim(idim, in_channel=in_channel, ceil_mode=True),
                               elayers, eunits, eprojs,
                               subsample, dropout)
            logging.info('Use CNN-Res-ceil + BLSTMP for encoder')
        elif etype == 'resorigblstm':
            self.enc1 = ResNetOrig(in_channels=in_channel)
            self.enc2 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel, mode='resnetorig'),
                              elayers, eunits, eprojs, dropout)
            logging.info('Use CNN-Res-Orig + BLSTM for encoder')


        # RCNN
        elif etype == 'rcnn':
            self.enc1 = RCNN(idim=idim, eprojs=eprojs, block=BasicBlock_BnReluConv, inplanes=in_channel, flag_bn=True,
                             flag_relu=True, flag_dp=False)
            logging.info('Use RCNN for encoder')
        elif etype == 'rcnnNObn':
            self.enc1 = RCNN(idim=idim, eprojs=eprojs, block=BasicBlock_BnReluConv, inplanes=in_channel, flag_bn=False,
                             flag_relu=True, flag_dp=False)
            logging.info('Use RCNN without BatchNorm for encoder')
        elif etype == 'rcnnDp':
            self.enc1 = RCNN(idim=idim, eprojs=eprojs, block=BasicBlock_BnReluConv, inplanes=in_channel, flag_bn=True,
                             flag_relu=True, flag_dp=True)
            logging.info('Use RCNN with dropout for encoder')
        elif etype == 'rcnnDpNObn':
            self.enc1 = RCNN(idim=idim, eprojs=eprojs, block=BasicBlock_BnReluConv, inplanes=in_channel, flag_bn=False,
                             flag_relu=True, flag_dp=True)
            logging.info('Use RCNN with dropout without BatchNorm for encoder')


        # multi-encoder, multi-band
        elif etype == 'multiVggblstmBlstmp':
            self.enc11 = VGG(in_channels=in_channel)
            self.enc12 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel),
                               elayers, eunits, eprojs, dropout)
            self.enc21 = BLSTMP(idim, elayers, eunits,
                                eprojs, subsample, dropout)
            logging.info('Multi-Encoder: VGGBLSTM, BLSTMP for encoders')
        elif etype == 'multiVggblstmpBlstmp':
            self.enc11 = VGG(in_channels=in_channel)
            self.enc12 = BLSTMP(get_maxpooling2_odim(idim, in_channel=in_channel),
                                elayers, eunits, eprojs,
                                subsample, dropout)
            self.enc21 = BLSTMP(idim, elayers, eunits,
                                eprojs, subsample, dropout)
            logging.info('Multi-Encoder: VGGBLSTMP, BLSTMP for encoders')

        elif etype == 'multiVggblstmBlstmpFixed4':
            self.enc11 = VGG(in_channels=in_channel)
            self.enc12 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel),
                               elayers, eunits, eprojs, dropout)
            self.enc21 = BLSTMP(idim, elayers=4, cdim=320,
                                hdim=320, subsample=np.array([1, 1, 1, 1, 1]).astype(int), dropout=0.0)
            logging.info('Multi-Encoder: VGGBLSTM, BLSTMP for encoders')
        elif etype == 'multiVggblstmpBlstmpFixed4':
            self.enc11 = VGG(in_channels=in_channel)
            self.enc12 = BLSTMP(get_maxpooling2_odim(idim, in_channel=in_channel),
                                elayers, eunits, eprojs,
                                subsample, dropout)
            self.enc21 = BLSTMP(idim, elayers=4, cdim=320,
                                hdim=320, subsample=np.array([1, 1, 1, 1, 1]).astype(int), dropout=0.0)
            logging.info('Multi-Encoder: VGGBLSTMP, BLSTMP for encoders')



        elif etype == 'multiVgg8blstmBlstmp':
            self.enc11 = VGG8(in_channels=in_channel)
            self.enc12 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel, mode='vgg8', out_channel=256),
                               elayers, eunits, eprojs, dropout)
            self.enc21 = BLSTMP(idim, elayers, eunits,
                                eprojs, subsample, dropout)
            logging.info('Multi-Encoder: VGG8BLSTM, BLSTMP for encoders')

        elif etype == 'multiVggdil2blstmBlstmp':
            self.enc11 = VGG(in_channels=in_channel, dilation=2)
            self.enc12 = BLSTM(get_maxpooling2_odim(idim, in_channel=in_channel, dilation=2),
                               elayers, eunits, eprojs, dropout)
            self.enc21 = BLSTMP(idim, elayers, eunits,
                                eprojs, subsample, dropout)
            logging.info('Multi-Encoder: VGGBLSTM+dilation2, BLSTMP for encoders')


        elif etype == 'multiBlstmpBlstmp4':
            subsample1 = np.array([1, 1, 1, 1, 1]).astype(int)
            subsample2 = np.array([1, 2, 2, 1, 1]).astype(int)
            self.enc1 = BLSTMP(idim, elayers, eunits,
                               eprojs, subsample1, dropout)
            self.enc2 = BLSTMP(idim, elayers, eunits,
                               eprojs, subsample2, dropout)
            logging.info('Use BLSTMP + BLSTMP(/4) for encoder')

        elif etype == 'multiBandBlstmpBlstmp':
            self.enc1 = BLSTMP(43, elayers, eunits,
                               eprojs, subsample, dropout)  # 40+3 pitch

            self.enc2 = BLSTMP(43, elayers, eunits,
                               eprojs, subsample, dropout)  # 40+3 pitch
            logging.info('Multi-Band: BLSTMP(40LF+3Pitch), BLSTMP(40HF+3Pitch) for encoders')
        elif etype == 'lowBandBlstmp':
            self.enc1 = BLSTMP(43, elayers, eunits,
                               eprojs, subsample, dropout)  # 40+3 pitch
            logging.info('Low-Band: BLSTMP(40LF+3Pitch) for encoders')
        elif etype == 'highBandBlstmp':
            self.enc1 = BLSTMP(43, elayers, eunits,
                               eprojs, subsample, dropout)  # 40+3 pitch
            logging.info('High-Band: BLSTMP(40HF+3Pitch) for encoders')

        # ami
        elif etype == 'amiCH1BlstmpCH2Blstmp':
            self.enc1 = BLSTMP(83, elayers, eunits,
                               eprojs, subsample, dropout)  # 40+3 pitch

            self.enc2 = BLSTMP(83, elayers, eunits,
                               eprojs, subsample, dropout)
            logging.info('AMI: BLSTMP(CH1), BLSTMP(CH2) for encoders')
        elif etype == 'amiCH1Blstmp':
            self.enc1 = BLSTMP(83, elayers, eunits,
                               eprojs, subsample, dropout)
            logging.info('AMI: BLSTMP(CH1) for encoders')
        elif etype == 'amiCH2Blstmp':
            self.enc1 = BLSTMP(83, elayers, eunits,
                               eprojs, subsample, dropout)
            logging.info('AMI: BLSTMP(CH2) for encoders')

        elif etype == 'amiCH1VggblstmCH2Vggblstm':
            self.enc11 = VGG(in_channels=in_channel)
            self.enc12 = BLSTM(get_maxpooling2_odim(83, in_channel=in_channel),
                               elayers, eunits, eprojs, dropout)
            self.enc21 = VGG(in_channels=in_channel)
            self.enc22 = BLSTM(get_maxpooling2_odim(83, in_channel=in_channel),
                               elayers, eunits, eprojs, dropout)
            logging.info('AMI: VGGBLSTM(CH1), VGGBLSTM(CH2) for encoders')
        elif etype == 'amiCH1Vggblstm':
            self.enc11 = VGG(in_channels=in_channel)
            self.enc12 = BLSTM(get_maxpooling2_odim(83, in_channel=in_channel),
                               elayers, eunits, eprojs, dropout)
            logging.info('AMI: VGGBLSTM(CH1) for encoders')
        elif etype == 'amiCH2Vggblstm':
            self.enc21 = VGG(in_channels=in_channel)
            self.enc22 = BLSTM(get_maxpooling2_odim(83, in_channel=in_channel),
                               elayers, eunits, eprojs, dropout)
            logging.info('AMI: VGGBLSTM(CH2) for encoders')
        else:
            logging.error(
                "Error: need to specify an appropriate encoder archtecture")
            sys.exit()

        self.etype = etype
        self.addgauss = addgauss
        self.addgauss_mean = addgauss_mean
        self.addgauss_std = addgauss_std
        self.addgauss_type = addgauss_type

    def forward(self, xs_pad, ilens):
        '''Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, erojs)
        :rtype: torch.Tensor
        '''
        if self.etype in ['blstm', 'blstmp', 'blstmss', 'blstmpbn', 'vgg', 'rcnn', 'rcnnNObn', 'rcnnDp', 'rcnnDpNObn']:

            if self.addgauss: # decoding stage #TODO hardcode the dim
                dims1 = list(range(40)) + list(range(80, 83))  # low frequency + 3 pitch
                dims2 = list(range(40, 80)) + list(range(80, 83))  # high frequency + 3 pitch
                xs_pad1 = xs_pad[:, :, dims1]
                xs_pad2 = xs_pad[:, :, dims2]
                gauss_dist = tdist.Normal(torch.tensor([self.addgauss_mean]), torch.tensor([self.addgauss_std]))
                if self.addgauss_type == 'low43':
                    gauss_noise = gauss_dist.sample(xs_pad1.size()).squeeze(len(xs_pad1.size()))
                    xs_pad1 += gauss_noise
                elif self.addgauss_type == 'high43':
                    gauss_noise = gauss_dist.sample(xs_pad2.size()).squeeze(len(xs_pad2.size()))
                    xs_pad2 += gauss_noise
                elif self.addgauss_type == 'all':
                    gauss_noise1 = gauss_dist.sample(xs_pad1.size()).squeeze(len(xs_pad1.size()))
                    gauss_noise2 = gauss_dist.sample(xs_pad2.size()).squeeze(len(xs_pad2.size()))
                    xs_pad1 += gauss_noise1
                    xs_pad2 += gauss_noise2
                else:
                    logging.error(
                        "Error: need to specify an appropriate addgauss type")
                    sys.exit()
                xs_pad = torch.cat((xs_pad1[:,:,:40],xs_pad2[:,:,:40],xs_pad1[:,:,40:]),2)

            xs_pad, ilens = self.enc1(xs_pad, ilens)

            xs_pad = fill_padded_part(xs_pad, ilens, 0.0)

        elif self.etype in ['vggblstm', 'vggblstmp', 'vggbnblstm', 'vggbnblstmp', 'vggceilblstm', 'vggceilblstmp',
                            'vggnbblstm', 'vggnbblstmp', 'vggsjblstm', 'vggresblstm', 'vggdil2blstm', 'vggresdil2blstm',
                            'vgg8blstm']:
            xs_pad, ilens = self.enc1(xs_pad, ilens)
            xs_pad, ilens = self.enc2(xs_pad, ilens)

            xs_pad = fill_padded_part(xs_pad, ilens, 0.0)

        elif self.etype in ['resblstm', 'resblstmp', 'resbnblstm', 'resbnblstmp', 'resceilblstm', 'resceilblstmp',
                            'resnbblstm', 'resnbblstmp', 'resorigblstm', 'resorigblstm']:
            xs_pad, ilens = self.enc1(xs_pad, ilens)
            xs_pad, ilens = self.enc2(xs_pad, ilens)

            xs_pad = fill_padded_part(xs_pad, ilens, 0.0)

        elif self.etype in ['multiVggblstmBlstmp', 'multiVggdil2blstmBlstmp', 'multiVgg8blstmBlstmp',
                            'multiVggblstmpBlstmp', 'multiVggblstmpBlstmpFixed4', 'multiVggblstmBlstmpFixed4']:
            xs_pad1, ilens1 = self.enc11(xs_pad, ilens)
            xs_pad1, ilens1 = self.enc12(xs_pad1, ilens1)
            xs_pad2, ilens2 = self.enc21(xs_pad, ilens)

            xs_pad1 = fill_padded_part(xs_pad1, ilens1, 0.0)
            xs_pad2 = fill_padded_part(xs_pad2, ilens2, 0.0)

            return (xs_pad1, xs_pad2), (ilens1, ilens2)
        elif self.etype in ['multiBlstmpBlstmp4']:
            xs_pad1, ilens1 = self.enc1(xs_pad, ilens)
            xs_pad2, ilens2 = self.enc2(xs_pad, ilens)

            xs_pad1 = fill_padded_part(xs_pad1, ilens1, 0.0)
            xs_pad2 = fill_padded_part(xs_pad2, ilens2, 0.0)

            return (xs_pad1, xs_pad2), (ilens1, ilens2)
        elif self.etype in ['multiBandBlstmpBlstmp']:
            # xs_pad: utt x frame x dim(83)
            dims1 = list(range(40)) + list(range(80, 83))  # low frequency + 3 pitch
            dims2 = list(range(40, 80)) + list(range(80, 83))  # high frequency + 3 pitch

            xs_pad1 = xs_pad[:, :, dims1]
            xs_pad2 = xs_pad[:, :, dims2]

            if self.addgauss: # decoding stage
                gauss_dist = tdist.Normal(torch.tensor([self.addgauss_mean]), torch.tensor([self.addgauss_std]))
                if self.addgauss_type == 'low43':
                    gauss_noise = gauss_dist.sample(xs_pad1.size()).squeeze(len(xs_pad1.size()))
                    xs_pad1 += gauss_noise
                elif self.addgauss_type == 'high43':
                    gauss_noise = gauss_dist.sample(xs_pad2.size()).squeeze(len(xs_pad2.size()))
                    xs_pad2 += gauss_noise
                elif self.addgauss_type == 'all':
                    gauss_noise1 = gauss_dist.sample(xs_pad1.size()).squeeze(len(xs_pad1.size()))
                    gauss_noise2 = gauss_dist.sample(xs_pad2.size()).squeeze(len(xs_pad2.size()))
                    xs_pad1 += gauss_noise1
                    xs_pad2 += gauss_noise2
                else:
                    logging.error(
                        "Error: need to specify an appropriate addgauss type")
                    sys.exit()

            xs_pad1, ilens1 = self.enc1(xs_pad1, ilens)
            xs_pad2, ilens2 = self.enc2(xs_pad2, ilens)

            xs_pad1 = fill_padded_part(xs_pad1, ilens1, 0.0)
            xs_pad2 = fill_padded_part(xs_pad2, ilens2, 0.0)

            return (xs_pad1, xs_pad2), (ilens1, ilens2)
        elif self.etype in ['highBandBlstmp']:
            # xs_pad: utt x frame x dim(83)
            dims = list(range(40, 80)) + list(range(80, 83))  # high frequency + 3 pitch
            xs_pad = xs_pad[:, :, dims]
            if self.addgauss: # decoding stage
                gauss_dist = tdist.Normal(torch.tensor([self.addgauss_mean]), torch.tensor([self.addgauss_std]))
                gauss_noise = gauss_dist.sample(xs_pad.size()).squeeze(len(xs_pad.size()))
                xs_pad += gauss_noise
            xs_pad, ilens = self.enc1(xs_pad, ilens)

            xs_pad = fill_padded_part(xs_pad, ilens, 0.0)

        elif self.etype in ['lowBandBlstmp']:
            # xs_pad: utt x frame x dim(83)
            dims = list(range(40)) + list(range(80, 83))  # low frequency + 3 pitch
            xs_pad = xs_pad[:, :, dims]
            if self.addgauss: # decoding stage
                gauss_dist = tdist.Normal(torch.tensor([self.addgauss_mean]), torch.tensor([self.addgauss_std]))
                gauss_noise = gauss_dist.sample(xs_pad.size()).squeeze(len(xs_pad.size()))
                xs_pad += gauss_noise
            xs_pad, ilens = self.enc1(xs_pad, ilens)

            xs_pad = fill_padded_part(xs_pad, ilens, 0.0)

        elif self.etype in ['amiCH1BlstmpCH2Blstmp']:
            # xs_pad: utt x frame x dim(83)
            dims1 = list(range(83))  # array 1
            dims2 = list(range(83, 83 * 2))  # array 2

            xs_pad1 = xs_pad[:, :, dims1]
            xs_pad2 = xs_pad[:, :, dims2]

            if self.addgauss: # decoding stage
                gauss_dist = tdist.Normal(torch.tensor([self.addgauss_mean]), torch.tensor([self.addgauss_std]))
                if self.addgauss_type == 'array1':
                    gauss_noise = gauss_dist.sample(xs_pad1.size()).squeeze(len(xs_pad1.size()))
                    xs_pad1 += gauss_noise
                elif self.addgauss_type == 'array2':
                    gauss_noise = gauss_dist.sample(xs_pad2.size()).squeeze(len(xs_pad2.size()))
                    xs_pad2 += gauss_noise
                elif self.addgauss_type == 'arrayall':
                    gauss_noise1 = gauss_dist.sample(xs_pad1.size()).squeeze(len(xs_pad1.size()))
                    gauss_noise2 = gauss_dist.sample(xs_pad2.size()).squeeze(len(xs_pad2.size()))
                    xs_pad1 += gauss_noise1
                    xs_pad2 += gauss_noise2
                else:
                    logging.error(
                        "Error: need to specify an appropriate addgauss type")
                    sys.exit()

            xs_pad1, ilens1 = self.enc1(xs_pad1, ilens)
            xs_pad2, ilens2 = self.enc2(xs_pad2, ilens)

            xs_pad1 = fill_padded_part(xs_pad1, ilens1, 0.0)
            xs_pad2 = fill_padded_part(xs_pad2, ilens2, 0.0)

            return (xs_pad1, xs_pad2), (ilens1, ilens2)

        elif self.etype in ['amiCH1Blstmp']:
            # xs_pad: utt x frame x dim(83)
            dims1 = list(range(83))  # high frequency + 3 pitch
            xs_pad = xs_pad[:, :, dims1]
            if self.addgauss: # decoding stage
                gauss_dist = tdist.Normal(torch.tensor([self.addgauss_mean]), torch.tensor([self.addgauss_std]))
                gauss_noise = gauss_dist.sample(xs_pad.size()).squeeze(len(xs_pad.size()))
                xs_pad += gauss_noise
            xs_pad, ilens = self.enc1(xs_pad, ilens)
            xs_pad = fill_padded_part(xs_pad, ilens, 0.0)

        elif self.etype in ['amiCH2Blstmp']:
            # xs_pad: utt x frame x dim(83)
            dims2 = list(range(83, 83 * 2))  # low frequency + 3 pitch
            xs_pad = xs_pad[:, :, dims2]
            if self.addgauss: # decoding stage
                gauss_dist = tdist.Normal(torch.tensor([self.addgauss_mean]), torch.tensor([self.addgauss_std]))
                gauss_noise = gauss_dist.sample(xs_pad.size()).squeeze(len(xs_pad.size()))
                xs_pad += gauss_noise
            xs_pad, ilens = self.enc1(xs_pad, ilens)
            xs_pad = fill_padded_part(xs_pad, ilens, 0.0)

        elif self.etype in ['amiCH1VggblstmCH2Vggblstm']:
            # xs_pad: utt x frame x dim(83)
            dims1 = list(range(83))  # low frequency + 3 pitch
            dims2 = list(range(83, 83 * 2))  # high frequency + 3 pitch

            xs_pad1, ilens1 = self.enc11(xs_pad[:, :, dims1], ilens)
            xs_pad1, ilens1 = self.enc12(xs_pad1, ilens1)

            xs_pad2, ilens2 = self.enc21(xs_pad[:, :, dims2], ilens)
            xs_pad2, ilens2 = self.enc22(xs_pad2, ilens2)

            xs_pad1 = fill_padded_part(xs_pad1, ilens1, 0.0)
            xs_pad2 = fill_padded_part(xs_pad2, ilens2, 0.0)

            return (xs_pad1, xs_pad2), (ilens1, ilens2)

        elif self.etype in ['amiCH1Vggblstm']:
            # xs_pad: utt x frame x dim(83)
            dims1 = list(range(83))  # high frequency + 3 pitch
            xs_pad, ilens = self.enc11(xs_pad[:, :, dims1], ilens)
            xs_pad, ilens = self.enc12(xs_pad, ilens)
            xs_pad = fill_padded_part(xs_pad, ilens, 0.0)
        elif self.etype in ['amiCH2Vggblstm']:
            # xs_pad: utt x frame x dim(83)
            dims2 = list(range(83, 83 * 2))  # low frequency + 3 pitch
            xs_pad, ilens = self.enc21(xs_pad[:, :, dims2], ilens)
            xs_pad, ilens = self.enc22(xs_pad, ilens)
            xs_pad = fill_padded_part(xs_pad, ilens, 0.0)
        else:
            logging.error(
                "Error: need to specify an appropriate encoder archtecture")
            sys.exit()

        # make mask to remove bias value in padded part
        mask = to_cuda(self, make_pad_mask(ilens).unsqueeze(-1))

        return xs_pad.masked_fill(mask, 0.0), ilens


class BLSTMP(torch.nn.Module):
    """Bidirectional LSTM with projection layer module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of lstm units (resulted in cdim * 2 due to biderectional)
    :param int hdim: number of projection units
    :param list subsample: list of subsampling numbers
    :param float dropout: dropout rate
    """

    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout):
        super(BLSTMP, self).__init__()
        for i in six.moves.range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim
            setattr(self, "bilstm%d" % i, torch.nn.LSTM(inputdim, cdim, dropout=dropout,
                                                        num_layers=1, bidirectional=True, batch_first=True))
            # bottleneck layer to merge
            setattr(self, "bt%d" % i, torch.nn.Linear(2 * cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample

    def forward(self, xs_pad, ilens):
        '''BLSTMP forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, hdim)
        :rtype: torch.Tensor
        '''
        # logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        for layer in six.moves.range(self.elayers):
            xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
            bilstm = getattr(self, 'bilstm' + str(layer))
            bilstm.flatten_parameters()
            ys, _ = bilstm(xs_pack)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ys_pad = ys_pad[:, ::sub]
                ilens = [int(i + 1) // sub for i in ilens]
            # (sum _utt frame_utt) x dim
            projected = getattr(self, 'bt' + str(layer)
                                )(ys_pad.contiguous().view(-1, ys_pad.size(2)))
            xs_pad = torch.tanh(projected.view(ys_pad.size(0), ys_pad.size(1), -1))

        return xs_pad, ilens  # x: utt list of frame x dim


class BLSTMPBN(torch.nn.Module):
    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout):
        super(BLSTMPBN, self).__init__()
        for i in six.moves.range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim
            setattr(self, "bilstm%d" % i, torch.nn.LSTM(inputdim, cdim, dropout=dropout,
                                                        num_layers=1, bidirectional=True, batch_first=True))
            # bottleneck layer to merge
            setattr(self, "bt%d" % i, torch.nn.Linear(2 * cdim, hdim))

            setattr(self, "batchnorm%d" % i, torch.nn.BatchNorm1d(hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample

    def forward(self, xpad, ilens):
        '''BLSTMP forward

        :param xs:
        :param ilens:
        :return:
        '''
        # logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        for layer in six.moves.range(self.elayers):
            xpack = pack_padded_sequence(xpad, ilens, batch_first=True)
            bilstm = getattr(self, 'bilstm' + str(layer))
            bilstm.flatten_parameters()
            ys, _ = bilstm(xpack)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            ypad, ilens = pad_packed_sequence(ys, batch_first=True)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ypad = ypad[:, ::sub]
                ilens = [(i + 1) // sub for i in ilens]
            # (sum _utt frame_utt) x dim
            projected = getattr(self, 'batchnorm' + str(layer)
                                )((getattr(self, 'bt' + str(layer)
                                           )(ypad.contiguous().view(-1, ypad.size(2)))))
            xpad = torch.tanh(projected.view(ypad.size(0), ypad.size(1), -1))

        return xpad, ilens  # x: utt list of frame x dim


class BLSTMSS(torch.nn.Module):
    def __init__(self, idim, elayers, cdim, hdim, subsample, dropout):
        super(BLSTMSS, self).__init__()
        for i in six.moves.range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = cdim * 2
            setattr(self, "bilstm%d" % i, torch.nn.LSTM(inputdim, cdim, dropout=dropout,
                                                        num_layers=1, bidirectional=True, batch_first=True))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample

        self.l_last = torch.nn.Linear(cdim * 2, hdim)

    def forward(self, xpad, ilens):
        '''BLSTMSS forward # TODO: memeory issue


        :param xs:
        :param ilens:
        :return:
        '''
        # logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        for layer in six.moves.range(self.elayers):
            xpack = pack_padded_sequence(xpad, ilens, batch_first=True)
            bilstm = getattr(self, 'bilstm' + str(layer))
            bilstm.flatten_parameters()
            ys, _ = bilstm(xpack)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            ypad, ilens = pad_packed_sequence(ys, batch_first=True)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ypad = ypad[:, ::sub]
                ilens = [(i + 1) // sub for i in ilens]
            # (sum _utt frame_utt) x dim
            xpad = ypad.contiguous().view(ypad.size(0), ypad.size(1), -1)

        # (sum _utt frame_utt) x dim
        projected = torch.tanh(self.l_last(
            xpad.contiguous().view(-1, xpad.size(2))))
        xpad = projected.view(xpad.size(0), xpad.size(1), -1)
        return xpad, ilens  # x: utt list of frame x dim


class BLSTM(torch.nn.Module):
    """Bidirectional LSTM module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of lstm units (resulted in cdim * 2 due to biderectional)
    :param int hdim: number of final projection units
    :param list subsample: list of subsampling numbers
    :param float dropout: dropout rate
    """

    def __init__(self, idim, elayers, cdim, hdim, dropout):
        super(BLSTM, self).__init__()
        self.nblstm = torch.nn.LSTM(idim, cdim, elayers, batch_first=True,
                                    dropout=dropout, bidirectional=True)
        self.l_last = torch.nn.Linear(cdim * 2, hdim)

    def forward(self, xs_pad, ilens):
        '''BLSTM forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, erojs)
        :rtype: torch.Tensor
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
        self.nblstm.flatten_parameters()
        ys, _ = self.nblstm(xs_pack)
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
        # (sum _utt frame_utt) x dim
        projected = torch.tanh(self.l_last(
            ys_pad.contiguous().view(-1, ys_pad.size(2))))
        xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)
        return xs_pad, ilens  # x: utt list of frame x dim


class VGG2L(torch.nn.Module):
    """VGG-like module

    :param int in_channel: number of input channels
    """

    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.in_channel = in_channel

    def forward(self, xs_pad, ilens):
        '''VGG2L forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of padded hidden state sequences (B, Tmax // 4, 128)
        :rtype: torch.Tensor
        '''
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        # xs_pad = F.pad_sequence(xs_pad)

        # x: utt x 1 (input channel num) x frame x dim
        xs_pad = xs_pad.contiguous().view(xs_pad.size(0), xs_pad.size(1), self.in_channel,
                                          xs_pad.size(2) // self.in_channel).transpose(1, 2)

        # NOTE: max_pool1d ?
        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)

        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3))
        xs_pad = [xs_pad[i, :ilens[i]] for i in range(len(ilens))]
        xs_pad = pad_list(xs_pad, 0.0)
        return xs_pad, ilens


######################## Reimplementation of CNN part (Ruizhi)

class ConvBasicBlock(torch.nn.Module):
    """convolution followed by optional batchnorm, nonlinear, spatialdropout, maxpooling"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, nonlinear=False,
                 batch_norm=False, spatial_dp=False, maxpooling=False, maxpool_kernel_size=3, maxpool_stride=2,
                 maxpool_padding=1, maxpool_ceil_mode=False, inplace=True, dilation=1):
        super(ConvBasicBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=bias, dilation=dilation)
        # todo see performace diff with/without eval() in dp and bn case

        if batch_norm: self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        if nonlinear: self.relu = torch.nn.ReLU(inplace=inplace)
        if spatial_dp: self.sdp = torch.nn.Dropout2d()  # spatial dropout
        if maxpooling: self.maxpool = torch.nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride,
                                                         padding=maxpool_padding, ceil_mode=maxpool_ceil_mode)

        self.batch_norm = batch_norm
        self.nonlinear = nonlinear
        self.spatial_dp = spatial_dp
        self.maxpooling = maxpooling

    def forward(self, x):
        out = self.conv(x)
        if self.batch_norm: out = self.batchnorm(out)
        if self.nonlinear: out = self.relu(out)
        if self.spatial_dp: out = self.sdp(out)
        if self.maxpooling: out = self.maxpool(out)
        return out


class CNNBasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, batch_norm=False, spatial_dp=False, resnet=False,
                 bias=False, dilation=1):
        super(CNNBasicBlock, self).__init__()

        # residual: conv3*3 --> (spatial_dp) --> (batchnorm) --> relu --> cov3*3 --> (spatial_dp) --> (batchnorm)
        # shortcut: conv1*1 --> (spatial_dp) --> (batchnorm)  (optional for resnet)
        # out: relu(residual + [shortcut])

        self.conv1 = ConvBasicBlock(in_channels, out_channels, 3, stride, dilation, bias, True, batch_norm, spatial_dp,
                                    dilation=dilation)
        self.conv2 = ConvBasicBlock(out_channels, out_channels, 3, 1, dilation, bias, False, batch_norm, spatial_dp,
                                    dilation=dilation)
        self.relu = torch.nn.ReLU(inplace=True)
        if (in_channels != out_channels or stride != 1) and resnet:
            self.downsample = ConvBasicBlock(in_channels, out_channels, 1, stride, 0, bias, False, batch_norm,
                                             spatial_dp)
            self.downsampling = True
        else:
            self.downsampling = False
        self.resnet = resnet

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsampling:
            residual = self.downsample(residual)

        if self.resnet:
            out += residual
        out = self.relu(out)
        return out


class VGG(torch.nn.Module):

    # default is vgg
    # VGG:  Vgg()
    # add Residual connection: (bias=False, resnet=True)

    def __init__(self, batch_norm=False, spatial_dp=False, ceil_mode=False, in_channels=1, bias=True, resnet=False,
                 dilation=1):
        super(VGG, self).__init__()

        # self.conv0 = ConvBasicBlock(in_channels, 16, 1, 1, 0, False, True, batch_norm, spatial_dp)
        self.resblock1 = CNNBasicBlock(in_channels, 64, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=resnet,
                                       bias=bias, dilation=dilation)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)

        self.resblock2 = CNNBasicBlock(64, 128, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=resnet, bias=bias,
                                       dilation=dilation)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)
        self.in_channels = in_channels
        self.ceil_mode = ceil_mode
        self.dilation = dilation

    def forward(self, xs, ilens):
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # xs: input: utt x frame x dim
        # xs: output: utt x 1 (input channel num) x frame x dim
        xs = xs.contiguous().view(xs.size(0), xs.size(1), self.in_channels, xs.size(2) // self.in_channels).transpose(1,
                                                                                                                      2)

        # xs = self.conv0(xs)
        xs = self.resblock1(xs)
        xs = self.maxpool1(xs)
        xs = self.resblock2(xs)
        xs = self.maxpool2(xs)

        # change ilens accordingly
        # in maxpooling layer: stride(2), padding(0), kernel(2)

        fnc = np.ceil if self.ceil_mode else np.floor

        # s, p, k = [2, 0, 2]
        # ilens = np.array(
        #     fnc(((np.array(ilens, dtype=np.float32)+2*p-k)/s)+1), dtype=np.int64)
        # ilens = np.array(
        #     fnc(((np.array(ilens, dtype=np.float32)+2*p-k)/s)+1), dtype=np.int64).tolist()

        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [2, 0, 2];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - k) / s) + 1), dtype=np.int64)

        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [2, 0, 2];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - k) / s) + 1), dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = xs.transpose(1, 2)

        xs = xs.contiguous().view(
            xs.size(0), xs.size(1), xs.size(2) * xs.size(3))

        xs = [xs[i, :ilens[i]] for i in range(len(ilens))]
        xs = pad_list(xs, 0.0)

        return xs, ilens


class VGG8(torch.nn.Module):

    # default is vgg
    # VGG:  Vgg() first 8 layers of vgg
    # add Residual connection: (bias=False, resnet=True)

    def __init__(self, batch_norm=False, spatial_dp=False, ceil_mode=False, in_channels=1, bias=True, resnet=False,
                 dilation=1):
        super(VGG8, self).__init__()

        # self.conv0 = ConvBasicBlock(in_channels, 16, 1, 1, 0, False, True, batch_norm, spatial_dp)
        self.resblock1 = CNNBasicBlock(in_channels, 64, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=resnet,
                                       bias=bias, dilation=dilation)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)

        self.resblock2 = CNNBasicBlock(64, 128, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=resnet, bias=bias,
                                       dilation=dilation)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)
        self.resblock3 = CNNBasicBlock(128, 256, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=resnet, bias=bias,
                                       dilation=dilation)
        self.resblock4 = CNNBasicBlock(256, 256, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=resnet, bias=bias,
                                       dilation=dilation)

        self.in_channels = in_channels
        self.ceil_mode = ceil_mode
        self.dilation = dilation

    def forward(self, xs, ilens):
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # xs: input: utt x frame x dim
        # xs: output: utt x 1 (input channel num) x frame x dim
        xs = xs.contiguous().view(xs.size(0), xs.size(1), self.in_channels, xs.size(2) // self.in_channels).transpose(1,
                                                                                                                      2)

        # xs = self.conv0(xs)
        xs = self.resblock1(xs)
        xs = self.maxpool1(xs)
        xs = self.resblock2(xs)
        xs = self.maxpool2(xs)
        xs = self.resblock3(xs)
        xs = self.resblock4(xs)

        # change ilens accordingly
        # in maxpooling layer: stride(2), padding(0), kernel(2)

        fnc = np.ceil if self.ceil_mode else np.floor

        # resblock 1 + maxpooling
        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [2, 0, 2];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - k) / s) + 1), dtype=np.int64)

        # resblock 2 + maxpooling
        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [2, 0, 2];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - k) / s) + 1), dtype=np.int64)

        # resblock 3
        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)

        # resblock 4
        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = xs.transpose(1, 2)

        xs = xs.contiguous().view(
            xs.size(0), xs.size(1), xs.size(2) * xs.size(3))

        xs = [xs[i, :ilens[i]] for i in range(len(ilens))]
        xs = pad_list(xs, 0.0)

        return xs, ilens


class VGGRes(torch.nn.Module):

    # default is 2 layer vgg followed by 1 layer resnet
    # VGGRes:  Vgg()
    # add Residual connection: (bias=False, resnet=True)

    def __init__(self, batch_norm=False, spatial_dp=False, ceil_mode=False, in_channels=1, bias=True, resnet=False,
                 dilation=1):
        super(VGGRes, self).__init__()

        # self.conv0 = ConvBasicBlock(in_channels, 16, 1, 1, 0, False, True, batch_norm, spatial_dp)
        self.resblock1 = CNNBasicBlock(in_channels, 64, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=resnet,
                                       bias=bias, dilation=dilation)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)

        self.resblock2 = CNNBasicBlock(64, 128, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=resnet, bias=bias,
                                       dilation=dilation)
        self.resblock3 = CNNBasicBlock(128, 128, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=True, bias=False)

        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)
        self.in_channels = in_channels
        self.ceil_mode = ceil_mode
        self.dilation = dilation

    def forward(self, xs, ilens):
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # xs: input: utt x frame x dim
        # xs: output: utt x 1 (input channel num) x frame x dim
        xs = xs.contiguous().view(xs.size(0), xs.size(1), self.in_channels, xs.size(2) // self.in_channels).transpose(1,
                                                                                                                      2)

        # xs = self.conv0(xs)
        xs = self.resblock1(xs)
        xs = self.maxpool1(xs)
        xs = self.resblock2(xs)
        xs = self.maxpool2(xs)
        xs = self.resblock3(xs)

        # change ilens accordingly
        # in maxpooling layer: stride(2), padding(0), kernel(2)

        fnc = np.ceil if self.ceil_mode else np.floor

        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [2, 0, 2];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - k) / s) + 1), dtype=np.int64)

        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [1, self.dilation, 3];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - self.dilation * (k - 1) - 1) / s) + 1),
                         dtype=np.int64)
        s, p, k = [2, 0, 2];
        ilens = np.array(fnc(((np.array(ilens, dtype=np.float32) + 2 * p - k) / s) + 1), dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = xs.transpose(1, 2)

        xs = xs.contiguous().view(
            xs.size(0), xs.size(1), xs.size(2) * xs.size(3))

        xs = [xs[i, :ilens[i]] for i in range(len(ilens))]
        xs = pad_list(xs, 0.0)

        return xs, ilens


class VGGOnly(torch.nn.Module):

    # default is vgg
    # VGG:  Vgg()
    # add Residual connection: (bias=False, resnet=True)

    def __init__(self, idim, eprojs, batch_norm=False, spatial_dp=False, ceil_mode=False, in_channels=1, bias=True,
                 resnet=False, dilation=1):
        super(VGGOnly, self).__init__()

        # self.conv0 = ConvBasicBlock(in_channels, 16, 1, 1, 0, False, True, batch_norm, spatial_dp)
        self.resblock1 = CNNBasicBlock(in_channels, 64, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=resnet,
                                       bias=bias, dilation=dilation)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)

        self.resblock2 = CNNBasicBlock(64, 128, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=resnet, bias=bias,
                                       dilation=dilation)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)

        self.linear = torch.nn.Linear(get_maxpooling2_odim(idim, in_channel=in_channels), eprojs)
        self.in_channels = in_channels
        self.ceil_mode = ceil_mode

    def forward(self, xs, ilens):
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # xs: input: utt x frame x dim
        # xs: output: utt x 1 (input channel num) x frame x dim
        xs = xs.contiguous().view(xs.size(0), xs.size(1), self.in_channels, xs.size(2) // self.in_channels).transpose(1,
                                                                                                                      2)

        # xs = self.conv0(xs)
        xs = self.resblock1(xs)
        xs = self.maxpool1(xs)
        xs = self.resblock2(xs)
        xs = self.maxpool2(xs)

        # change ilens accordingly
        # in maxpooling layer: stride(2), padding(0), kernel(2)
        s, p, k = [2, 0, 2]

        fnc = np.ceil if self.ceil_mode else np.floor

        ilens = np.array(
            fnc(((np.array(ilens, dtype=np.float32) + 2 * p - k) / s) + 1), dtype=np.int64)
        ilens = np.array(
            fnc(((np.array(ilens, dtype=np.float32) + 2 * p - k) / s) + 1), dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = xs.transpose(1, 2)

        xs = xs.contiguous().view(
            xs.size(0), xs.size(1), xs.size(2) * xs.size(3))

        xs = self.linear(xs)  # projection to eprojs

        xs = [xs[i, :ilens[i]] for i in range(len(ilens))]
        xs = pad_list(xs, 0.0)

        return xs, ilens


class ResNetOrig(torch.nn.Module):
    # ceil mode is not supported due to conv.
    def __init__(self, batch_norm=False, spatial_dp=False, ceil_mode=False, in_channels=1, inplace=True, dilation=1):
        super(ResNetOrig, self).__init__()

        self.conv0 = ConvBasicBlock(in_channels, 64, 7, 2, 3, False, True, batch_norm, spatial_dp)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=ceil_mode)

        self.resblock1 = CNNBasicBlock(64, 64, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=True,
                                       inplace=inplace, dilation=dilation)
        self.resblock2 = CNNBasicBlock(64, 64, batch_norm=batch_norm, spatial_dp=spatial_dp, resnet=True,
                                       inplace=inplace, dilation=dilation)

        self.in_channels = in_channels
        self.ceil_mode = ceil_mode

    def forward(self, xs, ilens):
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # xs: input: utt x frame x dim
        # xs: output: utt x 1 (input channel num) x frame x dim
        xs = xs.contiguous().view(xs.size(0), xs.size(1), self.in_channels, xs.size(2) // self.in_channels).transpose(1,
                                                                                                                      2)

        xs = self.conv0(xs)
        xs = self.resblock1(xs)
        xs = self.resblock2(xs)

        # change ilens accordingly
        # in maxpooling layer: stride(2), padding(0), kernel(2)
        s, p, k = [2, 3, 7]

        fnc = np.ceil if self.ceil_mode else np.floor
        s, p, k = [2, 3, 7]
        ilens = np.array(
            fnc(((np.array(ilens, dtype=np.float32) + 2 * p - k) / s) + 1), dtype=np.int64)
        s, p, k = [2, 1, 3]
        ilens = np.array(
            fnc(((np.array(ilens, dtype=np.float32) + 2 * p - k) / s) + 1), dtype=np.int64).tolist()

        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = xs.transpose(1, 2)

        xs = xs.contiguous().view(
            xs.size(0), xs.size(1), xs.size(2) * xs.size(3))

        xs = [xs[i, :ilens[i]] for i in range(len(ilens))]
        xs = pad_list(xs, 0.0)

        return xs, ilens


######################## Implementation of RCNN in https://arxiv.org/pdf/1702.07793.pdf

def conv3x3(inplanes, planes, stride=(1, 1), padding=(1, 1), dilation=(1, 1), bias=False):
    """3x3 convolution with padding
        Hout = floor((Hin+2xpadding-dilationx(kernel_size-1)-1)/stride+1)
    """
    return torch.nn.Conv2d(inplanes, planes,
                           kernel_size=3,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           bias=bias)


class BasicBlock_BnReluConv(torch.nn.Module):

    def __init__(self, inplanes, planes, stride=(1, 1), downsample=None, flag_bn=True, flag_relu=True, flag_dp=False):
        super(BasicBlock_BnReluConv, self).__init__()

        if flag_bn:
            self.batchnorm1 = torch.nn.BatchNorm2d(inplanes)
            self.batchnorm2 = torch.nn.BatchNorm2d(planes)

        if flag_relu:
            self.relu = torch.nn.ReLU(inplace=True)

        if flag_dp:
            self.dp = torch.nn.Dropout2d()  # plane dropout

        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.conv2 = conv3x3(planes, planes)

        self.downsample = downsample
        self.stride = stride
        self.flag_bn, self.flag_relu, self.flag_dp = flag_bn, flag_relu, flag_dp

    def forward(self, x):
        residual = x

        if self.flag_bn:
            out = self.batchnorm1(x)
        else:
            out = x
        if self.flag_relu: out = self.relu(out)

        out = self.conv1(out)

        if self.flag_dp: out = self.dp(out)

        if self.flag_bn:   out = self.batchnorm2(out)
        if self.flag_relu: out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class RCNN(torch.nn.Module):
    """
    Implementation of RCNN in Residual Convolutional CTC Networks for Automatic Speech Recognition
    https://arxiv.org/pdf/1702.07793.pdf
    """

    def __init__(self, idim, block, eprojs, layers=(2, 2, 2, 2), expansion=2, inplanes=1, flag_bn=True, flag_relu=True,
                 flag_dp=False):
        self.inplanes_updated = 32
        super(RCNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(11, 41), stride=(2, 2), padding=(5, 20), bias=False)

        self.layer1 = self._make_layer(block, 64 * expansion, layers[0], flag_bn=flag_bn, flag_relu=flag_relu,
                                       flag_dp=flag_dp)
        self.layer2 = self._make_layer(block, 128 * expansion, layers[1], flag_bn=flag_bn, flag_relu=flag_relu,
                                       flag_dp=flag_dp)
        self.layer3 = self._make_layer(block, 256 * expansion, layers[2], stride=(1, 2), flag_bn=flag_bn,
                                       flag_relu=flag_relu, flag_dp=flag_dp)
        self.layer4 = self._make_layer(block, 512 * expansion, layers[3], stride=(2, 2), flag_bn=flag_bn,
                                       flag_relu=flag_relu, flag_dp=flag_dp)

        # stride, padding, dilation, kernel_size
        self.params_freq = [[2, 20, 1, 41], [1, 1, 1, 3], [1, 1, 1, 3], [2, 1, 1, 3], [2, 1, 1, 3]]

        self.params_time = [[2, 5, 1, 11], [1, 1, 1, 3], [1, 1, 1, 3], [1, 1, 1, 3], [2, 1, 1, 3]]
        self.linear = torch.nn.Linear(get_RCNN_odim(idim, self.params_freq, inplanes=inplanes, planes=512 * expansion),
                                      eprojs)
        self.inplanes = inplanes

        # TODO
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=(1, 1), flag_bn=True, flag_relu=True, flag_dp=False):
        downsample = None

        if stride[0] != 1 or stride[1] != 1 or self.inplanes_updated != planes:
            downsample = torch.nn.Conv2d(self.inplanes_updated, planes, kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(block(self.inplanes_updated, planes, stride, downsample, flag_bn=flag_bn, flag_relu=flag_relu,
                            flag_dp=flag_dp))

        self.inplanes_updated = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes_updated, planes, flag_bn=flag_bn, flag_relu=flag_relu, flag_dp=flag_dp))

        return torch.nn.Sequential(*layers)

    def forward(self, xs, ilens):
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # xs: input: utt x frame x dim
        # xs: output: utt x 1 (input channel num) x frame x dim
        xs = xs.contiguous().view(xs.size(0), xs.size(1), self.inplanes, xs.size(2) // self.inplanes).transpose(1, 2)

        xs = self.conv1(xs)
        xs = self.layer1(xs)
        xs = self.layer2(xs)
        xs = self.layer3(xs)
        xs = self.layer4(xs)

        # change ilens accordingly
        ilens = get_RCNN_ilens(ilens, self.params_time)
        # x: utt_list of frame (remove zeropaded frames) x (input channel num x dim)
        xs = xs.transpose(1, 2)
        xs = xs.contiguous().view(
            xs.size(0), xs.size(1), xs.size(2) * xs.size(3))
        xs = self.linear(xs)  # projection to eprojs
        xs = [xs[i, :ilens[i]] for i in range(len(ilens))]
        xs = pad_list(xs, 0.0)

        return xs, ilens
