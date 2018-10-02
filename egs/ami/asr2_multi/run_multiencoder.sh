#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 Johns Hoplins University (Xiaofei Wang, Ruizhi Li)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0       # start from -1 if you need to start from data download
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false

# network architecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=8
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
adim=320
awin=5
aheads=4
aconv_chans=10
aconv_filts=100

# hybrid CTC/attention
mtlalpha=0.5 # TODO: check origin 0.5 and wsj 0.2

lsm_type=unigram # todo: orig '', wsj:unigram
lsm_weight=0.05 # todo: orig 0, wsj:0.05

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=15

# sgd parameters
lr=1e-3
lr_decay=1e-1
mom=0.9
wd=0

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_layers=1         # 2 for character LMs
lm_units=1000       # 650 for character LMs
lm_opt=sgd          # adam for character LMs
lm_batchsize=300    # 1024 for character LMs
lm_epochs=20        # number of epochs
lm_maxlen=40        # 150 for character LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs
has_fisher_swbd=  # ''[empty], fisher, swbd, fisher_swbd
use_lm=true

# lm data
fisher_dirs="/export/corpora3/LDC/LDC2004T19 /export/corpora3/LDC/LDC2005T19 /export/corpora3/LDC/LDC2004S13 /export/corpora3/LDC/LDC2005S13"
swbd1_dir=/export/corpora3/LDC/LDC97S62

# decoding parameter
lm_weight=1.0
beam_size=20 # TODO: orig: 20, wsj: 30
penalty=0.2 # TODO: orig:0.2, wsj:0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# You may set 'mic' to:
#  ihm [individual headset mic- the default which gives best results]
#  sdm1 [single distant microphone- the current script allows you only to select
#        the 1st of 8 microphones]
#  mdm8 [multiple distant microphones-- currently we only support averaging over
#       the 8 source microphones].
#  smdm8 [second multiple distant microphones]
# ... by calling this script as, for example,
# ./run.sh --mic sdm1
# ./run.sh --mic mdm8
# ./run.sh --mic smdm8
mic=ihm

# exp tag
tag="" # tag for managing experiments.

# multl-encoder multi-band
num_enc=1
share_ctc=true
l2_dropout=0.5

# for decoding only ; only works for multi case
l2_weight=0.5

# add gaussian noise to the features (only works for encoder type: 'multiBandBlstmpBlstmp', 'blstm', 'blstmp', 'blstmss', 'blstmpbn', 'vgg', 'rcnn', 'rcnnNObn', 'rcnnDp', 'rcnnDpNObn')
addgauss=false
addgauss_mean=0
addgauss_std=1
addgauss_type=all # all, high43 low43

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

base_mic=$(echo $mic | sed 's/[0-9]//g') # sdm, ihm or mdm
nmics=$(echo $mic | sed 's/[a-z]//g') # e.g. 8 for mdm8.

# Path where AMI gets downloaded (or where locally available):
AMI_DIR=$PWD/wav_db # Default,
case $(hostname -d) in
    clsp.jhu.edu) AMI_DIR=/export/corpora4/ami/amicorpus ;; # JHU,
esac

if [ ${stage} -le 0 ]; then

    echo "stage 0: Data concatenating ..."
    mkdir -p data/mdm_multistream/train_orig
    cp data/mdm8_train/* data/mdm_multistream/train_orig
    rm -r data/mdm_multistream/train_orig/feats.scp
    rm -r data/mdm_multistream/train_orig/cmvn.ark
    utils/data/copy_data_dir.sh data/mdm_multistream/train_orig data/mdm_multistream_train
    paste-feats scp:data/mdm8_train/feats.scp scp:data/smdm8_train/feats.scp ark,scp:fbank/raw_fbank_pitch_mdm_multistream_train.ark,data/mdm_multistream_train/feats.scp
    compute-cmvn-stats scp:data/mdm_multistream_train/feats.scp data/mdm_multistream_train/cmvn.ark

    mkdir -p data/mdm_multistream/dev_orig
    cp data/mdm8_dev/* data/mdm_multistream/dev_orig
    rm -r data/mdm_multistream/dev_orig/feats.scp
    utils/data/copy_data_dir.sh data/mdm_multistream/dev_orig data/mdm_multistream_dev
    paste-feats scp:data/mdm8_dev/feats.scp scp:data/smdm8_dev/feats.scp ark,scp:fbank/raw_fbank_pitch_mdm_multistream_dev.ark,data/mdm_multistream_dev/feats.scp
    compute-cmvn-stats scp:data/mdm_multistream_dev/feats.scp data/mdm_multistream_dev/cmvn.ark

    mkdir -p data/mdm_multistream/eval_orig
    cp data/mdm8_eval/* data/mdm_multistream/eval_orig
    rm -r data/mdm_multistream/eval_orig/feats.scp
    utils/data/copy_data_dir.sh data/mdm_multistream/eval_orig data/mdm_multistream_eval
    paste-feats scp:data/mdm8_eval/feats.scp scp:data/smdm8_eval/feats.scp ark,scp:fbank/raw_fbank_pitch_mdm_multistream_eval.ark,data/mdm_multistream_eval/feats.scp
    compute-cmvn-stats scp:data/mdm_multistream_eval/feats.scp data/mdm_multistream_eval/cmvn.ark

fi
train_set=mdm_multistream_train
train_dev=mdm_multistream_dev
train_test=mdm_multistream_eval
recog_set="mdm_multistream_dev mdm_multistream_eval"
feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{04,05,06,07}/${USER}/espnet-data/egs/ami/asr2_multi/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{04,05,06,07}/${USER}/espnet-data/egs/ami/asr2_multi/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

# It takes about one day. If you just want to do end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
    if [ $use_wordlm = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
    if [ ! -z $has_fisher_swbd ]; then
        lmtag=${lmtag}_$has_fisher_swbd
    fi
fi
lmexpdir=exp/train_rnnlm_${backend}_${lmtag}
mkdir -p ${lmexpdir}

if [[ ${stage} -le 3 && $use_lm == true ]]; then
    echo "stage 3: LM Preparation"

    if [ ! -z $has_fisher_swbd ];then
    echo "Prepare fisher and switchboard data for LM training"
    local/fisher_data_prep.sh ${fisher_dirs}
    local/swbd1_data_download.sh ${swbd1_dir}
    local/fisher_swbd_prepare_dict.sh
    chmod 644 data/local/dict_nosp/lexicon0.txt
    local/swbd1_data_prep.sh ${swbd1_dir}
    fi

    if [ $use_wordlm = true ]; then
        lmdatadir=data/local/wordlm_train
        [ ! -z $has_fisher_swbd ] && lmdatadir=${lmdatadir}_${has_fisher_swbd}
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}
        cat data/${train_set}/text | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt

        touch ${lmdatadir}/train_others.txt
        if [ $has_fisher_swbd == 'fisher' ]; then # TODO
            cat data/train_fisher/text | grep -v "]" | tr [a-z] [A-Z] > ${lmdatadir}/train_others.txt
        elif [ $has_fisher_swbd == 'swbd' ]; then
            cat data/train_swbd/text | grep -v "]" | tr [a-z] [A-Z] > ${lmdatadir}/train_others.txt
        elif [ $has_fisher_swbd == 'fisher_swbd' ]; then
            cat data/train_swbd/text data/train_fisher/text | grep -v "]" | tr [a-z] [A-Z] > ${lmdatadir}/train_others.txt
        fi

        cat data/${train_dev}/text | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        cat data/${train_test}/text | cut -f 2- -d" " > ${lmdatadir}/test.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
        lmdatadir=data/local/lm_train
        [ ! -z $has_fisher_swbd ] && lmdatadir=${lmdatadir}_${has_fisher_swbd}
        lmdict=$dict
        mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 data/${train_set}/text \
            | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt

        touch ${lmdatadir}/train_others.txt
        if [ $has_fisher_swbd == 'fisher' ]; then # TODO
            cat data/train_fisher/text | grep -v "]" | tr [a-z] [A-Z] \
            | text2token.py -n 1 | cut -f 2- -d" " > ${lmdatadir}/train_others.txt
        elif [ $has_fisher_swbd == 'swbd' ]; then
            cat data/train_swbd/text | grep -v "]" | tr [a-z] [A-Z] \
            | text2token.py -n 1 | cut -f 2- -d" " > ${lmdatadir}/train_others.txt
        elif [ $has_fisher_swbd == 'fisher_swbd' ]; then
            cat data/train_swbd/text data/train_fisher/text | grep -v "]" | tr [a-z] [A-Z] \
            | text2token.py -n 1 | cut -f 2- -d" " > ${lmdatadir}/train_others.txt
        fi

        text2token.py -s 1 -n 1 data/${train_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        text2token.py -s 1 -n 1 data/${train_test}/text \
                | cut -f 2- -d" " > ${lmdatadir}/test.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
    fi
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --test-label ${lmdatadir}/test.txt \
        --resume ${lm_resume} \
        --layer ${lm_layers} \
        --unit ${lm_units} \
        --opt ${lm_opt} \
        --batchsize ${lm_batchsize} \
        --epoch ${lm_epochs} \
        --maxlen ${lm_maxlen} \
        --dict ${lmdict}
fi

if [ -z ${tag} ]; then
    expdir=exp/${train_set}_${backend}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_shareCtc${share_ctc}

    if [ $atype == 'enc2_add_l2dp' ]; then
        expdir=${expdir}_l2attdp${l2_dropout}
    fi

    if [ "${lsm_type}" != "" ]; then
        expdir=${expdir}_lsm${lsm_type}${lsm_weight}
    fi
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/${train_set}_${backend}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json \
        --etype ${etype} \
        --elayers ${elayers} \
        --eunits ${eunits} \
        --eprojs ${eprojs} \
        --subsample ${subsample} \
        --dlayers ${dlayers} \
        --dunits ${dunits} \
        --atype ${atype} \
        --adim ${adim} \
        --awin ${awin} \
        --aheads ${aheads} \
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
        --lsm-type ${lsm_type} \
        --lsm-weight ${lsm_weight} \
        --batch-size ${batchsize} \
        --maxlen-in ${maxlen_in} \
        --maxlen-out ${maxlen_out} \
        --opt ${opt} \
        --epochs ${epochs} \
        --lr ${lr} \
        --lr_decay ${lr_decay} \
        --mom ${mom} \
        --wd ${wd} \
        --num-enc ${num_enc} \
        --share-ctc ${share_ctc} \
        --l2-dropout ${l2_dropout}
fi

if [ ${stage} -le 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}

        if [ $use_lm = true ]; then
            decode_dir=${decode_dir}_rnnlm${lm_weight}_${lmtag}
            if [ $use_wordlm = true ]; then
                recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
            else
                recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
            fi
        else
            echo "No language model is involved."
            recog_opts=""
        fi

        if [ $addgauss = true ]; then
            decode_dir=${decode_dir}_gauss-${addgauss_type}-mean${addgauss_mean}-std${addgauss_std}
            recog_opts+=" --addgauss true --addgauss-mean ${addgauss_mean} --addgauss-std ${addgauss_std} --addgauss-type ${addgauss_type}"
        fi

        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --lm-weight ${lm_weight} \
            $recog_opts &
        wait

        score_sclite.sh --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

