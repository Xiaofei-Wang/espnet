#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This is a baseline for "JSALT'18 Multilingual End-to-end ASR for Incomplete Data"
# We use 5 Babel language (Assamese Tagalog Swahili Lao Zulu), Librispeech (English), and CSJ (Japanese)
# as a target language, and use 10 Babel language (Cantonese Bengali Pashto Turkish Vietnamese
# Haitian Tamil Kurmanji Tok-Pisin Georgian) as a non-target language.
# The recipe first build language-independent ASR by using non-target languages

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
gpu=            # will be deprecated, please use ngpu
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=blstmp     # encoder architecture type
elayers=4
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aconv_chans=10
aconv_filts=100
adim=320

# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=50
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=20

# sgd parameters
lr=1e-3
lr_decay=1e-1
mom=0.9
wd=0

# decoding parameter
beam_size=20
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'

# exp tag
tag="" # tag for managing experiments.

# data set
# non-target languages: cantonese bengali pashto turkish vietnamese haitian tamil kurmanji tokpisin georgian
train_set=tr_babel10
train_dev=dt_babel10

recog_set="dt_babel_vietnamese"

# languages subset option
lang_list="vietnamese"

# multl-encoder multi-band
num_enc=1
share_ctc=true

# for decoding only ; only works for multi case
l2_weight=0.5


. utils/parse_options.sh || exit 1;

# data set
[ ! -d data/$train_set ] && echo "Need to generate data/$train_set first!" && exit 1
[ ! -d data/$train_dev ] && echo "Need to generate data/$train_dev first!" && exit 1
num_lang=$(echo $lang_list | awk '{print NF}')
lang=$(echo $lang_list| sed "s@ @-@g")


# data directories
csjdir=../../csj
libridir=../../librispeech
babeldir=../../babel

. ./path.sh
. ./cmd.sh

# check gpu option usage
if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


feat_tr_dir=${dumpdir}/${train_set}_${lang}_${train_set}_${lang}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}_${lang}_${train_set}_${lang}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ]; then

    for l in $lang_list; do
        grep $l data/${train_set}/spk2utt | awk '{print $1}' > data/${train_set}/spk_list_${l}
        grep $l data/${train_dev}/spk2utt | awk '{print $1}' > data/${train_dev}/spk_list_${l}
        utils/subset_data_dir.sh --spk-list data/${train_set}/spk_list_${l} data/${train_set} data/${train_set}_${l}
        utils/subset_data_dir.sh --spk-list data/${train_dev}/spk_list_${l} data/${train_dev} data/${train_dev}_${l}
    done


    if [ $num_lang -gt 1 ]; then
        data_list=$(echo $lang_list | awk -v dir=$train_set '{for(i=1;i<=NF;i++){ printf "data/%s_%s ",dir,$i} }')
        utils/combine_data.sh data/${train_set}_${lang}_orig $data_list
        data_list=$(echo $lang_list | awk -v dir=$train_dev '{for(i=1;i<=NF;i++){ printf "data/%s_%s ",dir,$i} }')
        utils/combine_data.sh data/${train_dev}_${lang}_orig $data_list
    else
        utils/copy_data_dir.sh data/${train_set}_${lang} data/${train_set}_${lang}_orig
        utils/copy_data_dir.sh data/${train_dev}_${lang} data/${train_dev}_${lang}_orig

    fi

    # remove utt having more than 3000 frames or less than 10 frames or
    # remove utt having more than 400 characters or no more than 0 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_${lang}_orig data/${train_set}_${lang}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_${lang}_orig data/${train_dev}_${lang}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}_${lang}/feats.scp data/${train_set}_${lang}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{01,02,03,04}/${USER}/espnet-data/egs/jsalt18e2e/asr1/dump/${train_set}_${lang}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{01,02,03,04}/${USER}/espnet-data/egs/jsalt18e2e/asr1/dump/${train_dev}_${lang}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    [ ! -d ${feat_tr_dir}/feats.scp ] && dump.sh --cmd "$train_cmd" --nj 40 --do_delta $do_delta \
        data/${train_set}_${lang}/feats.scp data/${train_set}_${lang}/cmvn.ark exp/dump_feats/${train_set}_${lang}_${train_set}_${lang} ${feat_tr_dir}
    [ ! -d ${feat_dt_dir}/feats.scp ] && dump.sh --cmd "$train_cmd" --nj 40 --do_delta $do_delta \
        data/${train_dev}_${lang}/feats.scp data/${train_set}_${lang}/cmvn.ark exp/dump_feats/${train_dev}_${lang}_${train_set}_${lang} ${feat_dt_dir}
   for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}_${train_set}_${lang}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        [ ! -d ${feat_recog_dir}/feats.scp ] && dump.sh --cmd "$train_cmd" --nj 40 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}_${lang}/cmvn.ark exp/dump_feats/recog/${rtask}_${train_set}_${lang} \
            ${feat_recog_dir}
    done
fi
train_set=${train_set}_${lang}
train_dev=${train_dev}_${lang}
dict=data/lang_1char/train_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list for all languages"
    cut -f 2- data/tr_*/text | grep -o -P '\[.*?\]|\<.*?\>' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cat data/tr_*/text | text2token.py -s 1 -n 1 -l ${nlsyms} | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}_${train_set}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

if [ -z ${tag} ]; then

    if [[ $opt == "sgd" ]]; then
        expdir=exp_lang${num_lang}/${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}-${lr}-${lr_decay}-${mom}-${wd}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_shareCtc${share_ctc}
    else
        expdir=exp_lang${num_lang}/${train_set}_${etype}_e${elayers}_subsample${subsample}_unit${eunits}_proj${eprojs}_d${dlayers}_unit${dunits}_${atype}_aconvc${aconv_chans}_aconvf${aconv_filts}_mtlalpha${mtlalpha}_${opt}_bs${batchsize}_mli${maxlen_in}_mlo${maxlen_out}_shareCtc${share_ctc}
    fi
    if [ $adim != 320 ];then
        expdir=${expdir}_adim${adim}
    fi
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp_lang${num_lang}/${train_set}_${tag}
fi
mkdir -p ${expdir}

if [ ${stage} -le 3 ]; then
    echo "stage 3: Network Training"
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
        --aconv-chans ${aconv_chans} \
        --aconv-filts ${aconv_filts} \
        --mtlalpha ${mtlalpha} \
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
        --adim ${adim}
fi


if [ ${stage} -le 4 ]; then
    echo "stage 4: Decoding"
    nj=32

    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_beam${beam_size}_e${recog_model}_p${penalty}_len${minlenratio}-${maxlenratio}_ctcw${ctc_weight}_l2w${l2_weight}
        feat_recog_dir=${dumpdir}/${rtask}_${train_set}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/model.${recog_model}  \
            --model-conf ${expdir}/results/model.conf  \
            --beam-size ${beam_size} \
            --penalty ${penalty} \
            --maxlenratio ${maxlenratio} \
            --minlenratio ${minlenratio} \
            --ctc-weight ${ctc_weight} \
            --l2-weight ${l2_weight} \
            &
        wait

        score_sclite.sh --nlsyms ${nlsyms} --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    done
    wait
    echo "Finished"
fi

