#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
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
etype=vggblstmp     # encoder architecture type
elayers=6
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
mtlalpha=0.2

# label smoothing
lsm_type=unigram
lsm_weight=0.05

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
use_lm=true

# decoding parameter
lm_weight=1.0
beam_size=30
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# data
mic="Beam_Circular_Array Beam_Linear_Array" # must be to elements
wsj0_folder=/export/b18/xwang/data/Data_processed

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

if [ ${stage} -le 0 ]; then
    echo "stage 0: Data preparation"
    for mic_sel in $mic; do
	wsj0_contaminated_folder=WSJ_contaminated_mic_$mic_sel # path of the training data
	DIRHA_wsj_data=/export/b18/xwang/data/Data_processed/DIRHA_wsj_oracle_VAD_mic_$mic_sel # path of the test data
    	
	local/wsj0_data_prep.sh $wsj0_folder $wsj0_contaminated_folder $mic_sel || exit 1;
	local/dirha_data_prep.sh $DIRHA_wsj_data/Sim dirha_sim_$mic_sel  || exit 1;
	local/dirha_data_prep.sh $DIRHA_wsj_data/Real dirha_real_$mic_sel  || exit 1;

	local/format_data.sh $mic_sel || exit 1;
    done
    
fi

if [ ${stage} -le 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame

    for mic_sel in $mic; do
        for x in tr05_cont_$mic_sel dirha_sim_$mic_sel dirha_real_$mic_sel; do
	    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
	        data/${x} exp/make_fbank/${x} ${fbankdir}
        done
	# compute global CMVN
	compute-cmvn-stats scp:data/tr05_cont_$mic_sel/feats.scp data/tr05_cont_$mic_sel/cmvn.ark

   	train_set=tr05_cont_$mic_sel 
   	train_dev=dirha_sim_$mic_sel 
   	train_test=dirha_real_$mic_sel
   	recog_set="dirha_sim_$mic_sel dirha_real_$mic_sel"

	feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
	feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

    	# dump features for training
    	if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
	    utils/create_split_dir.pl \
		    /export/b{10,11,12,13}/${USER}/espnet-data/egs/dirha_wsj/asr2_multi/dump/${train_set}/delta${do_delta}/storage \
		    ${feat_tr_dir}/storage
    	fi
    	if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
	    utils/create_split_dir.pl \
		    /export/b{10,11,12,13}/${USER}/espnet-data/egs/dirha_wsj/asr2_multi/dump/${train_dev}/delta${do_delta}/storage \
		    ${feat_dt_dir}/storage
    	fi
    	dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
	    data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    	dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
	    data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    	for rtask in ${recog_set}; do
	    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
	    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
		    data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
		    ${feat_recog_dir}
    	done

    done
fi

if [ ${stage} -le 1 ]; then
    echo "stage 1: Data concatenating ..."
    mkdir -p data/dirha_multistream/train_orig
    cp data/tr05_cont_Beam_Circular_Array/* data/dirha_multistream/train_orig
    rm -r data/dirha_multistream/train_orig/feats.scp
    rm -r data/dirha_multistream/train_orig/cmvn.ark
    utils/data/copy_data_dir.sh data/dirha_multistream/train_orig data/dirha_multistream_train
    paste-feats scp:data/tr05_cont_Beam_Circular_Array/feats.scp scp:data/tr05_cont_Beam_Linear_Array/feats.scp ark,scp:fbank/raw_fbank_pitch_dirha_multistream_train.ark,data/dirha_multistream_train/feats.scp
    compute-cmvn-stats scp:data/dirha_multistream_train/feats.scp data/dirha_multistream_train/cmvn.ark

    mkdir -p data/dirha_multistream/dev_orig
    cp data/dirha_sim_Beam_Circular_Array/* data/dirha_multistream/dev_orig
    rm -r data/dirha_multistream/dev_orig/feats.scp
    utils/data/copy_data_dir.sh data/dirha_multistream/dev_orig data/dirha_multistream_dev
    paste-feats scp:data/dirha_sim_Beam_Circular_Array/feats.scp scp:data/dirha_sim_Beam_Linear_Array/feats.scp ark,scp:fbank/raw_fbank_pitch_dirha_multistream_dev.ark,data/dirha_multistream_dev/feats.scp
    compute-cmvn-stats scp:data/dirha_multistream_dev/feats.scp data/dirha_multistream_dev/cmvn.ark

    mkdir -p data/dirha_multistream/test_orig
    cp data/dirha_real_Beam_Circular_Array/* data/dirha_multistream/test_orig
    rm -r data/dirha_multistream/test_orig/feats.scp
    utils/data/copy_data_dir.sh data/dirha_multistream/test_orig data/dirha_multistream_test
    paste-feats scp:data/dirha_real_Beam_Circular_Array/feats.scp scp:data/dirha_real_Beam_Linear_Array/feats.scp ark,scp:fbank/raw_fbank_pitch_dirha_multistream_test.ark,data/dirha_multistream_test/feats.scp
    compute-cmvn-stats scp:data/dirha_multistream_test/feats.scp data/dirha_multistream_test/cmvn.ark

fi

train_set=dirha_multistream_train
train_dev=dirha_multistream_dev
train_test=dirha_multistream_test
recog_set="dirha_multistream_dev dirha_multistream_test"
feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
    	
if [ ${stage} -le 1 ]; then
    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
	utils/create_split_dir.pl \
    		/export/b{10,11,12,13}/${USER}/espnet-data/egs/dirha_wsj/asr2_multi/dump/${train_set}/delta${do_delta}/storage \
		${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
        utils/create_split_dir.pl \
		/export/b{10,11,12,13}/${USER}/espnet-data/egs/dirha_wsj/asr2_multi/dump/${train_dev}/delta${do_delta}/storage \
		${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
	    data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
	    data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
	dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
		data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
		${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
	    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
	    data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
	    data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
	    feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
	    data2json.sh --feat ${feat_recog_dir}/feats.scp \
		    --nlsyms ${nlsyms} data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

# It takes about one day. If you just want to do end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=${lm_layers}layer_unit${lm_units}_${lm_opt}_bs${lm_batchsize}
    if [ $use_wordlm = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
    fi
fi
lmexpdir=exp/train_rnnlm_${backend}_${lmtag}
mkdir -p ${lmexpdir}

if [[ ${stage} -le 3 && $use_lm == true ]]; then
     echo "stage 3: LM Preparation"	    
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


