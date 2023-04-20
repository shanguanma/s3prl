#!/usr/bin/env bash

stage=-1
stop_stage=1000
. utils/parse_options.sh
. path_for_fairseq_speechtext.sh

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
    pretrain_model=exp/pretrain/pretrain_on_hubert_4gpu_8update_960h_mfcc_250k_update/checkpoint_289_400000.pt
    exp_dir=exp/downstream_task/downstream_task_ic
    python   s3prl/s3prl/upstream/hubert/convert.py\
               --fairseq_ckpt   $pretrain_model\
               --output_dir $exp_dir
                
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   exp_dir=exp/downstream_task/downstream_task_ic ## output 
   pretrain_model=$exp_dir/checkpoint_289_400000.pt
   num_gpus=4
   downstream_task_config=s3prl/s3prl/downstream/fluent_commands/config.yaml
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use_env --nproc_per_node $num_gpus s3prl/s3prl/run_downstream.py\
           -m train \
           -u hubert_local\
           -k $pretrain_model\
           -d fluent_commands\
           --config $downstream_task_config\
           -o config.runner.gradient_accumulate_steps=1\
           --auto_resume \
           --expdir $exp_dir
fi
