dset_name=tvsum
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results
exp_id=exp


######## data paths
train_path=data/tvsum/tvsum_train.jsonl
eval_path=data/tvsum/tvsum_val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=/mnt/sihui/datasets/tvsum

# # video features
v_feat_dim=2048
v_feat_dirs=()
v_feat_dirs+=(${feat_root}/video_features)

# # text features
t_feat_dir=${feat_root}/query_features/ # maybe not used
t_feat_dim=512

#### training
bsz=4
lr=1e-3

######## 修改特征破坏参数 ########
apply_to_modality=video         # 改为处理文本模态
corruption_type=mask    # 使用token遮蔽
corruption_ratio=0.3          # 增强破坏比例
# 新增自编码器开关
use_ae=true   
######## TVSUM domain name
for dset_domain in BK BT DS FM GA MS PK PR VT VU
do
    for seed in 0 1 2 3 2017
    do
        PYTHONPATH=$PYTHONPATH:. python qd_detr/train.py \
        --dset_name ${dset_name} \
        --ctx_mode ${ctx_mode} \
        --train_path ${train_path} \
        --eval_path ${eval_path} \
        --eval_split_name ${eval_split_name} \
        --v_feat_dirs ${v_feat_dirs[@]} \
        --v_feat_dim ${v_feat_dim} \
        --t_feat_dir ${t_feat_dir} \
        --t_feat_dim ${t_feat_dim} \
        --bsz ${bsz} \
        --results_root ${results_root}_${dset_domain} \
        --exp_id ${exp_id} \
        --max_v_l 1000 \
        --n_epoch 2000 \
        --lr_drop 2000 \
        --max_es_cnt -1 \
        --seed $seed \
        --lr ${lr} \
        --dset_domain ${dset_domain} \
        --apply_to_modality ${apply_to_modality} \
        --corruption_type ${corruption_type} \
        --corruption_ratio ${corruption_ratio} \
        $(if [ "$use_ae" = "true" ]; then echo "--use_ae"; fi) \
        ${@:1}
    done
done
