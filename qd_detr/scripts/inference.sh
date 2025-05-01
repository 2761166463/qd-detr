ckpt_path=$1
eval_split_name=$2
eval_path=data/highlight_${eval_split_name}_release.jsonl
echo ${ckpt_path}
echo ${eval_split_name}
echo ${eval_path}
# 新增特征破坏参数
apply_to_modality=${3:-"video"}  # 默认处理视频模态
corruption_type=${4:-"none"}     # 默认无破坏
corruption_ratio=${5:-0.0}       # 默认破坏比例0%

PYTHONPATH=$PYTHONPATH:. python qd_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
--apply_to_modality ${apply_to_modality} \
--corruption_type ${corruption_type} \
--corruption_ratio ${corruption_ratio} \
--use_ae ${use_ae} \ 
${@:7}
