ckpt_path=$1
eval_split_name=$2
a_feat_type=pann
a_feat_dim=2050
feat_root=../features
a_feat_dir=${feat_root}/umt_pann_features/
eval_path=data/highlight_${eval_split_name}_release.jsonl
# 新增特征破坏参数（位置参数$3-$5）
apply_to_modality=${3:-"audio"}  # 默认处理音频
corruption_type=${4:-"none"}     # 默认无破坏
corruption_ratio=${5:-0.0}       # 默认破坏比例0%
PYTHONPATH=$PYTHONPATH:. python qd_detr/inference.py \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
--a_feat_dir ${a_feat_dir} \
--a_feat_dim ${a_feat_dim} \
--apply_to_modality ${apply_to_modality} \
--corruption_type ${corruption_type} \
--corruption_ratio ${corruption_ratio} \
${@:6}  # 调整参数索引
