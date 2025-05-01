from qd_detr.utils.feature_utils import apply_corruption  # 新增导入
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from qd_detr.span_utils import span_xx_to_cxw

logger = logging.getLogger(__name__)


class StartEndDataset_audio(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir, a_feat_dir=None,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 dset_domain=None,corruption_type="none", 
                 corruption_ratio=0.0, apply_to_modality="video",
                 use_ae=False):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.a_feat_dir = a_feat_dir

        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()

        # load specific domain data for tvsum dataset
        if self.dset_name == 'tvsum':
            target_domain = dset_domain
            assert target_domain in ["BK", "BT", "DS", "FM", "GA", "MS", "PK", "PR", "VT", "VU"]

            new_data = []
            for d in self.data:
                if target_domain == d['domain']:
                    new_data.append(d)
            self.data = new_data
        self.is_test = "val" in data_path.lower() or "test" in data_path.lower()
        print(f"[DEBUG] 当前数据集模式 | {'测试集' if self.is_test else '训练集'}")
        # 特征破坏配置
        self.corruption_type = corruption_type
        self.corruption_ratio = corruption_ratio
        self.apply_to_modality = apply_to_modality.lower()
        print(f"[DEBUG] 当前设备状态 | 主设备: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
        print(f"[DEBUG] 参数生效检查 | ctx_mode: {self.ctx_mode} | apply_to_modality: {self.apply_to_modality}")
        self.use_ae = use_ae  # 新增自编码器开关
        # 新增统一设备管理
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            
    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()
        query_feat, orig_query_feat = self._get_query_feat_by_qid(meta["qid"]) 
        model_inputs["query_feat"] = query_feat
        model_inputs["orig_query_feat"] = orig_query_feat
        if self.use_video:
            video_feat, orig_video_feat = self._get_video_feat_by_vid(meta["vid"])
            model_inputs["video_feat"] = video_feat        # 破坏后的特征
            model_inputs["orig_video_feat"] = orig_video_feat  # 原始特征
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l
        if self.a_feat_dir is not None:
            audio_feat, orig_audio_feat = self._get_audio_feat_by_vid(meta["vid"])  # 修改此行
            model_inputs["audio_feat"] = audio_feat
            model_inputs["orig_audio_feat"] = orig_audio_feat  # 新增原始特征存储
            ctx_l_a = len(model_inputs["audio_feat"])
            
            if ctx_l_a < ctx_l:
                ctx_l = ctx_l_a
            model_inputs["video_feat"] = model_inputs["video_feat"][:ctx_l]
            model_inputs["audio_feat"] = model_inputs["audio_feat"][:ctx_l]

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            # print(tef.shape, model_inputs['video_feat'].shape, model_inputs['audio_feat'].shape)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef
            if self.a_feat_dir is not None:
                model_inputs["audio_feat"] = torch.cat(
                    [model_inputs["audio_feat"], tef], dim=1)  # (Lv, Dv+2)

        if len(model_inputs["query_feat"].shape) == 3:
            # There is batch dimension, which I should have removed at the feature extraction time, but didn't.
            # Remove it online
            model_inputs["query_feat"] = model_inputs["query_feat"][0]
            
        if self.load_labels:
            if self.dset_name == 'tvsum':
                model_inputs["span_labels"] = torch.tensor([[0., 0.]])
                meta_label = meta['label']

                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                            self.get_saliency_labels_all_tvsum(meta_label, ctx_l)

            else:
                model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)  # (#windows, 2)
                if "subs_train" not in self.data_path:
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_all(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
                else:
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], ctx_l)  # only one gt
        return dict(meta=meta, model_inputs=model_inputs)

    def get_saliency_labels_sub_as_query(self, gt_window, ctx_l, max_n=2):
        gt_st = int(gt_window[0] / self.clip_len)
        gt_ed = max(0, min(int(gt_window[1] / self.clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed+1), k=max_n)
        else:
            pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))
        neg_clip_indices = random.sample(neg_pool, k=max_n)
        # return pos_clip_indices, neg_clip_indices
        
        score_array = np.zeros(ctx_l)
        score_array[gt_st:gt_ed+1] = 1

        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels_all(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # score_array = [min(agg_scores[idx], ctx_l-1) for idx in range(ctx_l)]
        score_array = np.zeros(ctx_l)
        for idx in range(len(rel_clip_ids)):
            if rel_clip_ids[idx] >= ctx_l:
                score_array_new = np.zeros(ctx_l + 1)
                score_array_new[:ctx_l] = score_array
                score_array = score_array_new
            # if rel_clip_ids[idx] == ctx_l:
            #     print(rel_clip_ids[idx], ctx_l)
            score_array[rel_clip_ids[idx]] = agg_scores[idx]

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices, score_array


    def get_saliency_labels_all_tvsum(self, labels, ctx_l, max_n=1, add_easy_negative=False):

        agg_scores = np.sum(labels - np.ones_like(labels), axis=-1)[:ctx_l] # start from 1, so minus 1
        score_array = agg_scores / 80 * 12
        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array
    
    
    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows


    def _get_query_feat_by_qid(self, qid):
        if self.dset_name == 'tvsum':
            q_feat = np.load(join(self.q_feat_dir, "{}.npz".format(qid))) # 'token', 'text'
            token_array = q_feat['token']  # 直接获取numpy数组
            orig_query_feat = token_array.copy()  # 复制原始特征            
            q_feat_tensor = torch.from_numpy(token_array)
            # 应用文本特征破坏
            if self.apply_to_modality == "text" :
                q_feat_tensor = q_feat_tensor.to(self.device)
                print(f"[DEBUG] 文本特征破坏前")
                q_feat_tensor = apply_corruption(
                    q_feat_tensor,
                    corruption_type=self.corruption_type,
                    corruption_ratio=self.corruption_ratio
                )
                print(f"[DEBUG] 文本特征破坏了")
                q_feat_tensor = q_feat_tensor.cpu()
            return q_feat_tensor, torch.from_numpy(orig_query_feat)
        else:
            # QVhighlight dataset
            q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
            q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]
            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)
            if self.txt_drop_ratio > 0:
                q_feat = self.random_drop_rows(q_feat)
            if self.apply_to_modality == "text":
                q_feat_tensor = torch.from_numpy(q_feat)
                q_feat_tensor = apply_corruption(
                    q_feat_tensor,
                    corruption_type=self.corruption_type,
                    corruption_ratio=self.corruption_ratio
                )
                q_feat = q_feat_tensor.numpy()
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)


    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings


    def _get_video_feat_by_vid(self, vid):
        if self.dset_name == 'tvsum':
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                _feat_path = join(_feat_dir, f"{vid}_rgb.npy")
                _feat_rgb = np.load(_feat_path)[:self.max_v_l].astype(np.float32)

                _feat_path = join(_feat_dir, f"{vid}_opt.npy")
                _feat_opt = np.load(_feat_path)[:self.max_v_l].astype(np.float32)
                
                _feat = np.concatenate([_feat_rgb, _feat_opt], axis=-1)
                # _feat = _feat_rgb
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)
            # ========= 新增特征破坏 =========
            orig_video_feat = v_feat.copy()
            if self.apply_to_modality == "video" :
                print(f"[DEBUG] 视频特征统计 | 均值: {v_feat.mean():.4f} 方差: {v_feat.var():.4f} 零值比例: {(v_feat == 0).mean():.2%}")
                v_feat_tensor = torch.from_numpy(v_feat)
                v_feat_tensor = apply_corruption(
                    v_feat_tensor,
                    corruption_type=self.corruption_type,
                    corruption_ratio=self.corruption_ratio
                )
                v_feat = v_feat_tensor.cpu().numpy()
                print(f"[DEBUG] 破坏后统计 | 均值: {v_feat.mean():.4f} 方差: {v_feat.var():.4f} 零值比例: {(v_feat == 0).mean():.2%}")

        else:
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                _feat_path = join(_feat_dir, f"{vid}.npz")
                _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)
            # ========= 新增特征破坏逻辑 =========
            orig_video_feat = v_feat.copy()
            if self.apply_to_modality == "video":
                v_feat_tensor = torch.from_numpy(v_feat)
                v_feat_tensor = apply_corruption(
                    v_feat_tensor,
                    corruption_type=self.corruption_type,
                    corruption_ratio=self.corruption_ratio
                )
                v_feat = v_feat_tensor.cpu().numpy()  # 显式转到CPU再转换
        
        return torch.from_numpy(v_feat), torch.from_numpy(orig_video_feat)

    # def _get_audio_feat_by_vid(self, vid):
    #     a_feat_path = join(self.a_feat_dir, f"{vid}.npy")
    #     a_feat = np.load(a_feat_path)[:self.max_v_l].astype(np.float32)
    #     if self.normalize_v:
    #         a_feat = l2_normalize_np_array(a_feat)
    #
    #     return torch.from_numpy(a_feat)  # (D, ) or (Lq, D)
    def _get_audio_feat_by_vid(self, vid):
        a_feat_path = join(self.a_feat_dir, f"{vid}.npy")
        try:
            a_feat = np.load(a_feat_path).astype(np.float32)

            # === 动态修复逻辑 ===
            # 处理一维数组（假设原意是 (N, 2048) 但被错误保存为一维）
            if a_feat.ndim == 1:
                expected_elements = 75 * 2048
                if a_feat.size < expected_elements:
                    # 填充零到预期长度
                    a_feat = np.pad(a_feat, (0, expected_elements - a_feat.size), mode='constant')
                else:
                    # 截断到预期长度
                    a_feat = a_feat[:expected_elements]
                a_feat = a_feat.reshape(75, 2048)
            # 处理二维数组但长度不足
            elif a_feat.ndim == 2:
                if a_feat.shape[0] < 75 or a_feat.shape[1] != 2048:
                    # 填充或截断到 (75, 2048)
                    padded_feat = np.zeros((75, 2048), dtype=np.float32)
                    min_len = min(a_feat.shape[0], 75)
                    padded_feat[:min_len] = a_feat[:min_len]
                    a_feat = padded_feat
            else:
                raise ValueError(f"Unsupported audio feature dimension: {a_feat.ndim}")

            # 确保截断到 max_v_l (兼容原代码逻辑)
            a_feat = a_feat[:self.max_v_l]

            if self.normalize_v:
                a_feat = l2_normalize_np_array(a_feat)
            # ====== 新增特征破坏 ======
            orig_audio_feat = a_feat.copy()  # 新增原始特征保存
            if self.apply_to_modality == "audio":
                print(f"[DEBUG] 音频特征")
                a_feat_tensor = torch.from_numpy(a_feat)
                a_feat_tensor = apply_corruption(
                    a_feat_tensor,
                    corruption_type=self.corruption_type,
                    corruption_ratio=self.corruption_ratio
                )
                a_feat = a_feat_tensor.cpu().numpy()
            return torch.from_numpy(a_feat), torch.from_numpy(orig_audio_feat)  
        except Exception as e:
            print(f"❌ 加载音频特征失败: {a_feat_path}, 错误: {str(e)}，使用零矩阵替代")
            a_feat = np.zeros((self.max_v_l, 2048), dtype=np.float32)

        return torch.from_numpy(a_feat)


def start_end_collate_audio(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k in ["orig_video_feat", "orig_query_feat", "orig_audio_feat"]:

            batched_data[k] = pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], 
                dtype=torch.float32,  
                fixed_length=None
            )
            continue
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
            # print(pad_data, mask_data)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue

        batched_data[k] = pad_sequences_1d(
            [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data


def prepare_batch_inputs_audio(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
        src_aud=batched_model_inputs["audio_feat"][0].to(device, non_blocking=non_blocking),
        src_aud_mask=batched_model_inputs["audio_feat"][1].to(device, non_blocking=non_blocking),
        # 传递原始特征
        orig_video_feat=batched_model_inputs["orig_video_feat"][0].to(device, non_blocking=non_blocking),
        orig_query_feat=batched_model_inputs["orig_query_feat"][0].to(device, non_blocking=non_blocking),
        orig_audio_feat=batched_model_inputs["orig_audio_feat"][0].to(device, non_blocking=non_blocking)
    )
    # 添加音频特征的条件检查
    if "audio_feat" in batched_model_inputs:  # 新增条件判断
        model_inputs.update({
            "src_aud": batched_model_inputs["audio_feat"][0].to(device, non_blocking=non_blocking),
            "src_aud_mask": batched_model_inputs["audio_feat"][1].to(device, non_blocking=non_blocking)
        })
    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)

    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
