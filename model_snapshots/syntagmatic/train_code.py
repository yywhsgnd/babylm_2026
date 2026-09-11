#!/usr/bin/env python3
"""
train_clean_mlm_rae_v7_projprogress.py -- progress-gated projected Syn+Para RAE
============================================================================
This is an isolated clean-MLM version of v7. It keeps the v7
syntagmatic/paradigmatic losses, but removes AMLM/adaptive masking when
--regular_mlm is set, so RAE can be compared against a matched clean MLM.

  理论                          实现
  ───────────────────────────  ────────────────────────────
  组合轴: “和谁组合成结构”     注意力引导的 Syn target
                               → 对每个 [MASK]，取 last-layer
                                 attention 的 top-k 上下文 token
                                 → syn_target = 被关注 token 的 embedding 加权和
  ───────────────────────────  ────────────────────────────
  聚合轴: “能替换成什么”       embedding 空间近邻 Para target
                               → 对每个 [MASK]，在当前 embedding
                                 空间中找 gold token 的 k 近邻
                                 → para_target = 近邻 embedding 的均值
  ───────────────────────────  ────────────────────────────
  MLM 选槽位                   仅对 masked 位置施加 RAE 约束
  ───────────────────────────  ────────────────────────────
  实体 token 保护              五类 token 差异化 Para 权重
                               entity/reading → zero para

与 v2 的关键区别:
  v2 Syn  = 窗口配对（局部共现）    v7 Syn  = 注意力引导（结构依赖）
  v2 Para = 模型预测（自我指涉）    v7 Para = embedding 近邻（分布语义）
  v2 位置 = 所有内容词             v7 位置 = 仅 masked 位置
  v2 损失 = InfoNCE                 v7 损失 = 1 - cosine_similarity

Key controls:
  - --regular_mlm disables adaptive mask-weight updates.
  - Token-type gating uses clean gold/original ids, not corrupted inputs.
  - Syn/Para targets are detached, so auxiliary losses train hidden states
    without moving the target embedding space directly.
  - This progress variant uses training progress gates, not hard-coded step gates.
  - Syntax-token Para and content-token Para have independent progress gates.
  - Official-safe Syn can leave entity/reading tokens to the clean MLM objective.
  - Content Para is routed through a residual projection head while Syn and
    syntax Para stay directly on raw hidden states.
"""

import argparse, os, sys, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)

from transformers import (set_seed, AutoConfig, AutoModelForMaskedLM,
    DebertaV2Tokenizer, AutoTokenizer, PreTrainedTokenizerFast)
from transformers.optimization import get_cosine_schedule_with_warmup
from datasets import load_dataset
from preprocessing import tokenize, padding_collate_fn, group_texts

try: from bitsandbytes.optim import LAMB; LAMB_OK = True
except ImportError: LAMB_OK = False
try: import wandb; WB_OK = True
except ImportError: WB_OK = False

# ======================== Token 分类 ========================
# 复用 v6 的五类 token 体系（词表派生，不增加数据曝光）
SYNTAX_SETS = {
    "aux":  {"is","are","was","were","be","been","being","am",
             "do","does","did","have","has","had",
             "can","could","will","would","should","may","might","must",
             "shall","ought","need","dare","used"},
    "det":  {"a","an","the","this","that","these","those",
             "some","any","all","each","every","many","few","no",
             "much","more","most","several","both","either","neither","such","what"},
    "pron": {"he","she","it","they","him","her","them","his","their","its",
             "we","us","our","you","your","i","me","my","myself","yourself",
             "himself","herself","itself","ourselves","themselves","one","ones"},
    "neg":  {"not","n't","never","no","nor","neither"},
    "prep": {"in","on","at","by","with","from","to","of","for",
             "into","onto","over","under","near","about","between","through",
             "during","without","within","along","across","behind","beyond",
             "toward","towards","upon","among","amongst","beside","besides",
             "against","around","before","after","above","below","off","up","down","out"},
}
READING_SETS = {
    "conj": {"and","or","but","because","although","if","when","while",
             "before","after","since","until","unless","whereas","so","yet",
             "than","as","though","whether","once","till","lest","except",
             "provided","given","suppose","assuming","whenever","wherever"},
    "wh":   {"who","whom","whose","which","where","why","how",
             "what","whatever","whichever","whoever","however","wherever"},
    "comp": {"more","less","fewer","better","worse","bigger","smaller",
             "higher","lower","longer","shorter","older","younger"},
    "temp": {"now","then","ago","later","earlier","soon","already","still",
             "yet","finally","eventually","previously","formerly","currently",
             "recently","lately","immediately","suddenly","gradually"},
}


def _clean_token(s: str) -> str:
    for prefix in ("▁", "Ġ", "##", " "):
        if s.startswith(prefix): s = s[len(prefix):]
    return s.strip().rstrip(".,;:!?()[]{}\"'`-_=+/\\|@#$%^&*~")


def _has_digit(s: str) -> bool:
    return any(c.isdigit() for c in _clean_token(s))


def _is_byte_token(s: str) -> bool:
    import re
    return bool(re.match(r'^<0x[0-9a-fA-F]{2}>$', s.strip()))


def build_token_categories(tokenizer):
    vs = tokenizer.vocab_size
    is_syntax  = torch.zeros(vs, dtype=torch.bool)
    is_reading = torch.zeros(vs, dtype=torch.bool)
    is_entity  = torch.zeros(vs, dtype=torch.bool)
    is_content = torch.zeros(vs, dtype=torch.bool)
    is_punct   = torch.zeros(vs, dtype=torch.bool)

    all_syntax = set(); all_reading = set()
    for cat in SYNTAX_SETS.values(): all_syntax |= cat
    for cat in READING_SETS.values(): all_reading |= cat

    for token_str, token_id in tokenizer.get_vocab().items():
        w = _clean_token(token_str).lower()
        orig = _clean_token(token_str)

        if _is_byte_token(token_str):
            is_punct[token_id] = True; continue
        if not any(c.isalpha() for c in orig):
            is_punct[token_id] = True; continue
        if w in all_syntax:
            is_syntax[token_id] = True; continue
        if w in all_reading:
            is_reading[token_id] = True; continue
        is_numeric = _has_digit(token_str)
        is_proper  = (len(orig) >= 3 and orig[0].isupper() and orig[0].isalpha()
                      and all(c.isalpha() for c in orig))
        if is_numeric or is_proper:
            is_entity[token_id] = True; continue
        is_content[token_id] = True

    return is_syntax, is_reading, is_entity, is_content, is_punct


# ======================== Args ========================
parser = argparse.ArgumentParser(description="Clean MLM control for v7 syntagmatic/paradigmatic RAE")
for a in [
    ("--train_data",str,""),("--valid_data",str,"data/even.dev"),("--max_seq_len",str,"64"),
    ("--model_path",str,"microsoft/deberta-v3-base"),("--output_path",str,""),
    ("--tokenizer",str,None),("--batch_size",int,256),("--grad_acc",int,1),
    ("--lr",float,0.007),("--epochs",int,10),("--cpus",int,64),
    ("--logging_steps",int,100),("--eval_steps",int,1000),("--save_steps",int,1000),
    ("--all_checkpoints",bool,False),("--mask_update_steps",int,100),
    ("--hidden_size",int,768),("--intermediate_size",int,3072),("--dropout",float,0.1),
    ("--weight_decay",float,0.01),("--mlm_prob",float,0.15),
    ("--mask_replace_prob",float,0.8),("--random_replace_prob",float,0.1),
    ("--seed",int,0),("--pretrained",bool,False),("--debug",bool,False),
    ("--wandb",bool,False),("--wandb_project",str,"babylm2026-amlm"),
    ("--wandb_name",str,""),("--wandb_tags",str,""),
    ("--regular_mlm",bool,False),("--lamb",bool,False),("--lower",bool,False),
    ("--mask_decay",float,0.0),

    # ── 组合轴 Syn（注意力引导）──
    ("--rae_syn",bool,False,"Enable attention-guided syn loss"),
    ("--rae_syn_weight",float,0.0005,"Syn loss scale"),
    ("--rae_syn_topk",int,8,"Number of attended context tokens as syn target"),
    ("--rae_syn_layer",int,-1,"Which layer's attention (-1=last)"),
    ("--rae_syn_warmup_steps",int,1000,"Legacy Syn warmup step; used only if progress is not set"),
    ("--rae_syn_warmup_progress",float,-1.0,"Enable Syn after this training-progress fraction"),
    ("--rae_syn_temp",float,1.0,"Temperature for attention weight sharpening"),
    ("--rae_syn_exclude_entity",bool,False,"Do not apply Syn loss to entity tokens"),
    ("--rae_syn_exclude_reading",bool,False,"Do not apply Syn loss to reading-sensitive tokens"),
    ("--rae_syn_exclude_content_after_para",bool,False,
     "After content Para starts, do not apply Syn loss to masked content tokens"),
    ("--rae_syn_content_after_para_scale",float,-1.0,
     "If >=0, scale Syn loss on masked content tokens after content Para starts"),

    # ── 聚合轴 Para（embedding 近邻）──
    ("--rae_para",bool,False,"Enable embedding-neighbor para loss"),
    ("--rae_para_weight",float,0.0003,"Para loss scale (content token)"),
    ("--rae_para_topk",int,8,"Number of embedding-space neighbors"),
    ("--rae_para_warmup_steps",int,2000,"Legacy content Para warmup step; used only if progress is not set"),
    ("--rae_para_warmup_progress",float,-1.0,"Enable content Para after this training-progress fraction"),
    ("--rae_para_ramp_progress",float,0.0,"Linearly ramp content Para loss after its gate"),
    ("--rae_para_temp",float,1.0,"Temperature for neighbor similarity"),

    # ── Token-type-gated Para 权重 ──
    ("--rae_para_syntax_weight",float,0.0001,"Syntax token para (very light)"),
    ("--rae_para_syntax_warmup_steps",int,-1,"Legacy syntax-token Para warmup; -1 uses content Para warmup"),
    ("--rae_para_syntax_warmup_progress",float,-1.0,"Enable syntax-token Para after this training-progress fraction"),
    ("--rae_content_proj",bool,False,"Route content-token Para through a residual projection head"),
    ("--rae_content_proj_residual_alpha",float,0.5,"Residual scale for content Para projection"),
    ("--rae_para_entity_weight",float,0.0,"Entity token para (ZERO)"),
    ("--rae_para_reading_weight",float,0.0,"Reading token para (ZERO)"),
]:
    n, t, d = a[0], a[1], a[2]
    h = a[3] if len(a) > 3 else ""
    if t == bool: parser.add_argument(n, action="store_true", help=h)
    else: parser.add_argument(n, type=t, default=d, help=h)


# ======================== Eval / Data 工具 ========================
def evaluate(model, tokenizer, dataloader, args):
    model.eval(); c=t=0; s,n=0.0,0
    with torch.no_grad():
        for batch in dataloader:
            if len(batch["input_ids"])==0: continue
            batch=to_cuda(batch)
            mb=mask_batch(batch,tokenizer,None,0.15,0.8,0.1)
            for m in split_batch(mb,args):
                mc = to_cuda(m)
                model_inputs = {k: v for k, v in mc.items() if k != "original_input_ids"}
                with torch.autocast(dtype=torch.bfloat16,device_type="cuda:0"):
                    o=model(**model_inputs)
                s+=o.loss.item();n+=1
                p=o.logits.argmax(-1);lab=mc["labels"].to(device=p.device)
                mk=lab!=-100;c+=(p[mk]==lab[mk]).sum().item();t+=mk.sum().item()
    model.train()
    return {'acc':100*c/t if t else 0,'loss':s/max(1,n)}


def regroup_texts(args,ms):
    gd=args.dataset.map(group_texts,batched=True,fn_kwargs={'max_len':ms},num_proc=args.cpus)
    args.batch_size=max(1,int(args.batch_size/(ms/args.cur_max_seq_len)))
    tr=torch.utils.data.DataLoader(gd['train'],batch_size=args.batch_size,num_workers=args.cpus,shuffle=True,collate_fn=padding_collate_fn,pin_memory=True,persistent_workers=args.cpus>0)
    ev=torch.utils.data.DataLoader(gd['validation'],batch_size=args.batch_size,num_workers=args.cpus,shuffle=False,collate_fn=padding_collate_fn,pin_memory=True,persistent_workers=args.cpus>0)
    args.cur_max_seq_len=ms;return tr,ev


def mask_batch(batch,tokenizer,mask_weights=None,mlm_prob=0.15,mask_replace_prob=0.8,random_replace_prob=0.1):
    dev=batch["input_ids"].device
    input_ids=batch["input_ids"]
    if mask_weights is None:
        mask_weights=torch.full((tokenizer.vocab_size,),mlm_prob,device=dev)
    else:
        mask_weights=mask_weights.to(device=dev)

    original_input_ids=input_ids.clone()
    labels=batch["labels"].clone()
    weights=mask_weights[original_input_ids].float()
    weights=weights.masked_fill(original_input_ids==tokenizer.pad_token_id,0.0)
    denom=weights.sum(dim=1,keepdim=True).clamp_min(1e-8)
    probs=mlm_prob*input_ids.shape[1]*weights/denom

    selected=torch.rand(input_ids.shape,device=dev)<probs
    rand=torch.rand(input_ids.shape,device=dev)
    to_mask=(rand<mask_replace_prob)&selected
    to_replace=(rand>=mask_replace_prob)&(rand<mask_replace_prob+random_replace_prob)&selected

    masked_input_ids=input_ids.clone()
    random_ids=torch.randint(0,tokenizer.vocab_size,input_ids.shape,device=dev)
    masked_input_ids[to_mask]=tokenizer.mask_token_id
    masked_input_ids[to_replace]=random_ids[to_replace]
    labels[~selected]=-100

    out={
        "input_ids":masked_input_ids,
        "labels":labels,
        "original_input_ids":original_input_ids,
    }
    if "attention_mask" in batch: out["attention_mask"]=batch["attention_mask"].clone()
    return out


def get_batch_accuracy(logits,labels,stats):
    mk=labels!=-100
    if mk.sum()==0:return stats
    v=logits.shape[-1];lm=labels[mk];pr=logits.argmax(-1)[mk];cm=pr==lm
    stats['correct']+=torch.bincount(lm[cm],minlength=v)
    stats['incorrect']+=torch.bincount(lm[~cm],minlength=v)
    return stats


def update_mask_weights(mw,ms,mlm_prob=0.15):
    cp=(ms['correct']+0.5)/(ms['incorrect']+ms['correct']+1)
    nw=mlm_prob-(cp*mlm_prob);mw=0.2*mw+0.8*nw;mw=mw.clamp(0.005)
    return mlm_prob*mw.shape[0]*mw/mw.sum()


def reset_stats(s):return{'correct':torch.zeros_like(s['correct']),'incorrect':torch.zeros_like(s['incorrect'])}
def split_batch(batch,args):
    ms=args.batch_size//args.grad_acc
    if len(batch["input_ids"])==ms:return[batch]
    return[{k:v[i:i+ms]for k,v in batch.items()if v is not None}for i in range(0,len(batch["input_ids"]),ms)]
def to_cuda(d):return{k:v.to(device="cuda:0")for k,v in d.items()if v is not None}

def calc_total_steps(args):
    def c(tpk,ml):return sum([t//ml for t in tpk])
    epe=c(args.tokens_per_1000,args.init_max_seq_len);bpe=math.ceil(epe/args.batch_size);total=bpe*args.epochs
    if len(args.max_seq_len)>0:
        ce,pl,bs=0,args.init_max_seq_len,args.batch_size;t=0
        for en,sl in args.max_seq_len:
            t+=bpe*(en-ce);bs=int(bs*(pl/sl));epe=c(args.tokens_per_1000,sl);bpe=math.ceil(epe/bs);ce,pl=en,sl
        t+=bpe*(args.epochs-ce);return t
    return total

def _resolve_gate_progress(name, progress_value, step_value, total_steps):
    if progress_value >= 0:
        value = progress_value
        source = "progress"
    else:
        value = step_value / max(1, total_steps)
        source = "legacy_step_ratio"
    if value < 0 or value > 1:
        raise ValueError(f"{name} progress must be in [0, 1], got {value}")
    return value, source

def _progress_to_step(progress, total_steps):
    return int(round(progress * total_steps))

def _progress_ramp(cur_progress, start_progress, ramp_progress):
    if ramp_progress <= 0:
        return 1.0
    if cur_progress <= start_progress:
        return 0.0
    return min(1.0, (cur_progress - start_progress) / ramp_progress)

def is_step(st,gs,args):
    sa=getattr(args,f'{st}_steps')
    return gs in args.checkpoints if args.all_checkpoints else(gs%sa==0 and gs!=0)


# ======================== v7 核心: 注意力引导组合轴 ========================
def compute_syn_loss_attention(hidden, attentions, input_ids, labels, target_input_ids, emb_weight,
                               pad_id, cls_id, sep_id, mask_id, args, device,
                               syn_allowed_mask=None, loss_weights=None):
    """
    组合轴 (Syntagmatic) — 注意力引导的结构上下文 ──────────────────────

    理论（idea.md）:
      "一个槽位如何和句子里的其他槽位组合成结构"
      "boy ↔ who chased the dog (定语从句修饰), boy ↔ was tired (主句主谓)"

    实现:
      1. 取 last-layer attention (平均所有 head)
      2. 排除 self / pad / cls / sep / mask
      3. 对每个 [MASK] 位置，取 top-k 被关注 token
      4. syn_target = attention_weighted_average(embedding(被关注 token))
      5. loss = 1 - cosine_similarity(hidden[MASK], syn_target)

    直觉: "被 [MASK] 关注的 token 定义了它的结构角色，
           [MASK] 的 hidden state 应该编码这些 token 的身份信息"
    """
    mask = labels != -100
    if syn_allowed_mask is not None:
        mask = mask & syn_allowed_mask
    M = mask.sum().item()
    if M == 0:
        return torch.tensor(0.0, device=device), 0

    B, L, H = hidden.shape
    # 平均所有 head 的注意力 → [B, L, L]
    attn_avg = attentions.mean(dim=1)  # attentions: [B, num_heads, L, L]

    # 构建排除 mask: self + pad + cls + sep + mask token
    exclude = torch.zeros(B, L, dtype=torch.bool, device=device)
    for tid in [pad_id, cls_id, sep_id, mask_id]:
        if tid is not None:
            exclude = exclude | (input_ids == tid)
    # 也排除所有 MLM-supervised positions，避免 random/unchanged selected tokens 泄漏 gold target。
    exclude = exclude | (labels != -100)
    # 也排除自身注意力
    self_mask = torch.eye(L, device=device).unsqueeze(0).bool()  # [1, L, L]
    exclude_3d = exclude.unsqueeze(1) | self_mask  # [B, L, L]

    attn_masked = attn_avg.masked_fill(exclude_3d, float('-inf'))

    # 温度锐化注意力分布
    attn_sharp = attn_masked / max(args.rae_syn_temp, 0.01)

    # Top-k 被关注 token（全部位置，向量化）→ [B, L, K]
    K = min(args.rae_syn_topk, L)
    topk_scores, topk_idx = attn_sharp.topk(K, dim=-1)  # [B, L, K]

    # 只取 masked 位置
    mask_idx = mask.nonzero(as_tuple=False)  # [M, 2]
    batch_idx = mask_idx[:, 0]
    seq_idx   = mask_idx[:, 1]

    # 被关注 token 的 ID 和 attention 权重。短序列可能没有任何有效上下文，需跳过。
    ctx_indices = topk_idx[batch_idx, seq_idx]       # [M, K]
    ctx_scores = topk_scores[batch_idx, seq_idx]     # [M, K]
    valid_rows = torch.isfinite(ctx_scores).any(dim=-1)
    if valid_rows.sum().item() == 0:
        return torch.tensor(0.0, device=device), 0

    batch_idx = batch_idx[valid_rows]
    seq_idx = seq_idx[valid_rows]
    ctx_indices = ctx_indices[valid_rows]
    ctx_weights = F.softmax(ctx_scores[valid_rows], dim=-1)
    row_weights = None
    if loss_weights is not None:
        row_weights = loss_weights[batch_idx, seq_idx].float()
        keep = row_weights > 0
        if keep.sum().item() == 0:
            return torch.tensor(0.0, device=device), 0
        batch_idx = batch_idx[keep]
        seq_idx = seq_idx[keep]
        ctx_indices = ctx_indices[keep]
        ctx_weights = ctx_weights[keep]
        row_weights = row_weights[keep]

    # target_input_ids 是未污染原文/gold token，用它取 target embedding。
    ctx_token_ids = target_input_ids[batch_idx.unsqueeze(-1), ctx_indices]  # [M, K]
    ctx_embs = emb_weight[ctx_token_ids]  # [M, K, H]

    # 注意力加权: syn_target = Σ attention_weight[k] * emb(token[k])
    syn_target = (ctx_weights.unsqueeze(-1) * ctx_embs).sum(dim=1)  # [M, H]

    # Cosine similarity loss. Detach target so RAE trains hidden states, not the target space.
    syn_target = F.normalize(syn_target.detach().float(), dim=-1)
    h_masked = hidden[batch_idx, seq_idx]  # [M, H]
    h_norm = F.normalize(h_masked.float(), dim=-1)

    cos_sim = (h_norm * syn_target).sum(dim=-1)  # [M]
    per_token_loss = 1.0 - cos_sim
    if row_weights is not None:
        syn_loss = (per_token_loss * row_weights).sum() / row_weights.sum().clamp_min(1e-6)
    else:
        syn_loss = per_token_loss.mean()

    return syn_loss, int(batch_idx.numel())


# ======================== v7 核心: embedding 近邻聚合轴 ========================
def compute_para_loss_neighbors(hidden, labels, emb_weight, args, device):
    """
    聚合轴 (Paradigmatic) — embedding 空间近邻定义替换类 ──────────────────────

    理论（idea.md）:
      "同一个结构槽位中，可互相替换、功能相近的词聚在一起"
      "boy → {girl, child, student, teacher, man}"

    实现:
      1. 在当前 embedding 空间中，找 gold token 的 k 近邻（余弦相似度）
      2. para_target = mean(embedding(k 近邻))
      3. loss = 1 - cosine_similarity(hidden[MASK], para_target)

    直觉: "embedding 空间的近邻 = 分布语义上可替换的词，
           模型应该知道 [MASK] 属于哪个分布语义类"

    关键: 近邻是动态的——随 embedding 空间演化而演化
    """
    mask = labels != -100
    M = mask.sum().item()
    if M == 0:
        return torch.tensor(0.0, device=device), 0

    gold_ids = labels[mask]  # [M]
    h_masked = hidden[mask]  # [M, H]

    # gold token 的 context-free embedding
    gold_emb = emb_weight[gold_ids]  # [M, H]

    # 计算 gold embedding 与所有 embedding 的余弦相似度 → [M, V]
    gold_norm = F.normalize(gold_emb.float(), dim=-1)   # [M, H]
    emb_norm  = F.normalize(emb_weight.float(), dim=-1)  # [V, H]

    # 温度调节相似度
    sim = (gold_norm @ emb_norm.T) / max(args.rae_para_temp, 0.01)  # [M, V]

    # 找到 k 近邻（+1 保留 self 位置，后面排除）
    K = min(args.rae_para_topk + 1, emb_weight.shape[0])
    _, neighbor_ids = sim.topk(K, dim=-1)  # [M, K]

    # 排除 gold token 自身（它永远是最近邻）
    # 创建一个 mask: 哪些 neighbor 不是 gold token
    not_self = neighbor_ids != gold_ids.unsqueeze(-1)  # [M, K]
    # 取前 args.rae_para_topk 个非自身的邻居
    neighbor_ids_filtered = neighbor_ids[:, :K-1]  # [M, K-1]
    # 更简单的方法: 直接取 top-(K+1) 然后 skip index 0（因为 gold=closest）
    # 实际上 sim.topk(K) 中如果 K 够大，第一个一定是 gold 自身
    # 所以 neighbors = topk_indices[:, 1:] 即可
    neighbors = neighbor_ids[:, 1:args.rae_para_topk + 1]  # [M, topk] — skip self

    # para_target = mean of neighbor embeddings
    neighbor_embs = emb_weight[neighbors]  # [M, topk, H]
    para_target = neighbor_embs.float().mean(dim=1)  # [M, H]

    # Cosine similarity loss. Detach target so the auxiliary loss does not drag embeddings.
    para_target = F.normalize(para_target.detach(), dim=-1)
    h_norm = F.normalize(h_masked.float(), dim=-1)

    cos_sim = (h_norm * para_target).sum(dim=-1)  # [M]
    para_loss = (1.0 - cos_sim).mean()

    return para_loss, M


# ======================== 全局缓存 ========================
_G = {}


def make_content_projection(hidden_size):
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.LayerNorm(hidden_size),
    )


def save_content_projection(path, content_proj):
    if content_proj is not None:
        torch.save({"content_proj": content_proj.state_dict()}, os.path.join(path, "rae_content_proj.pt"))


# ======================== Train ========================
def train(args, model, tokenizer, train_dl, eval_dl):
    global _G
    is_syntax, is_reading, is_entity, is_content, is_punct = build_token_categories(tokenizer)
    for name, t in [("syntax",is_syntax),("reading",is_reading),("entity",is_entity),
                     ("content",is_content),("punct",is_punct)]:
        _G[f"is_{name}"] = t.to(device="cuda:0")
        print(f"  {name}: {t.sum().item()} tokens ({100*t.sum().item()/tokenizer.vocab_size:.1f}%)")

    syn_enabled = args.rae_syn and args.rae_syn_weight > 0
    para_enabled = args.rae_para and args.rae_para_weight > 0

    sprog = args.rae_syn_warmup_progress_effective
    pprog = args.rae_para_warmup_progress_effective
    psprog = args.rae_para_syntax_warmup_progress_effective

    print(f"Steps: {args.total_steps}")
    if syn_enabled:
        print(f"  Syn: attention-guided, layer={args.rae_syn_layer}, topk={args.rae_syn_topk}, "
              f"T={args.rae_syn_temp}, w={args.rae_syn_weight}, gate_progress={sprog:.6f} "
              f"(~step {_progress_to_step(sprog, args.total_steps)}/{args.total_steps}), "
              f"exclude_entity={args.rae_syn_exclude_entity}, "
              f"exclude_reading={args.rae_syn_exclude_reading}, "
              f"exclude_content_after_para={args.rae_syn_exclude_content_after_para}, "
              f"content_after_para_scale={args.rae_syn_content_after_para_scale}")
    else:
        print("  Syn: disabled")
    if para_enabled:
        print(f"  Content Para: embedding neighbors, topk={args.rae_para_topk}, "
              f"T={args.rae_para_temp}, w={args.rae_para_weight}, gate_progress={pprog:.6f} "
              f"(~step {_progress_to_step(pprog, args.total_steps)}/{args.total_steps}), "
              f"ramp_progress={args.rae_para_ramp_progress}")
        print(f"  Syntax Para: w={args.rae_para_syntax_weight}, gate_progress={psprog:.6f} "
              f"(~step {_progress_to_step(psprog, args.total_steps)}/{args.total_steps})")
        print(f"    Entity para={args.rae_para_entity_weight} Reading para={args.rae_para_reading_weight}")
        print(f"  RAE constraints only on masked positions")
    else:
        print("  Para: disabled")
    print("  Masking: regular clean MLM" if args.regular_mlm else "  Masking: adaptive AMLM-compatible weights")
    print(f"Progress gates: syn={sprog:.6f}, syntax_para={psprog:.6f}, content_para={pprog:.6f}", flush=True)

    model.train()
    model = model.to(dtype=torch.bfloat16, device="cuda:0")

    content_proj = None
    opt_params = list(model.parameters())
    if para_enabled and args.rae_content_proj:
        content_proj = make_content_projection(args.hidden_size).to(dtype=torch.bfloat16, device="cuda:0")
        opt_params += list(content_proj.parameters())
        print(f"  Content Para projection: residual alpha={args.rae_content_proj_residual_alpha}")
    else:
        print("  Content Para projection: disabled")

    if args.lamb:
        if not LAMB_OK: raise ImportError("bitsandbytes needed for LAMB")
        opt = LAMB(opt_params, lr=args.lr, betas=(0.9,0.95), eps=1e-08, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(opt_params, lr=args.lr, betas=(0.9,0.95), eps=1e-08, weight_decay=args.weight_decay)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=args.total_steps//100, num_training_steps=args.total_steps)

    mw = torch.full((tokenizer.vocab_size,), args.mlm_prob, device="cuda:0")
    ms = {'correct': torch.zeros(tokenizer.vocab_size, dtype=torch.float32, device="cuda:0"),
          'incorrect': torch.zeros(tokenizer.vocab_size, dtype=torch.float32, device="cuda:0")}
    gs = 0; _ew = model.get_input_embeddings().weight
    gate_logged = {"syn": False, "syntax_para": False, "content_para": False}

    with tqdm(total=args.total_steps) as pbar:
        for epoch in range(args.epochs):
            if len(args.max_seq_len) > 0 and epoch >= args.max_seq_len[0][0]:
                train_dl, eval_dl = regroup_texts(args, args.max_seq_len[0][1])
                args.max_seq_len = args.max_seq_len[1:]

            for step, batch in enumerate(train_dl):
                batch = to_cuda(batch)
                mask_weights_arg = None if args.regular_mlm else mw
                mb = mask_batch(batch, tokenizer, mask_weights_arg, args.mlm_prob, args.mask_replace_prob, args.random_replace_prob)
                batches = split_batch(mb, args)

                for minibatch in batches:
                    mc = to_cuda(minibatch); dev = mc["input_ids"].device
                    input_ids = mc["input_ids"]; labels = mc["labels"]
                    cur_progress = gs / max(1, args.total_steps)
                    original_input_ids = mc["original_input_ids"]
                    model_inputs = {k: v for k, v in mc.items() if k != "original_input_ids"}

                    # 构建 token 类型 mask：masked 位置用 gold label，其余位置用原始 token。
                    # 不能用 corrupted input_ids，否则 [MASK]/random token 会绕过 entity/reading gating。
                    type_ids = original_input_ids.clone()
                    masked_positions = labels != -100
                    type_ids[masked_positions] = labels[masked_positions]
                    cm = {
                        'syntax':  _G['is_syntax'][type_ids],
                        'reading': _G['is_reading'][type_ids],
                        'entity':  _G['is_entity'][type_ids],
                        'content': _G['is_content'][type_ids],
                        'punct':   _G['is_punct'][type_ids],
                        'valid':   (type_ids != tokenizer.pad_token_id) & (~_G['is_punct'][type_ids]),
                    }

                    if syn_enabled and (not gate_logged["syn"]) and cur_progress >= sprog:
                        print(f"Gate activate: Syn step={gs} progress={cur_progress:.6f} mlm_prob={args.mlm_prob:.6f}", flush=True)
                        gate_logged["syn"] = True
                    if para_enabled and args.rae_para_syntax_weight > 0 and (not gate_logged["syntax_para"]) and cur_progress >= psprog:
                        print(f"Gate activate: SyntaxPara step={gs} progress={cur_progress:.6f} mlm_prob={args.mlm_prob:.6f}", flush=True)
                        gate_logged["syntax_para"] = True
                    if para_enabled and (not gate_logged["content_para"]) and cur_progress >= pprog:
                        print(f"Gate activate: ContentPara step={gs} progress={cur_progress:.6f} mlm_prob={args.mlm_prob:.6f}", flush=True)
                        gate_logged["content_para"] = True

                    # 决定是否输出 attention（syn 需要）
                    need_attn = syn_enabled and cur_progress >= sprog

                    with torch.autocast(dtype=torch.bfloat16, device_type="cuda:0"):
                        o = model(**model_inputs, output_hidden_states=True,
                                  output_attentions=need_attn)
                        loss = o.loss
                        syn_l = para_l = para_s = torch.tensor(0.0, device=dev)
                        sm = pm = ps = 0

                        # ── 组合轴 Syn: 注意力引导 ──
                        if syn_enabled and cur_progress >= sprog and o.attentions is not None:
                            layer_attn = o.attentions[args.rae_syn_layer]  # [B, H, L, L]
                            syn_allowed = cm["valid"]
                            syn_weights = None
                            if args.rae_syn_exclude_entity:
                                syn_allowed = syn_allowed & ~cm["entity"]
                            if args.rae_syn_exclude_reading:
                                syn_allowed = syn_allowed & ~cm["reading"]
                            if args.rae_syn_exclude_content_after_para and cur_progress >= pprog:
                                syn_allowed = syn_allowed & ~cm["content"]
                            elif args.rae_syn_content_after_para_scale >= 0 and cur_progress >= pprog:
                                syn_weights = torch.ones_like(labels, dtype=torch.float32, device=dev)
                                syn_weights = torch.where(
                                    cm["content"],
                                    syn_weights * args.rae_syn_content_after_para_scale,
                                    syn_weights,
                                )
                                syn_weights = syn_weights.masked_fill(~cm["valid"], 0.0)
                            syn_l, sm = compute_syn_loss_attention(
                                o.hidden_states[-1], layer_attn, input_ids, labels, type_ids,
                                _ew,
                                tokenizer.pad_token_id, tokenizer.cls_token_id,
                                tokenizer.sep_token_id, tokenizer.mask_token_id,
                                args, dev, syn_allowed, loss_weights=syn_weights)
                            loss = loss + args.rae_syn_weight * syn_l

                        # ── 聚合轴 Para: embedding 近邻 ──
                        content_para_scale = 0.0
                        if para_enabled and cur_progress >= pprog:
                            content_para_scale = _progress_ramp(cur_progress, pprog, args.rae_para_ramp_progress)
                        if para_enabled and cur_progress >= min(pprog, psprog):
                            # 构建 per-token para 许可 mask
                            para_mask = cm['valid']
                            if args.rae_para_entity_weight <= 0:
                                para_mask = para_mask & ~cm['entity']
                            if args.rae_para_reading_weight <= 0:
                                para_mask = para_mask & ~cm['reading']

                            # Content tokens: 主力 para
                            content_para_mask = (labels != -100) & para_mask & cm['content']
                            if cur_progress >= pprog and content_para_mask.sum() > 0:
                                h_content = o.hidden_states[-1]
                                if content_proj is not None:
                                    alpha = args.rae_content_proj_residual_alpha
                                    h_content = h_content + alpha * content_proj(h_content)
                                l_content = labels.clone()
                                l_content[~content_para_mask] = -100
                                para_l, pm = compute_para_loss_neighbors(
                                    h_content, l_content, _ew, args, dev)
                                loss = loss + args.rae_para_weight * content_para_scale * para_l

                            # Syntax tokens: 轻量 para
                            if args.rae_para_syntax_weight > 0 and cur_progress >= psprog:
                                syn_para_mask = (labels != -100) & cm['valid'] & cm['syntax']
                                if syn_para_mask.sum() > 0:
                                    h_syn = o.hidden_states[-1].clone()
                                    l_syn = labels.clone()
                                    l_syn[~syn_para_mask] = -100
                                    para_s, ps = compute_para_loss_neighbors(
                                        h_syn, l_syn, _ew, args, dev)
                                    loss = loss + args.rae_para_syntax_weight * para_s

                    if not args.regular_mlm:
                        with torch.no_grad():
                            ms = get_batch_accuracy(o.logits.detach(), labels, ms)
                    (loss / args.grad_acc).backward()

                torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
                opt.step(); sch.step(); opt.zero_grad()

                if (not args.regular_mlm) and gs % args.mask_update_steps == 0 and gs != 0:
                    mw = update_mask_weights(mw, ms, args.mlm_prob)
                    ms = reset_stats(ms)

                # ── Logging ──
                if is_step("logging", gs, args):
                    ep = gs * args.epochs / args.total_steps
                    parts = [f"Ep{ep:.2f} MLM{o.loss.item():.3f}"]
                    if cur_progress >= sprog and syn_enabled:
                        parts.append(f"Syn{syn_l.item():.4f}(m{sm})")
                    if cur_progress >= min(pprog, psprog) and para_enabled:
                        if cur_progress >= pprog:
                            parts.append(f"Para{para_l.item():.4f}(m{pm})x{content_para_scale:.2f}")
                        if cur_progress >= psprog and para_s.item() > 0:
                            parts.append(f"SynP{para_s.item():.4f}")
                    parts.append(f"LR{sch.get_last_lr()[0]:.1e}")
                    print(" | ".join(parts), flush=True)
                    if args.wandb:
                        wd = {"train/epoch": ep, "train/mlm_loss": o.loss.item(), "train/lr": sch.get_last_lr()[0]}
                        if cur_progress >= sprog and syn_enabled:
                            wd["train/syn_loss"] = syn_l.item(); wd["train/syn_masked"] = sm
                        if cur_progress >= min(pprog, psprog) and para_enabled:
                            if cur_progress >= pprog:
                                wd["train/para_loss"] = para_l.item(); wd["train/para_masked"] = pm; wd["train/content_para_scale"] = content_para_scale
                            if cur_progress >= psprog:
                                wd["train/syntax_para_loss"] = para_s.item(); wd["train/syntax_para_masked"] = ps
                        wandb.log(wd, step=gs)

                if is_step("eval", gs, args):
                    m = evaluate(model, tokenizer, eval_dl, args)
                    print(f"----- Eval acc {m['acc']:.2f} Loss {m['loss']:.4f} -----", flush=True)
                    if args.wandb:
                        wandb.log({"eval/acc": m["acc"], "eval/loss": m["loss"]}, step=gs)

                if is_step("save", gs, args):
                    tag = getattr(args, "checkpoint_tags", {}).get(gs, str(gs))
                    sp = os.path.join(args.output_path, f"chck_{tag}")
                    model.save_pretrained(sp); tokenizer.save_pretrained(sp)
                    save_content_projection(sp, content_proj)
                    print(f"--- Saved: {sp} ---", flush=True)

                pbar.update(1); gs += 1
                if args.mask_decay > 0:
                    args.mlm_prob -= args.mask_decay / args.total_steps

    m = evaluate(model, tokenizer, eval_dl, args)
    print(f"Final eval acc {m['acc']:.2f} Loss {m['loss']:.4f}", flush=True)
    fp = os.path.join(args.output_path, "chck_100M")
    model.save_pretrained(fp); tokenizer.save_pretrained(fp)
    save_content_projection(fp, content_proj)
    if args.wandb:
        wandb.log({"final/eval_acc": m["acc"], "final/eval_loss": m["loss"]})
        wandb.finish()


# ======================== Init ========================
def load_tokenizer(tp, lower=False):
    if tp is None: raise ValueError("--tokenizer required")
    if os.path.isdir(tp): return AutoTokenizer.from_pretrained(tp, use_fast=False)
    try: return DebertaV2Tokenizer(vocab_file=tp, do_lower_case=lower)
    except:
        try:
            tok = PreTrainedTokenizerFast(tokenizer_file=tp)
            if tok.mask_token is None:
                tok.add_special_tokens({"pad":"[PAD]","unk":"[UNK]","cls":"[CLS]","sep":"[SEP]","mask":"[MASK]"})
            return tok
        except: return AutoTokenizer.from_pretrained(tp, use_fast=False)


def parse_msl(s):
    if "," in s: return [(int(v.split(":")[0]), int(v.split(":")[1])) for v in s.split(",")]
    if ":" in s: return [(0, int(s.split(":")[1]))]
    return [(0, int(s))]


def main():
    args = parser.parse_args()
    args.max_seq_len = parse_msl(args.max_seq_len); set_seed(args.seed)
    if args.lamb and not LAMB_OK: raise ImportError("--lamb requires bitsandbytes")

    if args.wandb:
        assert WB_OK, "wandb not installed"
        import wandb as wb
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wb.init(project=args.wandb_project, name=args.wandb_name or os.path.basename(args.output_path),
                config=vars(args), tags=tags)

    tokenizer = load_tokenizer(args.tokenizer, args.lower)
    print(f"Tokenizer: vocab={tokenizer.vocab_size}")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.vocab_size = tokenizer.vocab_size; config.output_hidden_states = True
    config.pad_token_id = tokenizer.pad_token_id; config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id; config.max_position_embeddings = 1024
    config.hidden_size = args.hidden_size; config.intermediate_size = args.intermediate_size
    config.dropout = args.dropout; config.hidden_dropout_prob = args.dropout

    model = AutoModelForMaskedLM.from_pretrained(args.model_path, config=config, trust_remote_code=True) \
        if args.pretrained else AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    ds = load_dataset('text', data_files={'train': args.train_data, 'validation': args.valid_data})
    if args.debug: ds['train'] = ds['train'].select(range(100)); ds['validation'] = ds['validation'].select(range(100))
    ds = ds.map(tokenize, batched=True, fn_kwargs={'tokenizer': tokenizer, 'input_field': 'text'},
                remove_columns=ds["train"].column_names, num_proc=args.cpus)
    args.dataset = ds
    msl0 = args.max_seq_len.pop(0)[1]
    args.init_max_seq_len = msl0; args.cur_max_seq_len = msl0
    args.tokens_per_1000 = ds['train'].map(
        lambda x: {'nt': [sum(len(x["input_ids"][i]) for i in range(len(x["input_ids"])))]},
        batched=True, num_proc=args.cpus, remove_columns=ds["train"].column_names)['nt']
    args.total_steps = calc_total_steps(args)
    args.rae_syn_warmup_progress_effective, _syn_src = _resolve_gate_progress(
        "rae_syn", args.rae_syn_warmup_progress, args.rae_syn_warmup_steps, args.total_steps)
    args.rae_para_warmup_progress_effective, _para_src = _resolve_gate_progress(
        "rae_para", args.rae_para_warmup_progress, args.rae_para_warmup_steps, args.total_steps)
    syntax_step = args.rae_para_syntax_warmup_steps if args.rae_para_syntax_warmup_steps >= 0 else args.rae_para_warmup_steps
    args.rae_para_syntax_warmup_progress_effective, _psyn_src = _resolve_gate_progress(
        "rae_para_syntax", args.rae_para_syntax_warmup_progress, syntax_step, args.total_steps)
    print(f"Resolved progress gates: syn={args.rae_syn_warmup_progress_effective:.6f} ({_syn_src}), "
          f"syntax_para={args.rae_para_syntax_warmup_progress_effective:.6f} ({_psyn_src}), "
          f"content_para={args.rae_para_warmup_progress_effective:.6f} ({_para_src})", flush=True)

    args.is_strict_small = (sum(args.tokens_per_1000) // 10e6) < 10
    if args.is_strict_small:
        s1 = np.round(np.linspace(args.total_steps//100, args.total_steps//10, 10)).astype(int)
        s10 = np.round(np.linspace(args.total_steps//10, args.total_steps, 10)).astype(int)
        args.checkpoints = list(s1) + list(s10)[1:]
        names = [f"{i}M" for i in range(1,11)] + [f"{i}M" for i in range(20,101,10)]
    else:
        s1 = np.linspace(args.total_steps//1000, args.total_steps//100, 10).astype(int)
        s10 = np.linspace(args.total_steps//100, args.total_steps//10, 10).astype(int)
        s100 = np.linspace(args.total_steps//10, args.total_steps, 10).astype(int)
        args.checkpoints = list(s1) + list(s10)[1:] + list(s100)[1:]
        names = [f"{i}M" for i in range(1,11)] + [f"{i}M" for i in range(20,101,10)] + [f"{i}M" for i in range(200,1001,100)]
    args.checkpoint_tags = {int(s): n for s, n in zip(args.checkpoints, names)}

    gd = ds.map(group_texts, batched=True, fn_kwargs={'max_len': msl0}, num_proc=args.cpus)
    tr = torch.utils.data.DataLoader(gd['train'], batch_size=args.batch_size, num_workers=args.cpus,
                                      shuffle=True, collate_fn=padding_collate_fn, pin_memory=True,
                                      persistent_workers=args.cpus>0)
    ev = torch.utils.data.DataLoader(gd['validation'], batch_size=args.batch_size, num_workers=args.cpus,
                                      shuffle=False, collate_fn=padding_collate_fn, pin_memory=True,
                                      persistent_workers=args.cpus>0)
    train(args, model, tokenizer, tr, ev)


if __name__ == "__main__":
    main()
