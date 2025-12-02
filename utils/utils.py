from typing import Dict, List, Any
from omegaconf import DictConfig

def get_task_name(cfg: DictConfig):
    return f"{cfg.benchmark}-{cfg.task}"

def preprocess_cb(cfg: DictConfig, examples, tokenizer) -> Dict[str, Any]:
    """
    CB (CommitmentBank): premise + hypothesis => 문장쌍 분류
    SuperGLUE-CB keys: 'premise', 'hypothesis', 'label' (라벨은 0/1/2 or str)
    """
    enc = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=cfg.truncation,
        padding=cfg.padding,
        max_length=cfg.max_length,
    )
    # (선택) 라벨이 문자열이면 숫자로 매핑
    if "label" in examples:
        label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
        if isinstance(examples["label"][0], str):
            enc["labels"] = [label_map[x] for x in examples["label"]]
        else:
            enc["labels"] = examples["label"]
    return enc


def _tokenize_mc_pairs(
    tokenizer, left_texts: List[str], right_texts_choices: List[List[str]],
    cfg: DictConfig
) -> Dict[str, Any]:
    """
    Multiple-Choice 공통 유틸:
    - left_texts[i]: i번째 예제의 공통 문장(프롬프트)
    - right_texts_choices[i]: i번째 예제의 선택지 리스트(길이 num_choices)
    반환: 각 예제마다 num_choices개 시퀀스를 만들어
         HF MultipleChoice가 기대하는 shape로 묶어서 리턴
    """
    all_input_ids, all_attention_mask = [], []
    for prompt, choices in zip(left_texts, right_texts_choices):
        # prompt를 choices 개수만큼 복제
        prompts = [prompt] * len(choices)
        tokenized = tokenizer(
            prompts,
            choices,
            truncation=cfg.truncation,
            padding=cfg.padding,        # Trainer가 collate할 때 "max_length" 맞춰줌
            max_length=cfg.max_length,
        )
        all_input_ids.append(tokenized["input_ids"])
        all_attention_mask.append(tokenized["attention_mask"])

    return {
        "input_ids": all_input_ids,                 # (batch, num_choices, seq_len)
        "attention_mask": all_attention_mask,       # (batch, num_choices, seq_len)
    }


def preprocess_copa(cfg: DictConfig, examples, tokenizer) -> Dict[str, Any]:
    """
    COPA: 상식적 인과추론 2지선다 (MultipleChoice)
    SuperGLUE-COPA keys: 'premise', 'choice1', 'choice2', 'question'('cause' or 'effect'), 'label'
    인코딩 전략:
      - cause 질문이면: pair = (premise, "because " + choiceX)
      - effect 질문이면: pair = (premise, "therefore " + choiceX)
    """
    premises: List[str] = examples["premise"]
    qtypes:   List[str] = examples["question"]  # 'cause' / 'effect'
    c1: List[str] = examples["choice1"]
    c2: List[str] = examples["choice2"]

    right_choices: List[List[str]] = []
    for qt, a, b in zip(qtypes, c1, c2):
        if qt == "cause":
            right_choices.append([f"because {a}", f"because {b}"])
        else:
            right_choices.append([f"therefore {a}", f"therefore {b}"])

    enc = _tokenize_mc_pairs(tokenizer, premises, right_choices, cfg)

    if "label" in examples:
        enc["labels"] = examples["label"]          # 0(=choice1) / 1(=choice2)
    return enc


def preprocess_wsc(cfg: DictConfig, examples, tokenizer) -> Dict[str, Any]:
    """
    WSC (WSC.fixed): 대명사 지시대상 판별 (MultipleChoice처럼 처리)
    SuperGLUE-WSC keys: 'text', 'span1_text', 'span2_text', 'label'
      - 보편적 캐스팅: (text, candidate) 쌍 두 개를 만들어 2지선다로 분류
      - 더 고급: pronoun 위치 중심으로 마킹/리포맷을 추가할 수 있으나 여기선 심플하게
    """
    texts:  List[str] = examples["text"]
    s1:     List[str] = examples["span1_text"]     # 후보1
    s2:     List[str] = examples["span2_text"]     # 후보2

    # 선택지: 텍스트+후보를 쌍으로 붙여 MultipleChoice 구성
    # (간단 전략) pair=(text, candidate) — 필요하면 접속사나 포맷을 바꿔도 됨
    right_choices = [[cand1, cand2] for cand1, cand2 in zip(s1, s2)]

    enc = _tokenize_mc_pairs(tokenizer, texts, right_choices, cfg)

    if "label" in examples:
        enc["labels"] = examples["label"]          # 0 or 1 (span1 vs span2)
    return enc


def preprocess_function(cfg: DictConfig, examples, tokenizer):
    task = cfg.task
    if task in {"cb"}:
        return preprocess_cb(cfg, examples, tokenizer)
    elif task in {"copa"}:
        return preprocess_copa(cfg, examples, tokenizer)
    elif task in {"wsc", "wsc.fixed", "wsc-fixed", "wsc_fixed"}:
        return preprocess_wsc(cfg, examples, tokenizer)
    else:
        raise ValueError(f"Unsupported task for this split-preprocessor: {task}")
