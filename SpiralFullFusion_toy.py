
# -*- coding: utf-8 -*-
# SpiralFullFusion Toy (compact complete) — reliability dynamics + partial optimization + RAG SHAP + T_S(Σ) backprop
# NumPy-only, single-file demo. Safe initializations and eps clamps to avoid NaNs.
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import re

EPS = 1e-8

# ------------------------------
# utils
# ------------------------------
def softmax_rows(Z: np.ndarray) -> np.ndarray:
    Zm = Z - np.max(Z, axis=-1, keepdims=True)
    E = np.exp(np.clip(Zm, -50.0, 50.0))
    S = E / np.maximum(E.sum(axis=-1, keepdims=True), EPS)
    return S.astype(np.float32)

GRAD_BOOST = 25.0  # amplify toy gradients so the student actually moves

def silu(x: np.ndarray) -> np.ndarray:
    return (x / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))).astype(np.float32)

def safe_tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -10.0, 10.0)).astype(np.float32)

def safe_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_clean = np.nan_to_num(a, nan=0.0, posinf=1e3, neginf=-1e3).astype(np.float32)
    b_clean = np.nan_to_num(b, nan=0.0, posinf=1e3, neginf=-1e3).astype(np.float32)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        y = a_clean @ b_clean
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    return y

# ------------------------------
# Pure-Python tokenizer (byte-level BPE-ish)
# ------------------------------
@dataclass
class TokenizerCfg:
    vocab_size: int = 1024      # desired vocabulary size (will be clamped to >= 260)
    min_freq: int = 2           # minimum pair frequency to merge
    add_bos: bool = True
    add_eos: bool = True
    lowercase: bool = False
    motif_topk: int = 48        # register top-N recurring multi-byte motifs as atomic tokens
    motif_min_len: int = 3
    motif_max_len: int = 8
    word_topk: int = 256        # register top-N words as atomic tokens before byte motifs
    word_min_freq: int = 2


class SpiralTokenizer:
    """
    Compact byte-level tokenizer with BPE-style merges + motif-aware atoms.
    No external dependencies; suitable for quick CLI demos when only a text path is available.
    """

    def __init__(self, cfg: TokenizerCfg):
        self.cfg = cfg
        self.special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
        self.unk_id, self.pad_id, self.bos_id, self.eos_id = range(4)
        self.byte_offset = len(self.special_tokens)
        # vocab bookkeeping
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        self.id_to_bytes: List[Optional[bytes]] = []
        self.merge_ranks: Dict[Tuple[int, int], int] = {}  # pair -> rank (lower is earlier/better)
        self.merge_to_id: Dict[Tuple[int, int], int] = {}  # pair -> new token id
        self.motif_to_id: Dict[bytes, int] = {}
        self.motif_lengths: List[int] = []
        self.word_to_id: Dict[bytes, int] = {}
        self.word_lengths: List[int] = []
        self._init_base_vocab()

    # ------------------ training helpers ------------------
    def _init_base_vocab(self):
        for tok in self.special_tokens:
            self._register_token(tok, None)
        for b in range(256):
            tok = f"<0x{b:02x}>"
            self._register_token(tok, bytes([b]))

    def _register_token(self, tok: str, piece: Optional[bytes]):
        self.token_to_id[tok] = len(self.token_to_id)
        self.id_to_token.append(tok)
        self.id_to_bytes.append(piece)

    def _piece(self, token_id: int) -> Optional[bytes]:
        if 0 <= token_id < len(self.id_to_bytes):
            return self.id_to_bytes[token_id]
        return None

    def _count_pairs(self, sequences: List[List[int]]) -> Dict[Tuple[int, int], int]:
        counts: Dict[Tuple[int, int], int] = {}
        for seq in sequences:
            for a, b in zip(seq, seq[1:]):
                if a < self.byte_offset or b < self.byte_offset:
                    # do not merge specials; keep structure
                    continue
                counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts

    def _merge_sequences(self, sequences: List[List[int]], pair: Tuple[int, int], new_id: int) -> List[List[int]]:
        a, b = pair
        merged: List[List[int]] = []
        for seq in sequences:
            i = 0; out = []
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    out.append(new_id); i += 2
                else:
                    out.append(seq[i]); i += 1
            merged.append(out)
        return merged

    def _merge_bytes(self, pair: Tuple[int, int]) -> bytes:
        a, b = pair
        pa = self._piece(a) or b""
        pb = self._piece(b) or b""
        return pa + pb

    def _format_piece(self, piece: bytes) -> str:
        # make piece human-readable for logging/debugging
        printable = ''.join(chr(c) if 32 <= c < 127 else '.' for c in piece)
        return f"<{len(piece)}:{printable}>"

    def _register_motifs(self, motifs: List[bytes]):
        # Register motifs as atomic tokens before merges to bias toward recurring structures.
        if not motifs:
            return
        for mot in motifs:
            name = f"<motif:{self._format_piece(mot)}>"
            if name in self.token_to_id:
                continue
            self._register_token(name, mot)
            self.motif_to_id[mot] = self.token_to_id[name]
        self.motif_lengths = sorted({len(m) for m in self.motif_to_id.keys()}, reverse=True)

    def _register_words(self, words: List[bytes]):
        if not words:
            return
        for w in words:
            name = f"<word:{self._format_piece(w)}>"
            if name in self.token_to_id:
                continue
            self._register_token(name, w)
            self.word_to_id[w] = self.token_to_id[name]
        self.word_lengths = sorted({len(w) for w in self.word_to_id.keys()}, reverse=True)

    def _find_top_motifs(self, lines: List[bytes]) -> List[bytes]:
        counts: Dict[bytes, int] = {}
        min_len = max(1, self.cfg.motif_min_len)
        max_len = max(min_len, self.cfg.motif_max_len)
        for raw in lines:
            n = len(raw)
            for i in range(n):
                for L in range(min_len, min(max_len, n - i) + 1):
                    mot = raw[i:i+L]
                    counts[mot] = counts.get(mot, 0) + 1
        # filter using min_freq to align with BPE merge threshold
        filtered = [(mot, c) for mot, c in counts.items() if c >= self.cfg.min_freq]
        filtered.sort(key=lambda mc: (mc[1], len(mc[0])), reverse=True)
        top = [mot for mot, _ in filtered[: self.cfg.motif_topk]]
        return top

    def _find_top_words(self, text: str) -> List[bytes]:
        pattern = re.compile(r"[\w']+")
        words = pattern.findall(text)
        counts: Dict[str, int] = {}
        for w in words:
            key = w if not self.cfg.lowercase else w.lower()
            counts[key] = counts.get(key, 0) + 1
        filtered = [(w, c) for w, c in counts.items() if c >= max(1, self.cfg.word_min_freq)]
        filtered.sort(key=lambda wc: (wc[1], len(wc[0])), reverse=True)
        top_words = [w for w, _ in filtered[: self.cfg.word_topk]]
        return [w.encode("utf-8") for w in top_words]

    def _tokenize_bytes_with_motifs(self, raw_bytes: bytes) -> List[int]:
        ids: List[int] = []
        i = 0; n = len(raw_bytes)
        # precompute word spans to avoid substring matches; bytes regex keeps offsets aligned
        word_spans = list(re.finditer(rb"[A-Za-z0-9_']+", raw_bytes))
        w_ptr = 0
        while i < n:
            while w_ptr < len(word_spans) and word_spans[w_ptr].end() <= i:
                w_ptr += 1
            # whole-word atom only if the pointer is exactly at the word start
            if w_ptr < len(word_spans):
                s, e = word_spans[w_ptr].span()
                if i == s:
                    chunk = raw_bytes[s:e]
                    tid = self.word_to_id.get(chunk)
                    if tid is not None:
                        ids.append(tid)
                        i = e
                        continue
            matched = False
            # prefer whole-word atoms, then motifs, else fallback bytes
            for L in self.motif_lengths:
                if i + L > n:
                    continue
                chunk = raw_bytes[i:i+L]
                tid = self.motif_to_id.get(chunk)
                if tid is not None:
                    ids.append(tid)
                    i += L
                    matched = True
                    break
            if matched:
                continue
            ids.append(self.byte_offset + raw_bytes[i])
            i += 1
        return ids

    def train_from_text(self, text: str):
        if self.cfg.lowercase:
            text = text.lower()
        top_words = self._find_top_words(text)
        self._register_words(top_words)
        # pre-tokenize raw bytes per line
        raw_lines = [line.encode("utf-8") for line in text.splitlines()]
        motifs = self._find_top_motifs(raw_lines)
        self._register_motifs(motifs)
        # initialize corpus as byte tokens per line to retain some locality
        sequences: List[List[int]] = []
        for raw_bytes in raw_lines:
            seq = self._tokenize_bytes_with_motifs(raw_bytes)
            if self.cfg.add_bos: seq = [self.bos_id] + seq
            if self.cfg.add_eos: seq = seq + [self.eos_id]
            if len(seq) > 0:
                sequences.append(seq)

        target_vocab = max(self.cfg.vocab_size, self.byte_offset + 256)
        while len(self.id_to_token) < target_vocab:
            pair_counts = self._count_pairs(sequences)
            if not pair_counts:
                break
            (best_a, best_b), freq = max(pair_counts.items(), key=lambda kv: kv[1])
            if freq < self.cfg.min_freq:
                break
            new_piece = self._merge_bytes((best_a, best_b))
            new_token = self._format_piece(new_piece)
            new_id = len(self.id_to_token)
            self._register_token(new_token, new_piece)
            pair = (best_a, best_b)
            self.merge_ranks[pair] = len(self.merge_ranks)
            self.merge_to_id[pair] = new_id
            sequences = self._merge_sequences(sequences, (best_a, best_b), new_id)

    # ------------------ encoding/decoding ------------------
    def _bpe(self, ids: List[int]) -> List[int]:
        # standard greedy BPE merge loop using merge ranks
        if not self.merge_ranks:
            return ids
        while True:
            candidates = []
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                rank = self.merge_ranks.get(pair, None)
                if rank is not None:
                    candidates.append((rank, i, pair))
            pairs = candidates
            if not pairs:
                break
            _, pos, pair = min(pairs, key=lambda x: x[0])
            new_id = self.merge_to_id[pair]
            ids = ids[:pos] + [new_id] + ids[pos + 2:]
        return ids

    def encode(self, text: str, add_special_tokens: bool = True) -> np.ndarray:
        if self.cfg.lowercase:
            text = text.lower()
        ids = self._tokenize_bytes_with_motifs(text.encode("utf-8"))
        ids = self._bpe(ids)
        if add_special_tokens and self.cfg.add_bos:
            ids = [self.bos_id] + ids
        if add_special_tokens and self.cfg.add_eos:
            ids = ids + [self.eos_id]
        # clamp to known vocab
        ids = [tid if tid < len(self.id_to_token) else self.unk_id for tid in ids]
        return np.array(ids, dtype=np.int64)

    def encode_corpus(self, text: str) -> np.ndarray:
        all_ids: List[np.ndarray] = []
        for line in text.splitlines():
            all_ids.append(self.encode(line, add_special_tokens=True))
        if not all_ids:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(all_ids)

    def decode(self, ids: np.ndarray) -> str:
        pieces: List[bytes] = []
        for tid in ids.tolist():
            if tid in (self.bos_id, self.eos_id, self.pad_id):
                continue
            piece = self._piece(tid)
            if piece is not None:
                pieces.append(piece)
        return b"".join(pieces).decode("utf-8", errors="ignore")

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    # ------------------ persistence helpers ------------------
    def state_dict(self) -> Dict[str, Any]:
        return dict(
            cfg=asdict(self.cfg),
            special_tokens=self.special_tokens,
            token_to_id=self.token_to_id,
            id_to_token=self.id_to_token,
            id_to_bytes=self.id_to_bytes,
            merge_ranks=self.merge_ranks,
            merge_to_id=self.merge_to_id,
            motif_to_id=self.motif_to_id,
            motif_lengths=self.motif_lengths,
            word_to_id=self.word_to_id,
            word_lengths=self.word_lengths,
        )

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "SpiralTokenizer":
        cfg = TokenizerCfg(**state["cfg"])
        tok = cls(cfg)
        tok.special_tokens = state["special_tokens"]
        tok.unk_id, tok.pad_id, tok.bos_id, tok.eos_id = range(4)
        tok.byte_offset = len(tok.special_tokens)
        tok.token_to_id = state["token_to_id"]
        tok.id_to_token = state["id_to_token"]
        tok.id_to_bytes = state["id_to_bytes"]
        tok.merge_ranks = state["merge_ranks"]
        tok.merge_to_id = state["merge_to_id"]
        tok.motif_to_id = state["motif_to_id"]
        tok.motif_lengths = state["motif_lengths"]
        tok.word_to_id = state.get("word_to_id", {})
        tok.word_lengths = state.get("word_lengths", [])
        return tok

    def save(self, path: str):
        np.savez_compressed(path, tokenizer_state=np.array([self.state_dict()], dtype=object))

    @classmethod
    def load(cls, path: str) -> "SpiralTokenizer":
        data = np.load(path, allow_pickle=True)
        state = data["tokenizer_state"][0].item()
        return cls.from_state(state)

# ------------------------------
# toy bigram data + RAG bCSR scorer
# ------------------------------
def make_bigram(V: int, seed: int = 0, latent: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    F = rng.normal(0, 1, size=(V, latent)).astype(np.float32)
    S = (F @ F.T) / max(1, latent**0.5)
    for i in range(V):
        for j in range(1, 4):
            S[i, (i + j) % V] += 1.5 / j
            S[i, (i - j) % V] += 1.5 / j
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P = P / np.maximum(P.sum(axis=1, keepdims=True), EPS)
    return P.astype(np.float32)

def bigram_from_tokens(tokens: np.ndarray, V: int, bos_id: int = 2, eos_id: int = 3,
                       smoothing: float = 0.5) -> np.ndarray:
    # Estimate transition probabilities directly from dataset tokens.
    counts = np.full((V, V), float(smoothing), dtype=np.float32)
    clipped = np.clip(tokens.astype(np.int64), 0, V - 1)
    for a, b in zip(clipped[:-1], clipped[1:]):
        if a == eos_id or b == bos_id:
            continue
        counts[a, b] += 1.0
    P = counts / np.maximum(counts.sum(axis=1, keepdims=True), EPS)
    return P.astype(np.float32)

def sample_batch_bigram(P: np.ndarray, ctx_len: int, B: int, rng: np.random.Generator):
    V = P.shape[0]
    tokens = np.zeros((B, ctx_len + 1), dtype=np.int64)
    tokens[:, 0] = rng.integers(0, V, size=(B,), endpoint=False)
    for t in range(1, ctx_len + 1):
        prev = tokens[:, t - 1]
        u = rng.random((B,))
        cdf = np.cumsum(P[prev], axis=1)
        tokens[:, t] = (u[:, None] > cdf).sum(axis=1)
    ctx = tokens[:, :ctx_len].copy()
    y = tokens[:, ctx_len].copy()
    return ctx, y

def bootstrap_trigram_from_bigram(P: np.ndarray, n_tokens: int = 50000, seed: int = 0) -> Dict[Tuple[int, int], np.ndarray]:
    # Sample a synthetic sequence from the bigram LM to get a richer trigram prior.
    V = P.shape[0]
    rng = np.random.default_rng(seed)
    seq = [int(rng.integers(0, V))]
    for _ in range(max(0, n_tokens - 1)):
        prev = seq[-1]
        nxt = int(rng.choice(np.arange(V), p=P[prev]))
        seq.append(nxt)
    tokens = np.array(seq, dtype=np.int64)
    return trigram_from_tokens(tokens, V)

def trigram_from_tokens(tokens: np.ndarray, V: int, bos_id: int = 2, eos_id: int = 3,
                        smoothing: float = 0.25) -> Dict[Tuple[int, int], np.ndarray]:
    # Sparse trigram with additive smoothing per (a,b) pair.
    counts: Dict[Tuple[int, int], np.ndarray] = {}
    clipped = np.clip(tokens.astype(np.int64), 0, V - 1)
    for a, b, c in zip(clipped[:-2], clipped[1:-1], clipped[2:]):
        if a == eos_id or b == eos_id or c == bos_id:
            continue
        key = (int(a), int(b))
        if key not in counts:
            counts[key] = np.full((V,), float(smoothing), dtype=np.float32)
        counts[key][c] += 1.0
    trigram_probs: Dict[Tuple[int, int], np.ndarray] = {}
    for key, cnt in counts.items():
        trigram_probs[key] = cnt / np.maximum(cnt.sum(), EPS)
    return trigram_probs


def pack_trigram(tri: Optional[Dict[Tuple[int, int], np.ndarray]], V: int) -> Optional[Dict[str, np.ndarray]]:
    if tri is None:
        return None
    if len(tri) == 0:
        return dict(keys=np.zeros((0, 2), dtype=np.int64), vals=np.zeros((0, V), dtype=np.float32))
    keys = np.array(list(tri.keys()), dtype=np.int64)
    vals = np.stack([tri[tuple(k)] for k in keys], axis=0).astype(np.float32)
    return dict(keys=keys, vals=vals)


def unpack_trigram(packed: Optional[Dict[str, np.ndarray]]) -> Optional[Dict[Tuple[int, int], np.ndarray]]:
    if packed is None:
        return None
    keys = packed.get("keys", None)
    vals = packed.get("vals", None)
    if keys is None or vals is None:
        return None
    if keys.shape[0] == 0:
        return {}
    return {(int(a), int(b)): vals[i].astype(np.float32) for i, (a, b) in enumerate(keys)}

def _line_spans_from_bos_eos(data: np.ndarray, ctx_len: int, bos_id: int = 2, eos_id: int = 3) -> List[Tuple[int, int]]:
    # Extract contiguous spans [bos_idx, eos_idx] that are long enough for a ctx_len+1 window.
    spans: List[Tuple[int, int]] = []
    bos_positions = np.where(data == bos_id)[0]
    eos_positions = np.where(data == eos_id)[0]
    if bos_positions.size == 0 or eos_positions.size == 0:
        return spans
    for bos_idx in bos_positions:
        # find first EOS strictly after this BOS
        eos_idx_pos = eos_positions[eos_positions > bos_idx]
        if eos_idx_pos.size == 0:
            break
        eos_idx = int(eos_idx_pos[0])
        # need at least ctx_len+1 tokens before EOS to form (ctx, y) without crossing boundary
        if eos_idx - bos_idx >= ctx_len + 1:
            spans.append((int(bos_idx), eos_idx))
    return spans

def sample_batch_from_array(data: np.ndarray, ctx_len: int, B: int, rng: np.random.Generator, V: int,
                            bos_id: int = 2, eos_id: int = 3):
    # data: 1D token ids; avoid crossing BOS/EOS boundaries when possible
    N = data.shape[0]
    if N < ctx_len + 1:
        raise ValueError("data sequence too short for given ctx_len")
    spans = _line_spans_from_bos_eos(data, ctx_len, bos_id=bos_id, eos_id=eos_id)
    ctx = np.zeros((B, ctx_len), dtype=np.int64)
    y = np.zeros((B,), dtype=np.int64)
    for b in range(B):
        if spans:
            smin, smax = spans[int(rng.integers(0, len(spans)))]
            start = int(rng.integers(smin, smax - ctx_len))
        else:
            start = int(rng.integers(0, N - ctx_len))
        span = data[start:start+ctx_len+1]
        ctx[b] = span[:-1]
        y[b] = span[-1]
    # clip token ids to vocab
    ctx = np.clip(ctx, 0, V - 1)
    y = np.clip(y, 0, V - 1)
    return ctx, y

def build_ctx_target_pairs(tokens: np.ndarray, ctx_len: int, V: int, bos_id: int = 2, eos_id: int = 3,
                           max_windows: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    spans = _line_spans_from_bos_eos(tokens, ctx_len, bos_id=bos_id, eos_id=eos_id)
    ctxs: List[np.ndarray] = []; ys: List[int] = []
    use_cap = max_windows is not None and max_windows > 0
    rng_cap = rng if rng is not None else np.random.default_rng(0)
    seen = 0

    def maybe_add(window_ctx: np.ndarray, target: int, seen_so_far: int):
        if not use_cap:
            ctxs.append(window_ctx); ys.append(target)
            return seen_so_far + 1
        if len(ctxs) < max_windows:  # fill reservoir
            ctxs.append(window_ctx); ys.append(target)
        else:
            j = int(rng_cap.integers(0, seen_so_far + 1))
            if j < max_windows:
                ctxs[j] = window_ctx; ys[j] = target
        return seen_so_far + 1

    clipped = np.clip(tokens.astype(np.int64), 0, V - 1)
    if spans:
        for smin, smax in spans:
            for start in range(smin, smax - ctx_len):
                window = clipped[start:start+ctx_len+1]
                seen = maybe_add(window[:-1], int(window[-1]), seen)
    else:
        for start in range(0, clipped.shape[0] - ctx_len):
            window = clipped[start:start+ctx_len+1]
            seen = maybe_add(window[:-1], int(window[-1]), seen)
    if not ctxs:
        return np.zeros((0, ctx_len), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.stack(ctxs, axis=0), np.array(ys, dtype=np.int64)

def rag_bcsr_ngram(tokens_batch: np.ndarray, vocab_size: int, bigram_P: np.ndarray,
                   trigram_map: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
                   m: int = 8, w: float = 1.5):
    B = tokens_batch.shape[0]
    indices_list = []; data_list = []; indptr = [0]
    for b in range(B):
        probs = None
        if trigram_map is not None and tokens_batch.shape[1] >= 2:
            key = (int(tokens_batch[b, -2]), int(tokens_batch[b, -1]))
            probs = trigram_map.get(key)
        if probs is None:
            last = int(tokens_batch[b, -1])
            probs = bigram_P[last]
        idx = np.argpartition(probs, -m)[-m:]
        vals = probs[idx] * float(w)
        indices_list.append(idx.astype(np.int64))
        data_list.append(vals.astype(np.float32))
        indptr.append(indptr[-1] + len(idx))
    indices = np.concatenate(indices_list) if indices_list else np.array([], dtype=np.int64)
    data = np.concatenate(data_list) if data_list else np.array([], dtype=np.float32)
    return ("bcsr", np.array(indptr, dtype=np.int64), indices, data)

# ------------------------------
# Teacher: ToyDeepLM + BayesHeadFuseLR + Reliability tracking + RAG SHAP
# ------------------------------
class ToyDeepLM:
    def __init__(self, V: int, d: int, H: int, L: int, seed: int = 0):
        self.V, self.d, self.H, self.L = V, d, H, L
        rng = np.random.default_rng(seed)
        self.E = rng.normal(0, 0.02, size=(V, d)).astype(np.float32)
        self.W_layers = rng.normal(0, 0.05, size=(L, d, d)).astype(np.float32) / np.sqrt(max(1, d))
        self.b_layers = np.zeros((L, d), dtype=np.float32)
        self.W_heads = rng.normal(0, 0.05, size=(L, H, d, d)).astype(np.float32) / np.sqrt(max(1, d))
        self.b_heads = np.zeros((L, H, d), dtype=np.float32)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return dict(E=self.E.copy(), W_layers=self.W_layers.copy(), b_layers=self.b_layers.copy(),
                    W_heads=self.W_heads.copy(), b_heads=self.b_heads.copy())

    def load_state(self, state: Dict[str, np.ndarray]):
        self.E = state["E"].astype(np.float32)
        self.W_layers = state["W_layers"].astype(np.float32)
        self.b_layers = state["b_layers"].astype(np.float32)
        self.W_heads = state["W_heads"].astype(np.float32)
        self.b_heads = state["b_heads"].astype(np.float32)

    def encode_heads_per_layer(self, tokens: np.ndarray) -> List[List[np.ndarray]]:
        if tokens.ndim == 2: tokens = tokens[0]
        embs = self.E[tokens]    # [T,d]
        h_in = embs.mean(axis=0) # [d]
        outs = []
        for l in range(self.L):
            z = safe_matmul(h_in[None, :], self.W_layers[l]) + self.b_layers[l]
            h = safe_tanh(z[0])
            outs.append([silu(safe_matmul(h[None, :], self.W_heads[l, hidx]) + self.b_heads[l, hidx])[0]
                         for hidx in range(self.H)])
            h_in = h
        return outs

class BayesHeadFuseLR:
    def __init__(self, d: int, V: int, H: int, r: int, rho: float = 0.7, tau0: float = 1e-4, seed: int = 0):
        self.r, self.V, self.H = r, V, H
        rng = np.random.default_rng(seed)
        self.A = rng.normal(0, 0.05, size=(d, r)).astype(np.float32) / np.sqrt(max(1, d))
        self.B = rng.normal(0, 0.05, size=(r, V)).astype(np.float32) / np.sqrt(max(1, r))
        hid = max(16, d // 4)
        self.W1 = rng.normal(0, 0.05, size=(H, d, hid)).astype(np.float32) / np.sqrt(max(1, d))
        self.b1 = np.zeros((H, hid), dtype=np.float32)
        self.W2 = rng.normal(0, 0.05, size=(H, hid, r)).astype(np.float32) / np.sqrt(max(1, hid))
        self.b2 = np.zeros((H, r), dtype=np.float32)
        self.rho = float(rho); self.tau0 = float(tau0)

    def fuse(self, head_outs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        ys = []; taus = []
        for h, o in enumerate(head_outs):
            y = safe_matmul(o[None, :], self.A)[0]                    # [r]
            h1 = silu(safe_matmul(o[None, :], self.W1[h]) + self.b1[h])[0]  # [hid]
            sigma = np.exp(np.clip(safe_matmul(h1[None, :], self.W2[h]) + self.b2[h], -6.0, 6.0))[0] + 1e-3  # [r]
            tau = np.power(1.0 / np.maximum(sigma**2, 1e-6), self.rho) # tempered precision
            ys.append(y * tau); taus.append(tau)
        tau_sum = np.sum(np.stack(taus, 0), axis=0) + self.tau0
        mu = np.sum(np.stack(ys, 0), axis=0) / np.maximum(tau_sum, 1e-6)
        var = 1.0 / np.maximum(tau_sum, 1e-6)
        return mu.astype(np.float32), var.astype(np.float32), taus

    def logits_from_mu(self, mu_r: np.ndarray) -> np.ndarray:
        return safe_matmul(mu_r[None, :], self.B)[0]

    def variance_full_logits(self, var_r: np.ndarray) -> np.ndarray:
        # var[logit_j] = sum_r var_r[r] * B[r,j]^2
        B2 = (self.B * self.B)                  # [r,V]
        return safe_matmul(var_r[None, :], B2)[0]  # [V]

    def state_dict(self) -> Dict[str, np.ndarray]:
        return dict(A=self.A.copy(), B=self.B.copy(), W1=self.W1.copy(), b1=self.b1.copy(),
                    W2=self.W2.copy(), b2=self.b2.copy(), rho=np.array(self.rho, dtype=np.float32),
                    tau0=np.array(self.tau0, dtype=np.float32))

    def load_state(self, state: Dict[str, np.ndarray]):
        self.A = state["A"].astype(np.float32)
        self.B = state["B"].astype(np.float32)
        self.W1 = state["W1"].astype(np.float32); self.b1 = state["b1"].astype(np.float32)
        self.W2 = state["W2"].astype(np.float32); self.b2 = state["b2"].astype(np.float32)
        self.rho = float(np.array(state["rho"]).item()); self.tau0 = float(np.array(state["tau0"]).item())

class ReliabilityTracker:
    def __init__(self, L: int, H: int, ema: float = 0.7):
        self.L, self.H = L, H; self.ema = float(ema)
        self.x = np.zeros((L, H), dtype=np.float32)  # dominance
        self.y = np.zeros((L, H), dtype=np.float32)  # reliability (precision)
        self.n = np.zeros((L, H), dtype=np.int32)

    def update(self, l: int, h: int, dom: float, rel: float):
        self.x[l, h] = self.ema * self.x[l, h] + (1 - self.ema) * float(dom)
        self.y[l, h] = self.ema * self.y[l, h] + (1 - self.ema) * float(rel)
        self.n[l, h] += 1

    def layer_stats(self, stab_thr: float = 1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # normalize dominance per layer
        xz = np.zeros_like(self.x)
        for l in range(self.L):
            s = np.std(self.x[l]) + 1e-6
            xz[l] = (self.x[l] - np.mean(self.x[l])) / s
        # stabilizers: high reliability (y large), explorers: low
        stab = (self.y > stab_thr).astype(np.float32)
        expl = (self.y <= stab_thr).astype(np.float32)
        return xz, stab, expl

    def state_dict(self) -> Dict[str, np.ndarray]:
        return dict(x=self.x.copy(), y=self.y.copy(), n=self.n.copy())

    def load_state(self, state: Dict[str, np.ndarray]):
        self.x = state["x"].astype(np.float32)
        self.y = state["y"].astype(np.float32)
        self.n = state["n"].astype(np.int32)

@dataclass
class TeacherCfg:
    T0: float = 1.0; lam: float = 1.0; gamma: float = 1.0; Tmin: float = 0.7; Tmax: float = 1.8
    topk: int = 40
    danger_rho_thr: float = 0.8
    rag_m: int = 8
    rag_w: float = 1.5

class SpiralTeacher:
    def __init__(self, V: int, d: int, H: int, L: int, r: int, P: np.ndarray, cfg: TeacherCfg, seed: int = 0):
        self.V, self.d, self.H, self.L, self.r = V, d, H, L, r
        self.P = P; self.cfg = cfg
        self.trigram_map: Optional[Dict[Tuple[int, int], np.ndarray]] = bootstrap_trigram_from_bigram(P, seed=seed)
        self.lm = ToyDeepLM(V, d, H, L, seed=seed)
        self.fusers = [BayesHeadFuseLR(d, V, H, r, rho=0.7, tau0=1e-4, seed=seed + l) for l in range(L)]
        # share B across layers
        B0 = self.fusers[0].B
        for f in self.fusers[1:]: f.B = B0
        self.B = B0
        self.rel = ReliabilityTracker(L, H, ema=0.7)
        # rag source weights (per vocab id) for SHAP attenuation
        self.rag_src_weight = np.ones((V,), dtype=np.float32)

    def _rag_rank_mu_delta(self, indices: np.ndarray, scores: np.ndarray) -> Optional[np.ndarray]:
        if indices.size == 0: return None
        B_J = self.B[:, indices]  # [r,m]
        M = safe_matmul(B_J, B_J.T) + 1e-4 * np.eye(self.r, dtype=np.float32)
        rhs = safe_matmul(B_J, scores.astype(np.float32))  # [r]
        # Solve M * mu_delta = rhs
        try:
            mu_delta = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            mu_delta = np.linalg.pinv(M) @ rhs
        return mu_delta.astype(np.float32)

    def _rag_shap(self, indices: np.ndarray, scores: np.ndarray, y_rel_layer: np.ndarray) -> Dict[int, float]:
        # Simple per-source attribution: |score_j| * mean(y_rel_layer)
        if indices.size == 0: return {}
        rel_scale = float(np.clip(np.mean(y_rel_layer), 0.1, 10.0))
        contrib = {}
        for j_idx, sc in zip(indices.tolist(), scores.tolist()):
            contrib[j_idx] = contrib.get(j_idx, 0.0) + abs(float(sc)) * rel_scale
        return contrib

    def forward_batch(self, tokens_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        Bn = tokens_batch.shape[0]
        logits = np.zeros((Bn, self.V), dtype=np.float32)
        Ts = np.zeros((Bn, 1), dtype=np.float32)
        meanVars = np.zeros((Bn,), dtype=np.float32)

        layer_hazard = np.zeros((self.L,), dtype=np.float32)
        keepk_suggest = np.zeros((self.L,), dtype=np.int32)

        # reliability accumulators for RAG SHAP
        rag_attr: Dict[int, float] = {}

        for b in range(Bn):
            HPL = self.lm.encode_heads_per_layer(tokens_batch[b])
            mu_l = []; var_l = []
            # fuse per layer
            for l in range(self.L):
                mu, var, taus = self.fusers[l].fuse(HPL[l])
                mu_l.append(mu); var_l.append(var)
                # log head reliability per layer from precision norm
                for h in range(self.H):
                    dom = float(np.linalg.norm(HPL[l][h])) / (np.sqrt(self.d) + 1e-6)
                    rel = float(np.mean(taus[h]))
                    self.rel.update(l, h, dom, rel)

            # posterior across layers (simple chain)
            mu_post, var_post = mu_l[-1], var_l[-1]
            # state regularization could be added here

            # logits before RAG
            logits_b = self.fusers[-1].logits_from_mu(mu_post)

            # RAG delta on logits (bCSR) with trigram fallback to bigram
            delta = rag_bcsr_ngram(tokens_batch[b][None, :], self.V, self.P,
                                   trigram_map=self.trigram_map, m=self.cfg.rag_m, w=self.cfg.rag_w)
            tag, indptr, indices, data = delta
            s, e = int(indptr[0]), int(indptr[1])
            inds = indices[s:e]; vals = data[s:e]
            # apply source weights
            if inds.size > 0:
                vals = vals * self.rag_src_weight[inds]
                logits_b[inds] += vals
                # rank delta for audit + SHAP
                mu_d = self._rag_rank_mu_delta(inds, vals)
                if mu_d is not None:
                    mu_post = (mu_post + mu_d).astype(np.float32)
                    # SHAP-ish attribution with reliability of last layer
                    _, stab, _ = self.rel.layer_stats()
                    y_rel = (stab[-1] + 1.0)  # crude proxy: stabilizers weight=2, others=1
                    shap = self._rag_shap(inds, vals, y_rel_layer=y_rel)
                    for k, v in shap.items():
                        rag_attr[k] = rag_attr.get(k, 0.0) + float(v)
                    # recompute logits after rank injection
                    logits_b = self.fusers[-1].logits_from_mu(mu_post)

            # variance on top-k
            var_logits = self.fusers[-1].variance_full_logits(var_post)  # [V]
            k = min(self.cfg.topk, self.V)
            idx = np.argpartition(logits_b, -k)[-k:]
            mean_var = float(np.mean(var_logits[idx]))
            T = float(np.clip(self.cfg.T0 * (1.0 + self.cfg.lam * mean_var) ** self.cfg.gamma, self.cfg.Tmin, self.cfg.Tmax))

            logits[b] = logits_b; Ts[b, 0] = T; meanVars[b] = mean_var

        # layer hazard & keep-k suggestion from reliability
        xz, stab, expl = self.rel.layer_stats(stab_thr=1.0)
        for l in range(self.L):
            expl_frac = float(np.mean(expl[l]))
            stab_frac = float(np.mean(stab[l]))
            rel_level = float(np.mean(self.rel.y[l]))
            dom_std = float(np.std(self.rel.x[l]))
            raw = 0.3 * (expl_frac - stab_frac) + 0.3 * dom_std + 0.4 * np.tanh(rel_level)
            rho_layer = float(np.clip(raw, 0.0, 1.0))
            layer_hazard[l] = rho_layer
            keepk_suggest[l] = int(np.clip( (1.0 + rho_layer) * (self.fusers[0].r // 2), 2, self.fusers[0].r))

        # attenuate rag sources with high attribution
        if len(rag_attr) > 0:
            keys = list(rag_attr.keys()); vals = np.array([rag_attr[k] for k in keys], dtype=np.float32)
            vals = vals / (vals.max() + 1e-6)
            decay = np.exp(-1.5 * vals)  # higher attribution ⇒ stronger decay
            self.rag_src_weight[keys] = np.clip(self.rag_src_weight[keys] * decay, 0.2, 1.0)

        meta = dict(meanVar=float(meanVars.mean()), layer_hazard=layer_hazard, keepk_suggest=keepk_suggest)
        return logits, Ts, meanVars, meta

    def state_dict(self) -> Dict[str, Any]:
        return dict(
            P=self.P.copy(),
            trigram_map=pack_trigram(self.trigram_map, self.V),
            lm=self.lm.state_dict(),
            fusers=[f.state_dict() for f in self.fusers],
            rel=self.rel.state_dict(),
            rag_src_weight=self.rag_src_weight.copy(),
            cfg=asdict(self.cfg),
        )

    def load_state(self, state: Dict[str, Any]):
        self.P = state["P"].astype(np.float32)
        trig = state.get("trigram_map", None)
        trig_map = unpack_trigram(trig) if isinstance(trig, dict) else None
        if trig_map is None and trig is not None and isinstance(trig, dict):
            # legacy tuple-keyed dict fallback
            trig_map = {tuple(k): np.array(v, dtype=np.float32) for k, v in trig.items()}
        self.trigram_map = trig_map
        self.cfg = TeacherCfg(**state.get("cfg", asdict(self.cfg)))
        self.lm.load_state(state["lm"])
        for f, saved in zip(self.fusers, state["fusers"]):
            f.load_state(saved)
        # ensure B is shared after load
        B0 = self.fusers[0].B
        for f in self.fusers[1:]:
            f.B = B0
        self.B = B0
        self.rel.load_state(state["rel"])
        self.rag_src_weight = state.get("rag_src_weight", np.ones((self.V,), dtype=np.float32)).astype(np.float32)

# ------------------------------
# Student: Rank-K blocks + RankParamHead + UncertaintyLR + T_S(Σ) backprop (approx)
# ------------------------------
class Linear:
    def __init__(self, din: int, dout: int, rng):
        self.W = (rng.normal(0, 0.02, size=(din, dout)) / np.sqrt(max(1, din))).astype(np.float32)
        self.b = np.zeros((dout,), dtype=np.float32)
        self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b); self._x = None
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x.copy()
        y = safe_matmul(x, self.W) + self.b
        return y
    def backward(self, dy: np.ndarray) -> np.ndarray:
        x = self._x
        x2 = x.reshape(-1, x.shape[-1]); dy2 = dy.reshape(-1, dy.shape[-1])
        # accumulate full gradient (already normalized upstream by loss/B)
        self.dW += safe_matmul(x2.T, dy2)
        self.db += dy2.mean(axis=0)
        dx = safe_matmul(dy, self.W.T)
        return dx
    def step(self, lr: float):
        # clip grads to avoid explosions while keeping movement
        dWc = np.clip(self.dW, -1.0, 1.0)
        dbc = np.clip(self.db, -1.0, 1.0)
        self.W -= lr * dWc; self.b -= lr * dbc
        self.dW.fill(0.0); self.db.fill(0.0)

class Embedding:
    def __init__(self, V: int, d: int, rng):
        self.W = (rng.normal(0, 0.02, size=(V, d)) / np.sqrt(max(1, d))).astype(np.float32)
        self.dW = np.zeros_like(self.W); self._idx = None
    def forward(self, ids: np.ndarray) -> np.ndarray:
        self._idx = ids.copy()
        return self.W[ids]
    def backward(self, dE: np.ndarray):
        B, T, d = dE.shape
        ids = self._idx.reshape(-1)
        dflat = dE.reshape(B * T, d)
        # accumulate per token id (sparse)
        for i, idx in enumerate(ids):
            self.dW[idx] += dflat[i]
    def step(self, lr: float):
        self.W -= lr * self.dW; self.dW.fill(0.0)

class SiLU:
    def __init__(self): self._x = None
    def forward(self, x: np.ndarray) -> np.ndarray: self._x = x; return silu(x)
    def backward(self, dy: np.ndarray) -> np.ndarray:
        x = self._x; s = 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
        return dy * (s * (1.0 + x * (1.0 - s)))

class RankKAttention:
    def __init__(self, d: int, k: int, rng):
        self.q = Linear(d, k, rng); self.k = Linear(d, k, rng); self.v = Linear(d, d, rng)
        self.kdim = k; self._cache = None
    def forward(self, x: np.ndarray, keep_k: int) -> Tuple[np.ndarray, np.ndarray]:
        keep_k = int(np.clip(keep_k, 1, self.kdim))
        Q = self.q.forward(x); K = self.k.forward(x); V = self.v.forward(x)
        Qm = Q[:, :, :keep_k]; Km = K[:, :, :keep_k]
        S = safe_matmul(Qm, np.transpose(Km, (0, 2, 1))) / np.sqrt(max(1, keep_k))
        W = softmax_rows(S)
        H = safe_matmul(W, V)
        self._cache = (x, Q, K, V, W, keep_k)
        return H, W
    def backward(self, dH: np.ndarray) -> np.ndarray:
        x, Q, K, V, W, keep_k = self._cache
        dV = safe_matmul(np.transpose(W, (0, 2, 1)), dH)
        dW = safe_matmul(dH, np.transpose(V, (0, 2, 1)))
        sumdw = np.sum(dW * W, axis=-1, keepdims=True)
        dS = W * (dW - sumdw)
        dQm = safe_matmul(dS, K[:, :, :keep_k]) / np.sqrt(max(1, keep_k))
        dKm = safe_matmul(np.transpose(dS, (0, 2, 1)), Q[:, :, :keep_k]) / np.sqrt(max(1, keep_k))
        dQ = np.zeros_like(Q); dK = np.zeros_like(K)
        dQ[:, :, :keep_k] = dQm; dK[:, :, :keep_k] = dKm
        dx = self.q.backward(dQ) + self.k.backward(dK) + self.v.backward(dV)
        return dx
    def step(self, lr: float):
        self.q.step(lr); self.k.step(lr); self.v.step(lr)

class FFN:
    def __init__(self, d: int, rng):
        self.fc1 = Linear(d, 4 * d, rng); self.act = SiLU(); self.fc2 = Linear(4 * d, d, rng)
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.fc2.forward(self.act.forward(self.fc1.forward(x)))
    def backward(self, dy: np.ndarray) -> np.ndarray:
        dx2 = self.fc2.backward(dy); dx1 = self.act.backward(dx2); return self.fc1.backward(dx1)
    def step(self, lr: float):
        self.fc1.step(lr); self.fc2.step(lr)

class SpiralBlock:
    def __init__(self, d: int, k: int, rng):
        self.attn = RankKAttention(d, k, rng); self.ffn = FFN(d, rng)
    def forward(self, x: np.ndarray, keep_k: int) -> Tuple[np.ndarray, np.ndarray]:
        H, W = self.attn.forward(x, keep_k=keep_k); x1 = x + H; y = self.ffn.forward(x1); return x1 + y, W
    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx1 = dout + self.ffn.backward(dout); return dx1 + self.attn.backward(dx1)
    def step(self, lr: float):
        self.attn.step(lr); self.ffn.step(lr)

class RankParamHead:
    def __init__(self, d: int, r: int, rng):
        self.mean = Linear(d, r, rng); self.logtau = Linear(d, r, rng)
        self._last = None
    def forward(self, h_last: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # h_last: [B,T,d] → use last token state
        last = h_last[:, -1, :]
        mu = self.mean.forward(last)                       # [B,r]
        lt = np.clip(self.logtau.forward(last), -6.0, 6.0) # [B,r]
        tau = np.exp(lt) + 1e-6
        self._last = dict(last=last, lt=lt, tau=tau)
        return mu.astype(np.float32), tau.astype(np.float32)
    def backward(self, dmu: np.ndarray, dtau: np.ndarray) -> np.ndarray:
        # dmu, dtau: [B,r]
        # propagate to last-token features
        dlast = self.mean.backward(dmu) + self.logtau.backward(dtau * np.exp(self._last["lt"]))
        B, d = dlast.shape
        dh = np.zeros((B, 1, d), dtype=np.float32)
        dh[:, -1, :] = dlast
        return dh
    def step(self, lr: float):
        self.mean.step(lr); self.logtau.step(lr)

class UncertaintyLR:
    def __init__(self, L: int, init: float = 0.0):
        self.rho = np.full((L,), float(init), dtype=np.float32)
    def eta(self, base_lr: float) -> np.ndarray:
        # η = base_lr * softplus(-ρ); softplus(x)=log(1+e^x)
        sp = np.log1p(np.exp(-np.clip(self.rho, -20.0, 20.0)))
        return (base_lr * sp).astype(np.float32)

@dataclass
class StudentCfg:
    L: int = 4; d: int = 96; k: int = 24; V: int = 128; r: int = 64
    base_lr: float = 5e-2; head_lr: float = 5e-2; emb_lr: float = 5e-2
    grad_target: float = 0.1
    seed: int = 123

class StudentV9:
    def __init__(self, cfg: StudentCfg, shared_B: np.ndarray):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.embed = Embedding(cfg.V, cfg.d, rng)
        self.blocks = [SpiralBlock(cfg.d, cfg.k, rng) for _ in range(cfg.L)]
        self.rank_head = RankParamHead(cfg.d, cfg.r, rng)
        self.B = shared_B.astype(np.float32)  # [r,V]
        self.u = UncertaintyLR(cfg.L, init=0.0)
        self._keepk = np.full((cfg.L,), max(2, cfg.k // 2), dtype=np.int32)

    # ------------------ persistence ------------------
    def _linear_state(self, lin: Linear) -> Dict[str, np.ndarray]:
        return dict(W=lin.W.copy(), b=lin.b.copy())

    def state_dict(self) -> Dict[str, Any]:
        blocks = []
        for blk in self.blocks:
            blocks.append(dict(
                attn=dict(q=self._linear_state(blk.attn.q),
                          k=self._linear_state(blk.attn.k),
                          v=self._linear_state(blk.attn.v)),
                ffn=dict(fc1=self._linear_state(blk.ffn.fc1),
                         fc2=self._linear_state(blk.ffn.fc2)),
            ))
        return dict(
            cfg=asdict(self.cfg),
            B=self.B.copy(),
            keepk=self._keepk.copy(),
            rho=self.u.rho.copy(),
            embed=dict(W=self.embed.W.copy()),
            blocks=blocks,
            rank_head=dict(mean=self._linear_state(self.rank_head.mean),
                           logtau=self._linear_state(self.rank_head.logtau)),
        )

    def load_state(self, state: Dict[str, Any]):
        self.cfg = StudentCfg(**state["cfg"])
        self.B = state["B"].astype(np.float32)
        self._keepk = state["keepk"].astype(np.int32)
        self.u.rho = state["rho"].astype(np.float32)
        self.embed.W = state["embed"]["W"].astype(np.float32)
        for blk, sd in zip(self.blocks, state["blocks"]):
            for lin, ld in zip((blk.attn.q, blk.attn.k, blk.attn.v), (sd["attn"]["q"], sd["attn"]["k"], sd["attn"]["v"])):
                lin.W = ld["W"].astype(np.float32); lin.b = ld["b"].astype(np.float32)
            blk.ffn.fc1.W = sd["ffn"]["fc1"]["W"].astype(np.float32); blk.ffn.fc1.b = sd["ffn"]["fc1"]["b"].astype(np.float32)
            blk.ffn.fc2.W = sd["ffn"]["fc2"]["W"].astype(np.float32); blk.ffn.fc2.b = sd["ffn"]["fc2"]["b"].astype(np.float32)
        self.rank_head.mean.W = state["rank_head"]["mean"]["W"].astype(np.float32)
        self.rank_head.mean.b = state["rank_head"]["mean"]["b"].astype(np.float32)
        self.rank_head.logtau.W = state["rank_head"]["logtau"]["W"].astype(np.float32)
        self.rank_head.logtau.b = state["rank_head"]["logtau"]["b"].astype(np.float32)
        # keepk size might mismatch if cfg changed; clamp safely
        self._keepk = np.clip(self._keepk, 2, self.cfg.k)

    def set_keepk_layerwise(self, keepk: np.ndarray):
        # Accept keepk arrays shorter/longer than student depth and broadcast safely.
        kk = np.clip(keepk.astype(np.int32), 2, self.cfg.k)
        if kk.size < self.cfg.L:
            pad_val = kk[-1] if kk.size > 0 else max(2, self.cfg.k // 2)
            kk = np.concatenate([kk, np.full((self.cfg.L - kk.size,), pad_val, dtype=np.int32)])
        self._keepk = kk[: self.cfg.L].copy()

    def forward(self, tokens: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        x = self.embed.forward(tokens)  # [B,T,d]
        h = x; attn_maps = []
        for l, blk in enumerate(self.blocks):
            h, W = blk.forward(h, keep_k=int(self._keepk[l]))
            attn_maps.append(W)
        mu, tau = self.rank_head.forward(h)  # [B,r], [B,r]
        logits = safe_matmul(mu, self.B)     # [B,V]
        Sigma = 1.0 / np.maximum(tau, 1e-6)  # [B,r]
        return logits, dict(h=h, mu=mu, tau=tau, Sigma=Sigma, attn=attn_maps)

    def backward(self, dlogits: np.ndarray, aux: Dict[str, Any], dSigma_from_T: Optional[np.ndarray] = None):
        # dmu from logits
        dmu = safe_matmul(dlogits, self.B.T)    # [B,r]
        dtau = np.zeros_like(aux["tau"])        # [B,r]
        if dSigma_from_T is not None:
            # Sigma = 1/tau ⇒ dL/dtau = dL/dSigma * (-1/tau^2); logtau path multiplies τ
            tau = aux["tau"]
            dTau_direct = (-dSigma_from_T) / np.maximum(tau**2, 1e-8)
            dtau += dTau_direct
        dh_last = self.rank_head.backward(dmu, dtau)  # [B,1,d]
        # expand to full seq: only last token has grad
        Bn, Tn, d = aux["h"].shape
        dh = np.zeros_like(aux["h"]); dh[:, -1, :] = dh_last[:, -1, :]
        for l in range(self.cfg.L - 1, -1, -1):
            dh = self.blocks[l].backward(dh)
        self.embed.backward(dh)

    def step(self):
        # Adapt the effective LR to keep gradient magnitudes near a target.
        g_rms = float(np.mean(self.grad_rms()))
        g_scale = float(np.clip(self.cfg.grad_target / max(g_rms, 1e-6), 0.2, 100.0))
        base = self.cfg.base_lr * g_scale
        etas = self.u.eta(base)
        for l, blk in enumerate(self.blocks):
            blk.step(float(etas[l]))
        self.rank_head.step(self.cfg.head_lr * g_scale)
        self.embed.step(self.cfg.emb_lr * g_scale)
        return etas, g_rms, g_scale

    def grad_rms(self) -> np.ndarray:
        vals = []
        for blk in self.blocks:
            g2 = 0.0; n = 0
            for lin in (blk.attn.q, blk.attn.k, blk.attn.v, blk.ffn.fc1, blk.ffn.fc2):
                g2 += float(np.mean(lin.dW**2) + np.mean(lin.db**2)); n += 2
            vals.append(float(np.sqrt(g2 / max(1, n))))
        return np.array(vals, dtype=np.float32)

# ------------------------------
# Losses + T from Sigma + Backprop wrt T
# ------------------------------
def ce_with_temp(logits: np.ndarray, y: np.ndarray, T: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    z = logits / np.maximum(T, 1e-6); p = softmax_rows(z)
    B = y.shape[0]; oh = np.zeros_like(p); oh[np.arange(B), y] = 1.0
    ce = -np.sum(np.log(np.maximum(p, 1e-12)) * oh) / max(1, B)
    # scale gradient a bit higher to speed learning in the toy setting
    dlogits = GRAD_BOOST * (p - oh) / np.maximum(T, 1e-6)
    dlogits = np.clip(dlogits, -5.0, 5.0)
    return float(ce), dlogits.astype(np.float32), p

def kl_teacher_student(teacher_logits: np.ndarray, student_logits: np.ndarray, T: np.ndarray) -> Tuple[float, np.ndarray]:
    q = softmax_rows(teacher_logits / np.maximum(T, 1e-6))
    p = softmax_rows(student_logits / np.maximum(T, 1e-6))
    B = p.shape[0]
    kl = np.sum(q * (np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(p, 1e-12)))) / max(1, B)
    dlogits = GRAD_BOOST * (p - q) / np.maximum(T, 1e-6)
    dlogits = np.clip(dlogits, -5.0, 5.0)
    return float(kl), dlogits.astype(np.float32)

def T_from_Sigma(B: np.ndarray, Sigma: np.ndarray, logits: np.ndarray,
                 T0: float = 1.0, lam: float = 1.0, gamma: float = 1.0,
                 Tmin: float = 0.7, Tmax: float = 1.8, k: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Sigma: [B,r], B: [r,V], logits: [B,V]
    B2 = (B * B)  # [r,V]
    var_full = safe_matmul(Sigma, B2)  # [B,V]
    k_eff = min(k, logits.shape[1])
    idx = np.argpartition(logits, -k_eff, axis=1)[:, -k_eff:]  # [B,k]
    # gather top-k variances
    Bn = logits.shape[0]
    var_top = np.zeros((Bn, k_eff), dtype=np.float32)
    for b in range(Bn):
        var_top[b] = var_full[b, idx[b]]
    meanVar = var_top.mean(axis=1, keepdims=True)  # [B,1]
    T = T0 * np.power(1.0 + lam * meanVar, gamma)
    T = np.clip(T, Tmin, Tmax).astype(np.float32)
    return T, meanVar.squeeze(-1), var_top

def dCEdT(logits: np.ndarray, y: np.ndarray, T: np.ndarray) -> np.ndarray:
    # ∂CE/∂T ≈ (1/T^2) * (E_p[z] - z_y)
    z = logits / np.maximum(T, 1e-6); p = softmax_rows(z)
    Ez = (p * logits).sum(axis=1, keepdims=True)  # [B,1]
    zy = logits[np.arange(logits.shape[0]), y][:, None]
    d = (Ez - zy) / (np.maximum(T, 1e-6)**2)
    return d.astype(np.float32)  # [B,1]

def dTdSigma(B: np.ndarray, logits: np.ndarray, Sigma: np.ndarray,
             T: np.ndarray, meanVar: np.ndarray, var_top: np.ndarray,
             T0: float = 1.0, lam: float = 1.0, gamma: float = 1.0, k: int = 40,
             Tmin: float = 0.7, Tmax: float = 1.8) -> np.ndarray:
    # ∂T/∂Σ_r = T0 * γ * λ * (1+λ m)^{γ-1} * (1/k) * sum_{j∈topk} B_{rj}^2
    B2 = (B * B)  # [r,V]
    k_eff = var_top.shape[1]
    # pick same topk as in T_from_Sigma
    idx = np.argpartition(logits, -k_eff, axis=1)[:, -k_eff:]  # [B,k]
    Bn, r = Sigma.shape
    avg_B2 = np.zeros((Bn, r), dtype=np.float32)
    for b in range(Bn):
        avg_B2[b] = B2[:, idx[b]].mean(axis=1)  # [r]
    rawT = (T0 * np.power(1.0 + lam * meanVar[:, None], gamma)).astype(np.float32)
    factor = (T0 * gamma * lam * np.power(1.0 + lam * meanVar[:, None], max(0.0, gamma - 1.0))).astype(np.float32)
    dT_dSigma = (factor / max(1, k_eff)) * avg_B2  # [B,r]
    clamped = (rawT <= (Tmin + 1e-6)) | (rawT >= (Tmax - 1e-6))
    if clamped.any():
        dT_dSigma[clamped[:, 0]] = 0.0
    return dT_dSigma.astype(np.float32)

# ------------------------------
# Trainer
# ------------------------------
@dataclass
class TrainCfg:
    steps: int = 30; batch: int = 16; ctx_len: int = 8
    data_path: Optional[str] = None  # optional npy file of token ids for local data
    valid_frac: float = 0.1
    eval_every: int = 10
    max_eval_windows: int = 20000
    # student T hyper
    T0: float = 1.0; lam: float = 1.0; gamma: float = 1.0; Tmin: float = 0.7; Tmax: float = 1.8; topk: int = 40
    # distillation weight (single membrane for compactness)
    lam_distil: float = 0.1
    distil_warmup: int = 5     # steps before blending in KL
    ce_weight: float = 1.0
    kl_weight: float = 1.0
    # reliability → rho nudging
    lr_k_stab: float = 0.02; lr_k_expl: float = 0.01
    # hazard → partial optimize
    keepk_boost: int = 2; rho_boost: float = 0.0
    # T backprop switch & scale
    backprop_T: bool = True; T_grad_scale: float = 0.1

class SpiralV9:
    def __init__(self, V: int = 128, d: int = 96, H: int = 4, L: int = 4, r: int = 64,
                 seed: int = 0, student_cfg: Optional[StudentCfg] = None, teacher_cfg: Optional[TeacherCfg] = None):
        self.V, self.d, self.H, self.L, self.r = V, d, H, L, r
        self.seed = seed
        if student_cfg is not None and student_cfg.V != V:
            raise ValueError("StudentCfg.V must match SpiralV9 V to keep vocabulary consistent")
        self.P = make_bigram(V, seed=7 + seed)
        self.teacher_cfg = teacher_cfg or TeacherCfg()
        self.teacher = SpiralTeacher(V, d, H, L, r, self.P, self.teacher_cfg, seed=42 + seed)
        cfg = student_cfg or StudentCfg(L=max(4, L), d=d, k=max(16, d // 4), V=V, r=r, seed=123 + seed)
        self.student = StudentV9(cfg, self.teacher.B)
        self.tokenizer: Optional[SpiralTokenizer] = None

    # ------------------ persistence + inference ------------------
    def set_tokenizer(self, tokenizer: SpiralTokenizer):
        self.tokenizer = tokenizer

    def save_weights(self, path: str):
        state = dict(
            student_state=self.student.state_dict(),
            teacher_meta=dict(V=self.V, vocab_size=self.V, d=self.d, H=self.H, L=self.L, r=self.r, seed=self.seed),
            teacher_state=self.teacher.state_dict(),
        )
        if self.tokenizer is not None:
            state["tokenizer_state"] = self.tokenizer.state_dict()
        np.savez_compressed(path, **{k: np.array([v], dtype=object) for k, v in state.items()})

    @classmethod
    def load(cls, path: str) -> "SpiralV9":
        data = np.load(path, allow_pickle=True)
        student_raw = data["student_state"]
        student_state = student_raw[0] if isinstance(student_raw, np.ndarray) else student_raw
        if hasattr(student_state, "item") and not isinstance(student_state, dict):
            student_state = student_state.item()
        meta_raw = data["teacher_meta"] if "teacher_meta" in data else {}
        meta = meta_raw[0] if isinstance(meta_raw, np.ndarray) else meta_raw
        if hasattr(meta, "item") and not isinstance(meta, dict):
            meta = meta.item()
        teacher_raw = data["teacher_state"] if "teacher_state" in data else None
        teacher_state = None
        if teacher_raw is not None:
            teacher_state = teacher_raw[0] if isinstance(teacher_raw, np.ndarray) else teacher_raw
            if hasattr(teacher_state, "item") and not isinstance(teacher_state, dict):
                teacher_state = teacher_state.item()
        cfg = StudentCfg(**student_state["cfg"])
        V_meta = meta.get("vocab_size", meta.get("V", cfg.V))
        V = V_meta; d = meta.get("d", cfg.d); H = meta.get("H", 4); L = meta.get("L", cfg.L); r = meta.get("r", cfg.r)
        base_seed = meta.get("seed", 0)
        eng = cls(V=V, d=d, H=H, L=L, r=r, seed=base_seed, student_cfg=cfg)
        eng.student.load_state(student_state)
        if teacher_state is not None:
            eng.teacher.load_state(teacher_state)
            eng.teacher_cfg = eng.teacher.cfg
        if eng.student.cfg.V != eng.V:
            raise ValueError(f"Loaded student vocab ({eng.student.cfg.V}) does not match engine V ({eng.V})")
        # assert shapes before sharing projection
        assert eng.student.B.shape == eng.teacher.fusers[0].B.shape, "B shape mismatch"
        assert eng.student.cfg.r == eng.student.B.shape[0], "student r mismatch"
        assert eng.V == eng.student.B.shape[1], "V mismatch"
        # align teacher logits with loaded student projection (B is shared)
        eng.teacher.B = eng.student.B
        for f in eng.teacher.fusers:
            f.B = eng.student.B
        if "tokenizer_state" in data:
            tok_raw = data["tokenizer_state"]
            tok_state = tok_raw[0] if isinstance(tok_raw, np.ndarray) else tok_raw
            if hasattr(tok_state, "item") and not isinstance(tok_state, dict):
                tok_state = tok_state.item()
            eng.tokenizer = SpiralTokenizer.from_state(tok_state)
            if eng.tokenizer.vocab_size != eng.V:
                raise ValueError(f"Tokenizer vocab ({eng.tokenizer.vocab_size}) does not match model V ({eng.V}); ensure saved vocab is reused.")
        return eng

    def generate(self, prompt_ids: np.ndarray, max_new_tokens: int = 32,
                 temperature: float = 1.0, top_p: float = 0.9, top_k: Optional[int] = None,
                 rng_seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(rng_seed)
        tokens = prompt_ids.astype(np.int64).reshape(1, -1)
        eos_id = getattr(self.tokenizer, "eos_id", 3)
        for _ in range(max_new_tokens):
            logits, _ = self.student.forward(tokens)
            logits = logits / max(temperature, 1e-4)
            probs = softmax_rows(logits)
            p = probs[0]
            # nucleus sampling with optional top-k truncation before top-p
            sorted_idx = np.argsort(p)[::-1]
            if top_k is not None and top_k > 0:
                sorted_idx = sorted_idx[:top_k]
            sorted_p = p[sorted_idx]
            cdf = np.cumsum(sorted_p)
            keep_len = int(np.searchsorted(cdf, top_p, side="right"))
            keep_len = max(1, keep_len)
            keep = sorted_idx[:keep_len]
            keep_p = p[keep] / np.maximum(p[keep].sum(), EPS)
            chosen_idx = int(rng.choice(keep, p=keep_p))
            tokens = np.concatenate([tokens, np.array([[chosen_idx]], dtype=np.int64)], axis=1)
            if chosen_idx == eos_id:
                break
        return tokens[0]

    def generate_text(self, prompt: str, tokenizer: SpiralTokenizer, max_new_tokens: int = 32,
                      temperature: float = 1.0, top_p: float = 0.9, top_k: Optional[int] = None,
                      rng_seed: Optional[int] = None) -> str:
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        gen = self.generate(ids, max_new_tokens=max_new_tokens,
                            temperature=temperature, top_p=top_p, top_k=top_k, rng_seed=rng_seed)
        return tokenizer.decode(gen)

    def _train_valid_split(self, tokens: np.ndarray, ctx_len: int, valid_frac: float,
                           bos_id: int, eos_id: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if tokens is None or tokens.shape[0] < ctx_len + 2:
            return tokens, None
        N = tokens.shape[0]
        vf = float(np.clip(valid_frac, 0.0, 0.9))
        split = int(max(ctx_len + 1, min(N - (ctx_len + 1), round(N * (1.0 - vf)))))
        train_tokens = tokens[:split].copy()
        valid_tokens = tokens[split:].copy()
        # ensure BOS/EOS boundaries are respected by keeping the boundary token at the start of valid split
        if valid_tokens.size > 0 and train_tokens.size > 0 and train_tokens[-1] == bos_id:
            train_tokens = train_tokens[:-1]
        if valid_tokens.size > 0 and valid_tokens[0] == eos_id:
            valid_tokens = valid_tokens[1:]
        return train_tokens, valid_tokens if valid_tokens.size > 0 else None

    def _eval_split(self, ctxs: np.ndarray, ys: np.ndarray, batch: int) -> Optional[Tuple[float, float]]:
        if ctxs.shape[0] == 0:
            return None
        total_loss = 0.0; total = 0; correct = 0
        for start in range(0, ctxs.shape[0], batch):
            ctx_b = ctxs[start:start+batch]; y_b = ys[start:start+batch]
            logits, _ = self.student.forward(ctx_b)
            p = softmax_rows(logits)
            nll = -np.log(np.maximum(p[np.arange(y_b.shape[0]), y_b], 1e-12))
            total_loss += float(nll.sum())
            total += y_b.shape[0]
            correct += int((np.argmax(p, axis=1) == y_b).sum())
        if total == 0:
            return None
        ppl = float(np.exp(total_loss / max(1, total)))
        acc = float(correct) / max(1, total)
        return ppl, acc

    def train(self, cfg: TrainCfg, data_tokens: Optional[np.ndarray] = None) -> List[Dict[str, float]]:
        rng = np.random.default_rng(0)
        logs = []
        bos_id = getattr(self.tokenizer, "bos_id", 2)
        eos_id = getattr(self.tokenizer, "eos_id", 3)
        train_tokens = None; valid_tokens = None
        eval_train_ctx = eval_train_y = eval_valid_ctx = eval_valid_y = None
        eval_rng = np.random.default_rng(1234)
        if data_tokens is not None:
            flat = data_tokens.astype(np.int64).reshape(-1)
            train_tokens, valid_tokens = self._train_valid_split(flat, ctx_len=cfg.ctx_len,
                                                                 valid_frac=cfg.valid_frac,
                                                                 bos_id=bos_id, eos_id=eos_id)
            if train_tokens is None or train_tokens.shape[0] < cfg.ctx_len + 1:
                raise ValueError("Provided data_tokens are too short for the requested ctx_len")
            self.P = bigram_from_tokens(train_tokens, self.V, bos_id=bos_id, eos_id=eos_id)
            self.teacher.P = self.P
            self.teacher.trigram_map = trigram_from_tokens(train_tokens, self.V, bos_id=bos_id, eos_id=eos_id)
            eval_train_ctx, eval_train_y = build_ctx_target_pairs(train_tokens, ctx_len=cfg.ctx_len, V=self.V,
                                                                  bos_id=bos_id, eos_id=eos_id,
                                                                  max_windows=cfg.max_eval_windows, rng=eval_rng)
            if valid_tokens is not None and valid_tokens.shape[0] >= cfg.ctx_len + 1:
                eval_valid_ctx, eval_valid_y = build_ctx_target_pairs(valid_tokens, ctx_len=cfg.ctx_len, V=self.V,
                                                                      bos_id=bos_id, eos_id=eos_id,
                                                                      max_windows=cfg.max_eval_windows, rng=eval_rng)
        # initial keepk from teacher suggestion
        logits_dummy, Ts_dummy, mv_dummy, meta = self.teacher.forward_batch(np.zeros((1, cfg.ctx_len), dtype=np.int64))
        keepk = np.clip(meta["keepk_suggest"], 2, self.student.cfg.k).astype(np.int32)
        self.student.set_keepk_layerwise(keepk)
        assert self.student.B.shape == (self.student.cfg.r, self.V), "student B shape mismatch with cfg/V"

        for step in range(cfg.steps):
            Bn = cfg.batch
            if train_tokens is None:
                ctx, y = sample_batch_bigram(self.P, ctx_len=cfg.ctx_len, B=Bn, rng=rng)
            else:
                ctx, y = sample_batch_from_array(train_tokens, ctx_len=cfg.ctx_len, B=Bn, rng=rng, V=self.V,
                                                 bos_id=bos_id, eos_id=eos_id)
            embed_prev = self.student.embed.W.copy()

            # teacher
            t_logits, Ts, meanVars, meta = self.teacher.forward_batch(ctx)

            # student forward
            s_logits, aux = self.student.forward(ctx)

            # student T from Sigma
            T_S, meanVar_S, var_top_S = T_from_Sigma(self.student.B, aux["Sigma"], s_logits,
                                                     T0=cfg.T0, lam=cfg.lam, gamma=cfg.gamma,
                                                     Tmin=cfg.Tmin, Tmax=cfg.Tmax, k=cfg.topk)

            # losses
            ce, dlogits_ce, p_s = ce_with_temp(s_logits, y, T_S)
            kl, dlogits_kl = kl_teacher_student(t_logits, s_logits, T_S)
            warm = max(0, step + 1 - cfg.distil_warmup)
            distil_phase = float(np.clip(warm / max(1, cfg.distil_warmup), 0.0, 1.0))
            ce_scale = float(cfg.ce_weight)
            kl_scale = float(cfg.lam_distil * cfg.kl_weight * distil_phase)
            dlogits = ce_scale * dlogits_ce + kl_scale * dlogits_kl

            # dCE/dT * dT/dSigma (optional)
            dSigma_T = None
            if cfg.backprop_T:
                dC = dCEdT(s_logits, y, T_S)                    # [B,1]
                dTdS = dTdSigma(self.student.B, s_logits, aux["Sigma"], T_S, meanVar_S, var_top_S,
                                 T0=cfg.T0, lam=cfg.lam, gamma=cfg.gamma, k=cfg.topk,
                                 Tmin=cfg.Tmin, Tmax=cfg.Tmax)  # [B,r]
                dSigma_T = (cfg.T_grad_scale * dC) * dTdS       # [B,r]

            # backward + step
            self.student.backward(dlogits, aux, dSigma_from_T=dSigma_T)
            embed_gnorm = float(np.linalg.norm(self.student.embed.dW))
            etas, g_rms, g_scale = self.student.step()
            embed_delta = float(np.linalg.norm(self.student.embed.W - embed_prev))

            # reliability → rho nudging
            xz, stab, expl = self.teacher.rel.layer_stats(stab_thr=1.0)
            for l in range(self.student.cfg.L):
                # map L mismatch (teacher L vs student L): clamp index
                l_t = min(l, self.teacher.L - 1)
                stab_ratio = float(np.mean(stab[l_t])); expl_ratio = float(np.mean(expl[l_t]))
                self.student.u.rho[l] += cfg.lr_k_expl * expl_ratio - cfg.lr_k_stab * stab_ratio
                self.student.u.rho[l] = float(np.clip(self.student.u.rho[l], -2.0, 2.0))

            # partial optimization (hazard)
            hazard_teach = meta["layer_hazard"].copy()  # [L_teach] from teacher reliability
            hazard_grad = hazard_teach.copy()           # gradient-derived proxy for control
            grad_layers = self.student.grad_rms()
            for l_t in range(self.teacher.L):
                idx = min(l_t, grad_layers.shape[0]-1)
                hazard_grad[l_t] = float(np.clip(grad_layers[idx] / 0.2, 0.0, 1.0))
            keepk_new = self.student._keepk.copy()
            for l in range(self.student.cfg.L):
                l_t = min(l, self.teacher.L - 1)
                if hazard_grad[l_t] > self.teacher.cfg.danger_rho_thr:
                    keepk_new[l] = int(np.clip(keepk_new[l] + cfg.keepk_boost, 2, self.student.cfg.k))
                    self.student.u.rho[l] += cfg.rho_boost  # reduce LR
            # re-clamp after hazard nudging to prevent runaway shrinkage of learning rate
            self.student.u.rho = np.clip(self.student.u.rho, -2.0, 2.0)
            self.student.set_keepk_layerwise(keepk_new)

            # logging
            rag_w = self.teacher.rag_src_weight
            rag_stats = dict(rag_min=float(rag_w.min()), rag_max=float(rag_w.max()), rag_mean=float(rag_w.mean()))
            log_entry = dict(step=step+1, CE=float(ce), KL=float(kl),
                             ECE=float(np.mean(np.max(p_s, axis=-1) - (np.argmax(p_s, axis=-1) == y).astype(np.float32))),
                             T=float(T_S.mean()), var=float(meanVars.mean()),
                             eta0=float(etas[0]), rho0=float(self.student.u.rho[0]), keepk0=int(self.student._keepk[0]),
                             grad_rms=float(g_rms), grad_scale=float(g_scale),
                             keepk_all=self.student._keepk.tolist(), rho_all=self.student.u.rho.tolist(),
                             hazard=float(hazard_grad.mean()),
                             hazard_teach=float(hazard_teach.mean()), hazard_teach_all=hazard_teach.tolist(),
                             hazard_grad=float(hazard_grad.mean()), hazard_grad_all=hazard_grad.tolist(),
                             embed_dnorm=embed_delta, embed_gnorm=embed_gnorm,
                             ce_scale=ce_scale, kl_scale=kl_scale, distil_phase=distil_phase,
                             **rag_stats)
            if train_tokens is not None and ((step + 1) % cfg.eval_every == 0 or (step + 1) == cfg.steps):
                eval_train = self._eval_split(eval_train_ctx, eval_train_y, cfg.batch) if eval_train_ctx is not None else None
                if eval_train is not None:
                    log_entry["ppl_train"], log_entry["acc_train"] = eval_train
                if eval_valid_ctx is not None and eval_valid_y is not None:
                    eval_valid = self._eval_split(eval_valid_ctx, eval_valid_y, cfg.batch)
                    if eval_valid is not None:
                        log_entry["ppl_valid"], log_entry["acc_valid"] = eval_valid
                if eval_train is not None or eval_valid_ctx is not None:
                    msg = [f"[eval step {step+1}]"]
                    if eval_train is not None:
                        msg.append(f"train ppl={log_entry.get('ppl_train', float('nan')):.3f} acc={log_entry.get('acc_train', float('nan')):.3f}")
                    if eval_valid is not None:
                        msg.append(f"valid ppl={log_entry.get('ppl_valid', float('nan')):.3f} acc={log_entry.get('acc_valid', float('nan')):.3f}")
                    print(" | ".join(msg))
            logs.append(log_entry)
            if (step+1) % 5 == 0:
                print(f"[{step+1:03d}] CE={ce:.4f} KL={kl:.4f} ECE={logs[-1]['ECE']:.4f} | "
                      f"T={T_S.mean():.3f} var~={meanVars.mean():.4f} | η0={etas[0]:.4e} ρ0={self.student.u.rho[0]:+.3f} keepk0={self.student._keepk[0]} | "
                      f"g_rms={g_rms:.4e} g_scale={g_scale:.2f} | hazard_t~{hazard_teach.mean():.3f} hazard_g~{hazard_grad.mean():.3f} | "
                      f"keepk={self.student._keepk.tolist()} rho={self.student.u.rho.tolist()} "
                      f"hazards_t={hazard_teach.tolist()} hazards_g={hazard_grad.tolist()} embed_dnorm={embed_delta:.4e} embed_gnorm={embed_gnorm:.4e} | "
                      f"rag_w[min/mean/max]={rag_stats['rag_min']:.3f}/{rag_stats['rag_mean']:.3f}/{rag_stats['rag_max']:.3f}")
        return logs

# ------------------------------
# Demo / CLI
# ------------------------------
def demo(args=None):
    import argparse
    import json
    def _looks_like_id_list(s: str) -> bool:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        if not parts:
            return False
        for p in parts:
            if p.startswith("-"):
                p = p[1:]
            if not p.isdigit():
                return False
        return True
    p = argparse.ArgumentParser(description="SpiralFullFusion toy demo")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--ctx_len", type=int, default=8)
    p.add_argument("--valid_frac", type=float, default=0.1, help="fraction of tokens reserved for validation")
    p.add_argument("--eval_every", type=int, default=10, help="evaluation interval (steps)")
    p.add_argument("--V", type=int, default=128)
    p.add_argument("--d", type=int, default=96)
    p.add_argument("--H", type=int, default=4)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--r", type=int, default=64)
    p.add_argument("--seed", type=int, default=0, help="base seed for synthetic data/initialization")
    p.add_argument("--data_path", type=str, default=None, help="optional npy file with token ids")
    p.add_argument("--text_path", type=str, default=None, help="optional utf-8 text file to train tokenizer + dataset")
    p.add_argument("--tok_vocab", type=int, default=256, help="tokenizer vocab size target when using text_path")
    p.add_argument("--tok_min_freq", type=int, default=2, help="minimum pair frequency for BPE merges")
    p.add_argument("--tok_word_topk", type=int, default=256, help="top-N full-word atoms to seed the tokenizer")
    p.add_argument("--tok_word_min_freq", type=int, default=2, help="minimum frequency for full-word atoms")
    p.add_argument("--tok_lowercase", action="store_true", help="lowercase text before tokenization")
    p.add_argument("--tok_no_bos", action="store_true", help="disable adding BOS during tokenization")
    p.add_argument("--tok_no_eos", action="store_true", help="disable adding EOS during tokenization")
    p.add_argument("--save_path", type=str, default=None, help="npz path to save trained student/ tokenizer")
    p.add_argument("--load_path", type=str, default=None, help="npz path to load a trained model")
    p.add_argument("--prompt", type=str, default=None, help="text or comma-separated token ids for inference")
    p.add_argument("--max_new_tokens", type=int, default=32, help="number of new tokens to generate during inference")
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    p.add_argument("--top_p", type=float, default=0.9, help="top-p nucleus threshold for sampling")
    p.add_argument("--rng_seed", type=int, default=0, help="seed for sampler reproducibility")
    p.add_argument("--top_k", type=int, default=None, help="optional top-k cutoff before top-p (nucleus) sampling")
    p.add_argument("--log_path", type=str, default=None, help="optional JSON path to save training logs")
    p.add_argument("--max_eval_windows", type=int, default=20000, help="cap evaluation windows to avoid OOM")
    p.add_argument("--distil_warmup", type=int, default=5, help="warmup steps before enabling KL distillation")
    p.add_argument("--ce_weight", type=float, default=1.0, help="scaling for CE loss term")
    p.add_argument("--kl_weight", type=float, default=1.0, help="scaling for KL distillation term")
    p.add_argument("--rag_m", type=int, default=8, help="top-m RAG sources to fuse from n-gram prior")
    p.add_argument("--rag_w", type=float, default=1.5, help="logit boost weight for n-gram RAG suggestions")
    p.add_argument("--override_rag", action="store_true", help="override loaded teacher rag_m/rag_w with CLI values")
    parsed = p.parse_args(args=args)

    data_tokens = None
    vocab_override = parsed.V
    tokenizer: Optional[SpiralTokenizer] = None

    if parsed.data_path:
        data_tokens = np.load(parsed.data_path).astype(np.int64).reshape(-1)
        data_vmax = int(data_tokens.max()) + 1 if data_tokens.size > 0 else 0
        vocab_override = max(vocab_override, data_vmax)
        data_tokens = np.clip(data_tokens, 0, vocab_override - 1)
        print(f"[data] Loaded tokens from {parsed.data_path} (N={data_tokens.size}, vmax={data_vmax})")
    if parsed.text_path and data_tokens is None:
        with open(parsed.text_path, "r", encoding="utf-8") as f:
            text = f.read()
        tok_cfg = TokenizerCfg(vocab_size=parsed.tok_vocab, min_freq=parsed.tok_min_freq,
                               word_topk=parsed.tok_word_topk, word_min_freq=parsed.tok_word_min_freq,
                               add_bos=not parsed.tok_no_bos, add_eos=not parsed.tok_no_eos,
                               lowercase=parsed.tok_lowercase)
        tokenizer = SpiralTokenizer(tok_cfg)
        tokenizer.train_from_text(text)
        data_tokens = tokenizer.encode_corpus(text)
        vocab_override = tokenizer.vocab_size
        data_tokens = np.clip(data_tokens, 0, vocab_override - 1)
        print(f"[tokenizer] Trained tokenizer (vocab={tokenizer.vocab_size}) from {parsed.text_path}; corpus tokens={data_tokens.size}")
    if tokenizer is not None:
        vocab_override = max(vocab_override, tokenizer.vocab_size)

    teacher_cfg = TeacherCfg(rag_m=parsed.rag_m, rag_w=parsed.rag_w)

    if parsed.load_path:
        eng = SpiralV9.load(parsed.load_path)
        tokenizer = tokenizer or eng.tokenizer
    else:
        eng = SpiralV9(V=vocab_override, d=parsed.d, H=parsed.H, L=parsed.L, r=parsed.r,
                       seed=parsed.seed, teacher_cfg=teacher_cfg)

    if parsed.override_rag:
        eng.teacher.cfg.rag_m = parsed.rag_m
        eng.teacher.cfg.rag_w = parsed.rag_w

    if tokenizer is not None and eng.tokenizer is None:
        eng.set_tokenizer(tokenizer)
    elif eng.tokenizer is not None and tokenizer is not None and eng.tokenizer.vocab_size != tokenizer.vocab_size:
        raise ValueError("Tokenizer vocab must match the loaded model vocabulary")
    # ensure data tokens fit the engine vocab (no silent truncation)
    if data_tokens is not None:
        vocab_cap = eng.V
        if data_tokens.size > 0 and int(data_tokens.max()) >= vocab_cap:
            raise ValueError(f"data tokens contain ids >= model vocab ({vocab_cap}); regenerate with the saved tokenizer")
        data_tokens = np.clip(data_tokens, 0, vocab_cap - 1)

    logs: List[Dict[str, float]] = []
    if parsed.steps > 0:
        cfg = TrainCfg(steps=parsed.steps, batch=parsed.batch, ctx_len=parsed.ctx_len,
                       valid_frac=parsed.valid_frac, eval_every=parsed.eval_every, max_eval_windows=parsed.max_eval_windows,
                       T0=1.0, lam=1.0, gamma=1.0, Tmin=0.7, Tmax=1.8, topk=48,
                       lam_distil=0.1, distil_warmup=parsed.distil_warmup,
                       ce_weight=parsed.ce_weight, kl_weight=parsed.kl_weight,
                       lr_k_stab=0.02, lr_k_expl=0.01,
                       keepk_boost=2, rho_boost=0.0, backprop_T=True, T_grad_scale=0.1,
                       data_path=parsed.data_path)
        logs = eng.train(cfg, data_tokens=data_tokens)
        if logs and parsed.log_path:
            with open(parsed.log_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            print(f"[logs] Saved training logs to {parsed.log_path}")
        if parsed.save_path:
            eng.save_weights(parsed.save_path)
            print(f"Saved weights to {parsed.save_path}")
    elif parsed.save_path and parsed.load_path:
        # allow re-saving loaded weights without further training
        eng.save_weights(parsed.save_path)
        print(f"Re-saved weights to {parsed.save_path}")

    # Inference / decoder
    if parsed.prompt is not None:
        rng_seed = parsed.rng_seed
        prompt_is_ids = _looks_like_id_list(parsed.prompt)
        if eng.tokenizer is not None and not prompt_is_ids:
            text = eng.generate_text(parsed.prompt, eng.tokenizer,
                                     max_new_tokens=parsed.max_new_tokens,
                                     temperature=parsed.temperature, top_p=parsed.top_p, top_k=parsed.top_k,
                                     rng_seed=rng_seed)
            print(">> generated text:", text)
        else:
            try:
                ids = np.array([int(x) for x in parsed.prompt.split(",") if x.strip() != ""], dtype=np.int64) if prompt_is_ids else \
                  np.array([min(b, eng.V - 1) for b in parsed.prompt.encode("utf-8")], dtype=np.int64)
            except ValueError:
                raise ValueError("Provide a tokenizer or pass comma-separated token ids for prompt")
            gen = eng.generate(ids, max_new_tokens=parsed.max_new_tokens,
                               temperature=parsed.temperature, top_p=parsed.top_p, top_k=parsed.top_k, rng_seed=rng_seed)
            print(">> generated token ids:", gen.tolist())

    print("== SpiralFullFusion V9 (compact) Demo — DONE ==")
    return logs

if __name__ == "__main__":
    demo()
