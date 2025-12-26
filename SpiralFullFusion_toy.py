
# -*- coding: utf-8 -*-
# SpiralFullFusion Toy (compact complete) — reliability dynamics + partial optimization + RAG SHAP + T_S(Σ) backprop
# NumPy-only, single-file demo. Safe initializations and eps clamps to avoid NaNs.
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

EPS = 1e-8

# ------------------------------
# utils
# ------------------------------
def softmax_rows(Z: np.ndarray) -> np.ndarray:
    Zm = Z - np.max(Z, axis=-1, keepdims=True)
    E = np.exp(np.clip(Zm, -50.0, 50.0))
    S = E / np.maximum(E.sum(axis=-1, keepdims=True), EPS)
    return S.astype(np.float32)

GRAD_BOOST = 20.0  # amplify toy gradients so the student actually moves

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

def rag_bcsr_bigram(tokens_batch: np.ndarray, vocab_size: int, P: np.ndarray, m: int = 3, w: float = 0.6):
    B = tokens_batch.shape[0]
    indices_list = []; data_list = []; indptr = [0]
    for b in range(B):
        last = int(tokens_batch[b, -1])
        probs = P[last]
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

class ReliabilityTracker:
    def __init__(self, L: int, H: int, ema: float = 0.9):
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

@dataclass
class TeacherCfg:
    T0: float = 1.0; lam: float = 1.0; gamma: float = 1.0; Tmin: float = 0.7; Tmax: float = 1.8
    topk: int = 40
    danger_rho_thr: float = 0.8

class SpiralTeacher:
    def __init__(self, V: int, d: int, H: int, L: int, r: int, P: np.ndarray, cfg: TeacherCfg, seed: int = 0):
        self.V, self.d, self.H, self.L, self.r = V, d, H, L, r
        self.P = P; self.cfg = cfg
        self.lm = ToyDeepLM(V, d, H, L, seed=seed)
        self.fusers = [BayesHeadFuseLR(d, V, H, r, rho=0.7, tau0=1e-4, seed=seed + l) for l in range(L)]
        # share B across layers
        B0 = self.fusers[0].B
        for f in self.fusers[1:]: f.B = B0
        self.B = B0
        self.rel = ReliabilityTracker(L, H, ema=0.9)
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

            # RAG delta on logits (bCSR)
            delta = rag_bcsr_bigram(tokens_batch[b][None, :], self.V, self.P, m=3, w=0.6)
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
        xz, stab, expl = self.rel.layer_stats(stab_thr=1.5)
        for l in range(self.L):
            expl_frac = float(np.mean(expl[l]))
            stab_frac = float(np.mean(stab[l]))
            total = max(expl_frac + stab_frac, 1e-6)
            rho_layer = expl_frac / total  # normalized explorer ratio in [0,1]
            # down-weight hazard while reliability counts are still low
            mean_n = float(np.mean(self.rel.n[l]))
            count_scale = min(1.0, mean_n / 5.0)
            rel_level = float(np.mean(self.rel.y[l]))
            rel_scale = min(1.0, rel_level / 1.5)  # only trust hazard once precision stabilizes
            rho_layer *= (count_scale * rel_scale)
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
        self.W -= lr * self.dW; self.b -= lr * self.db
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
    L: int = 4; d: int = 64; k: int = 16; V: int = 64; r: int = 16
    base_lr: float = 1e-2; head_lr: float = 1e-2; emb_lr: float = 1e-2
    grad_target: float = 0.02
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
    return float(ce), dlogits.astype(np.float32), p

def kl_teacher_student(teacher_logits: np.ndarray, student_logits: np.ndarray, T: np.ndarray) -> Tuple[float, np.ndarray]:
    q = softmax_rows(teacher_logits / np.maximum(T, 1e-6))
    p = softmax_rows(student_logits / np.maximum(T, 1e-6))
    B = p.shape[0]
    kl = np.sum(q * (np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(p, 1e-12)))) / max(1, B)
    dlogits = GRAD_BOOST * (p - q) / np.maximum(T, 1e-6)
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
             T0: float = 1.0, lam: float = 1.0, gamma: float = 1.0, k: int = 40) -> np.ndarray:
    # ∂T/∂Σ_r = T0 * γ * λ * (1+λ m)^{γ-1} * (1/k) * sum_{j∈topk} B_{rj}^2
    B2 = (B * B)  # [r,V]
    k_eff = var_top.shape[1]
    # pick same topk as in T_from_Sigma
    idx = np.argpartition(logits, -k_eff, axis=1)[:, -k_eff:]  # [B,k]
    Bn, r = Sigma.shape
    avg_B2 = np.zeros((Bn, r), dtype=np.float32)
    for b in range(Bn):
        avg_B2[b] = B2[:, idx[b]].mean(axis=1)  # [r]
    factor = (T0 * gamma * lam * np.power(1.0 + lam * meanVar[:, None], max(0.0, gamma - 1.0))).astype(np.float32)
    # if clamped at bounds, stop gradient
    clamp_mask = ((T <= (T0 * (1.0 + lam * meanVar[:, None])**gamma) - 1e-6) &
                  (T >= (T0 * (1.0 + lam * meanVar[:, None])**gamma) + 1e-6))
    dT_dSigma = (factor / max(1, k_eff)) * avg_B2  # [B,r]
    return dT_dSigma.astype(np.float32)

# ------------------------------
# Trainer
# ------------------------------
@dataclass
class TrainCfg:
    steps: int = 30; batch: int = 16; ctx_len: int = 8
    # student T hyper
    T0: float = 1.0; lam: float = 1.0; gamma: float = 1.0; Tmin: float = 0.7; Tmax: float = 1.8; topk: int = 40
    # distillation weight (single membrane for compactness)
    lam_distil: float = 0.5
    # reliability → rho nudging
    lr_k_stab: float = 0.02; lr_k_expl: float = 0.01
    # hazard → partial optimize
    keepk_boost: int = 2; rho_boost: float = 0.0
    # T backprop switch & scale
    backprop_T: bool = True; T_grad_scale: float = 0.1

class SpiralV9:
    def __init__(self, V: int = 64, d: int = 64, H: int = 3, L: int = 3, r: int = 16, seed: int = 0):
        self.V, self.d, self.H, self.L, self.r = V, d, H, L, r
        self.P = make_bigram(V, seed=7)
        self.teacher = SpiralTeacher(V, d, H, L, r, self.P, TeacherCfg(), seed=42)
        self.student = StudentV9(StudentCfg(L=4, d=d, k=16, V=V, r=r, seed=123), self.teacher.B)

    def train(self, cfg: TrainCfg) -> List[Dict[str, float]]:
        rng = np.random.default_rng(0)
        logs = []
        # initial keepk from teacher suggestion
        logits_dummy, Ts_dummy, mv_dummy, meta = self.teacher.forward_batch(np.zeros((1, cfg.ctx_len), dtype=np.int64))
        keepk = np.clip(meta["keepk_suggest"], 2, self.student.cfg.k).astype(np.int32)
        self.student.set_keepk_layerwise(keepk)

        for step in range(cfg.steps):
            Bn = cfg.batch
            ctx, y = sample_batch_bigram(self.P, ctx_len=cfg.ctx_len, B=Bn, rng=rng)

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
            dlogits = dlogits_ce + cfg.lam_distil * dlogits_kl

            # dCE/dT * dT/dSigma (optional)
            dSigma_T = None
            if cfg.backprop_T:
                dC = dCEdT(s_logits, y, T_S)                    # [B,1]
                dTdS = dTdSigma(self.student.B, s_logits, aux["Sigma"], T_S, meanVar_S, var_top_S,
                                 T0=cfg.T0, lam=cfg.lam, gamma=cfg.gamma, k=cfg.topk)  # [B,r]
                dSigma_T = (cfg.T_grad_scale * dC) * dTdS       # [B,r]

            # backward + step
            self.student.backward(dlogits, aux, dSigma_from_T=dSigma_T)
            etas, g_rms, g_scale = self.student.step()

            # reliability → rho nudging
            xz, stab, expl = self.teacher.rel.layer_stats(stab_thr=1.5)
            for l in range(self.student.cfg.L):
                # map L mismatch (teacher L vs student L): clamp index
                l_t = min(l, self.teacher.L - 1)
                stab_ratio = float(np.mean(stab[l_t])); expl_ratio = float(np.mean(expl[l_t]))
                self.student.u.rho[l] += cfg.lr_k_expl * expl_ratio - cfg.lr_k_stab * stab_ratio
                self.student.u.rho[l] = float(np.clip(self.student.u.rho[l], -2.0, 2.0))

            # partial optimization (hazard)
            hazard = meta["layer_hazard"]  # [L_teach]
            keepk_new = self.student._keepk.copy()
            for l in range(self.student.cfg.L):
                l_t = min(l, self.teacher.L - 1)
                if hazard[l_t] > self.teacher.cfg.danger_rho_thr:
                    keepk_new[l] = int(np.clip(keepk_new[l] + cfg.keepk_boost, 2, self.student.cfg.k))
                    self.student.u.rho[l] += cfg.rho_boost  # reduce LR
            # re-clamp after hazard nudging to prevent runaway shrinkage of learning rate
            self.student.u.rho = np.clip(self.student.u.rho, -2.0, 2.0)
            self.student.set_keepk_layerwise(keepk_new)

            # logging
            logs.append(dict(step=step+1, CE=float(ce), KL=float(kl),
                             ECE=float(np.mean(np.max(p_s, axis=-1) - (np.argmax(p_s, axis=-1) == y).astype(np.float32))),
                             T=float(T_S.mean()), var=float(meanVars.mean()),
                             eta0=float(etas[0]), rho0=float(self.student.u.rho[0]), keepk0=int(self.student._keepk[0]),
                             grad_rms=float(g_rms), grad_scale=float(g_scale),
                             hazard=float(hazard.mean())))
            if (step+1) % 5 == 0:
                print(f"[{step+1:03d}] CE={ce:.4f} KL={kl:.4f} ECE={logs[-1]['ECE']:.4f} | "
                      f"T={T_S.mean():.3f} var~={meanVars.mean():.4f} | η0={etas[0]:.4e} ρ0={self.student.u.rho[0]:+.3f} keepk0={self.student._keepk[0]} | "
                      f"g_rms={g_rms:.4e} g_scale={g_scale:.2f} | hazard~{hazard.mean():.3f}")
        return logs

# ------------------------------
# Demo
# ------------------------------
def demo():
    eng = SpiralV9(V=64, d=64, H=3, L=3, r=16, seed=0)
    cfg = TrainCfg(steps=100, batch=16, ctx_len=8,
                   T0=1.0, lam=1.0, gamma=1.0, Tmin=0.7, Tmax=1.8, topk=32,
                   lam_distil=0.5, lr_k_stab=0.02, lr_k_expl=0.01,
                   keepk_boost=2, rho_boost=0.0, backprop_T=True, T_grad_scale=0.1)
    logs = eng.train(cfg)
    print("== SpiralFullFusion V9 (compact) Demo — DONE ==")
    return logs

if __name__ == "__main__":
    demo()
