# SpiralFullFusion Toy

A single-file, NumPy-only playground where a reliability-savvy teacher and a rank-reduced student co-evolve. It’s a tiny demo, but it spotlights how RAG nudges, variance-shaped temperatures, and hazard-aware capacity tweaks can keep a miniature language model steady and sharp.

## Why it’s interesting
- **Reliability-first teaching**: multi-head features are fused with Bayesian precision, dominance tracking, and layer-level hazard to decide how aggressively to prune or explore.
- **RAG with accountability**: bCSR top-m biasing gets SHAP-like attributions and per-source decay, so retrieval hints get audited—not just applied.
- **Temperature as a control knob**: temperatures are derived from variance and can even backprop through Σ, letting calibration feedback loop into the student’s parameters.
- **Compact by design**: no external data, no deps beyond NumPy, and the teacher/student share a rank projection to stay lean.

## How it works
1. **Generate toy text**: craft a structured bigram distribution and sample short contexts.
2. **Teach with reliability**: the teacher fuses heads, logs dominance/reliability, injects RAG deltas, and tempers risky sources via decay.
3. **Distill into the student**: the student mirrors the shared rank matrix, uses rank-k attention blocks, and gets per-layer `keep_k` nudges from hazard signals.
4. **Close the loop**: temperatures come from variance; CE + KL (distillation) plus optional dT/dΣ backprop tighten calibration; metrics expose CE, KL, ECE, T, hazard, and LR/keep_k shifts.

## Run it
```bash
python SpiralFullFusion_toy.py
```

You’ll see a quick training run that visualizes the spiral: generate → teach with reliability and RAG → distill → adapt temperatures from uncertainty → repeat. It’s a minimal but opinionated lab for reliability-aware distillation with retrieval tweaks.
