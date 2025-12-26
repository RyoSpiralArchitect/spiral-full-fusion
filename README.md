# SpiralFullFusion Toy

A single-file, NumPy-only mad science lab where a reliability-obsessed teacher and a rank-clipped student co-evolve under RAG jolts, variance-forged temperatures, and hazard-triggered capacity flips. Tiny, loud, and hell-bent on stress-testing calibration hacks in the smallest loop we could get away with.

## Why it’s interesting (and experimental)
- **Reliability cult leader**: multi-head signals are fused with Bayesian precision, dominance meters, and layer hazard to decide when to prune, poke, or unleash exploration.
- **RAG with receipts**: bCSR top-m biasing hauls in SHAP-ish attributions and per-source decay—retrieval hints must prove their worth or get throttled to oblivion.
- **Temperature as feedback control**: temperatures are carved out of variance and can backprop through Σ, letting calibration gradients melt back into the student’s parameters.
- **Compact on purpose**: zero external data, only NumPy, and a shared rank projection between teacher and student—small enough to grok, weird enough to grin.
- **Python-gonzo loop**: hand-rolled autograd-ish blocks, rank-k attention, uncertainty-tuned learning rates, and per-layer keep_k mutators—no frameworks, just raw NumPy chaos.

## How the spiral mutates
1. **Spin up toy text**: sculpt a structured bigram world and sample short contexts.
2. **Teach with reliability**: fuse heads, log dominance/reliability, inject RAG deltas, and damp sketchy sources via decay.
3. **Distill into the student**: mirror the shared rank matrix, train rank-k attention blocks, and nudge per-layer `keep_k` using hazard pulses.
4. **Close the loop**: derive temperatures from variance; mix CE + KL (distillation) with optional dT/dΣ backprop; watch CE, KL, ECE, T, hazard, and LR/keep_k dance every few steps.

## Run it
```bash
python SpiralFullFusion_toy.py
```

You’ll get a brisk training run that paints the loop: generate → reliability + RAG audit → distill → temperature feedback → repeat. It’s a pocket-sized, proudly experimental demo of reliability-aware distillation with retrieval twists.
