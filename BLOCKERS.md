# NeuroAgent — Known Blockers

This file documents integration attempts that were explicitly dropped
(not deferred) and the rationale, so future contributors don't repeat the
same investigation.

---

## CANYA pretrained model — dropped

**Attempted:** Step 2.6 — integrate [github.com/lehner-lab/canya](https://github.com/lehner-lab/canya)
as a frozen pretrained baseline via isolated subprocess.

**Blocker:** CANYA's `setup.py` hard-pins `numpy==1.19.5`, which has no
prebuilt Windows wheel for Python 3.10+ and requires Microsoft Visual C++
Build Tools 14.0+ to compile from source (not installed on this dev machine).
The TF + numpy stack is from ~2021 (TF 2.6.0 / numpy 1.19.5), predating
Python 3.10 prebuilt wheel support for that numpy version.

**Exact failure:**
```
error: Microsoft Visual C++ 14.0 is required.
ERROR: No matching distribution found for numpy==1.19.5
```

**Python versions tried:** Python 3.10 (only available 3.x below 3.12 that
supports a TF with numpy<2.0 constraint). Python 3.9 is not installed on
this machine; it would have prebuilt numpy 1.19.5 wheels and avoid the
compile step.

**Decision: dropped, not deferred.**

Cost-benefit does not justify the isolation overhead:
- Requires either VS Build Tools (~6 GB install) + numpy compile, or a
  second Python 3.9 install, or Docker isolation
- CANYA is a CLI tool (no importable Python API); the wrapper would be
  a subprocess caller with a temp-file I/O round-trip for every prediction
- CANYA is a frozen model trained on ~100k *random* synthetic peptides from
  a massively parallel screen — it is not improvable by our pipeline
- **ESM-2 + CORAL already provides a comparable SOTA-embedding baseline**
  without any install friction; its embeddings are trained on real protein
  sequence databases, not synthetic random peptides
- **WaltzDB / CPAD / APR external datasets (Step 2.5)** already provide the
  benchmark-comparison value CANYA would have added as an external reference

**If revisited later:**
- Option A: Install [VS Build Tools 2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
  with "Desktop development with C++" workload, then retry `py -3.10 -m venv external/canya_env`
- Option B: Install Python 3.9 from python.org; run `py -3.9 -m venv external/canya_env`;
  numpy 1.19.5 has a prebuilt wheel for 3.9 so no compile step needed
- Option C: Docker container with Python 3.9 + CANYA pre-installed, called
  via subprocess from the main venv (most reproducible across machines)

**Artefacts left on disk:**
- `external/canya_env/` — empty Python 3.10 venv created during the attempt,
  excluded from git via `.gitignore`. Safe to delete.
