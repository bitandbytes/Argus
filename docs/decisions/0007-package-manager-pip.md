# ADR-0007: Package Manager — pip + requirements.txt

## Status
Accepted

## Context

Python has multiple package management options:

1. **pip + requirements.txt** — Universal, simple, no learning curve, no extra tooling.
2. **Poetry** — Better dependency resolution, lock files, virtual env management built in.
3. **uv** — Modern, fast, written in Rust, drop-in pip replacement with lock file support.
4. **conda/mamba** — Cross-language, great for scientific Python, heavyweight.

This project uses scientific Python libraries (numpy, pandas, torch, xgboost) which all install cleanly via pip wheels on modern systems. The project doesn't have complex non-Python dependencies that would require conda.

## Decision

Use **pip + requirements.txt** for dependency management.

## Consequences

**Positive:**
- **Universal compatibility**: Every Python developer knows pip.
- **No tooling overhead**: Standard library tools, no extra installs.
- **Simple workflow**: `pip install -r requirements.txt` is enough.
- **CI/CD friendly**: Every CI system supports pip out of the box.
- **VS Code integration**: Works with the standard Python extension without configuration.

**Negative:**
- **No automatic lock file**: Reproducibility relies on pinning versions in `requirements.txt`.
- **No dependency resolution improvements**: pip's resolver is good but slower than uv's.
- **Manual virtual environment**: Developers must remember to create and activate `.venv`.

**Mitigation:**
- Pin major versions in `requirements.txt` (e.g., `pandas>=2.1.0,<3.0`) to balance stability and updates.
- Document virtual environment setup clearly in README and CLAUDE.md.
- Future migration to `uv` is trivial since `uv` reads `requirements.txt` natively.
