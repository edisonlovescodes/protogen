# Changelog

## 2026-10-22 — Simpler state and contracts

- Simplified model loading and required internal arguments; incompatible artifacts now fail explicitly at load time. Missing artifacts still leave inference unavailable.
- Removed redundant flags, fallback values and unused wrapper behavior.
- Consolidated workflow state and added regression coverage for invalid artifacts and asynchronous interactions.
- React workflows now use TypeScript discriminated unions; production builds include strict type checking.

## 2026-10-13

- Added default causal and padding masks with invariance regression tests.
- Added exact-sequence deduplication, deterministic split IDs, EOS preservation, token-weighted evaluation and a unigram baseline.
- Added a versioned train/save/load bundle; retained legacy checkpoint files without silently serving them.
- Fixed launch-directory dependence and bounded API inputs/export; disabled external folding.
- Replaced hard-coded remote frontend calls with a local proxy and clarified composition controls.
- Removed tracked node_modules/bytecode/OS metadata from the proposed Git tree; kept package locks and historical Git commits.
- Added regression tests, CI and reproducible local instructions.

## 2026-10-13 — Distinct interface redesign

Dark ink/acid-green sequence studio, self-hosted Space Grotesk/IBM Plex Mono, amino-acid schematic, composition sliders, colored sequence display, clipboard/FASTA controls, responsive layout and generation/copy/export/error tests.

Existing research/evaluation limitations remain in CLAIMS.md. Font licenses are included alongside the assets.
