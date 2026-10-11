# Migration Manifest

This document tracks migration from legacy artifact workspace into `edm_ciper_research`.

## Source to Target Mapping

- `artifacts/rc-20260331-174028-0bb271/deliverables/paper_final.md`
  -> `edm_ciper_research/paper/paper_final.md` (new managed draft with auto-results block)
- Legacy experiment code under `artifacts/.../deliverables/code/`
  -> Rebuilt modular code under `edm_ciper_research/src/`
- Legacy charts under `artifacts/.../deliverables/charts/`
  -> Regenerated charts under `edm_ciper_research/artifacts/figures/`

## Migration Policy

- Legacy workspace kept as historical reference only.
- All new experiments and manuscript updates must happen in `edm_ciper_research`.

