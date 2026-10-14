# Provenance

Public source snapshot: https://github.com/vtsouval/DA_MLDG

Source commit: `8a95bc8003777f84a0cb2ea18efd9a53e929507d`. Original code and any upstream licensing files are retained. This copy adds a paper project page and poster; experiment reproduction has not been performed.

## Maintained release (2026-10-15)

The maintained `damldg/` package ports the supplied ResNet-18 architecture and repairs MLDG/DA-MLDG with Keras 3 stateless tensor updates and explicit full second-order differentiation. Original `src/` stays as historical evidence. `run.py` now invokes the maintained CLI and no longer deletes shared log/checkpoint directories. Paper-oriented defaults, all seven scenarios and source differences are documented in `docs/METHOD.md`.

The original snapshot commit was verified through the public GitHub API; it is attributed to Vasileios Tsouvalas. The recovered source includes no software license or third-party notice file. Attribution to Raeshak King Gandhi, Vasileios Tsouvalas and Nirvana Meratnia is retained; no absent third-party licensing permission is invented. The author has authorized publication of the maintained source, but reuse terms are not supplied by that publication permission alone.

Pretraining commands reconstruct a documented speaker-classification recipe; original pretraining source and weights were not recovered. Prepared feature arrays are supported, while exact TRILLsson extraction/reshape remains unavailable. The original report, website, poster and citation assets are preserved. Numerical correction and passing smoke tests do not establish reproduction of historical paper results.
