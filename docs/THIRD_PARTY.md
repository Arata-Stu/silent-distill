# Third-party references

## Fast Feature Fields

- Repository: https://github.com/grasp-lyrl/fast-feature-fields
- Audited revision: `320eec6`
- License: Apache License 2.0

The dense downstream implementation references the dataset schemas, M3ED Cityscapes
19-to-11 class mapping, event-supported optical-flow evaluation protocol, and the full versus
event-masked flow visualization layout in Fast Feature Fields. Model, training, and rendering
code in this repository is independently implemented against the existing `slassl` encoder,
Hydra, TensorBoard, manifest abstractions, NumPy, and Pillow.

Unlike the centered event context used by the referenced F3 ground-truth flow loader, this
repository ends each event window at the target timestamp so that dense evaluation remains
causal. This protocol difference must be reported when comparing benchmark numbers.

## E-RAFT

- Repository: https://github.com/uzh-rpg/E-RAFT
- Audited revision: `c58ce05`
- License: MIT

The DSEC test adapter cross-checks E-RAFT's test sequence discovery, event timestamp offset,
distorted-to-rectified coordinate mapping, 100 ms forward-flow interval, CSV-selected output
indices, and six-digit filename convention. The model and dataloader are not copied. Submission
PNG encoding is implemented independently against the official DSEC format.

## EVA-Flow

- Repository: https://github.com/Yaozhuwa/EVA-Flow
- Audited revision: `5534f52`

EVA-Flow was used as a second cross-check for DSEC's split event HDF5 layout, rectification map,
forward timestamp triples, 16-bit flow decoder, and 100 ms displacement convention. The local
clone does not include a complete benchmark exporter, so it is not the primary submission
reference.

## ADMFlow

- Repository: https://github.com/boomluo02/ADMFlow
- Audited revision: `0e44913`

ADMFlow is used only to audit MVSEC protocol choices. Its `dt1`/`dt4` flow construction,
sequence-specific valid ranges, event-supported sparse mask, outdoor crop, and per-sample metric
averaging are MVSEC choices and are deliberately not applied to DSEC server submissions.
