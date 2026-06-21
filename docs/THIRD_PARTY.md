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

`src/slassl/cli/generate_m3ed_flow.py` is adapted from
`src/f3/tasks/optical_flow/generate_gt.py` at the audited revision. It retains the camera-motion,
depth projection, velocity smoothing, and HDF5 schema while removing movie/histogram rendering
and adding input validation, non-positive-depth masking, atomic output, overwrite protection,
and provenance attributes. The original Apache-2.0 license is included at
`third_party/fast-feature-fields/LICENSE`.

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

## E-STMFlow

- Repository: https://github.com/AhmedHumais/E-STMFlow
- Audited revision: `2ce4821d`
- License: no repository-level license file found at the audited revision

E-STMFlow provides a third cross-check for DSEC's test event layout, timestamp offset,
rectification map, 100 ms forward-flow convention, CSV-selected six-digit output indices, and
16-bit flow encoding. Its single input sample is a 100 ms event interval represented as a
normalized, signed 32-bin voxel grid. The model consumes those bins as an internal
spatio-temporal token sequence, but does not carry recurrent state between successive flow
samples.

The test loader reconstructs 10 Hz intervals from every second image timestamp and uses the
official CSV only to select which file indices to submit. This repository instead indexes the
CSV's explicit `from_timestamp_us,to_timestamp_us,file_index` rows directly and verifies the
complete output file set. No E-STMFlow model or loader code is copied.

## ADMFlow

- Repository: https://github.com/boomluo02/ADMFlow
- Audited revision: `0e44913`

ADMFlow is used only to audit MVSEC protocol choices. Its `dt1`/`dt4` flow construction,
sequence-specific valid ranges, event-supported sparse mask, outdoor crop, and per-sample metric
averaging are MVSEC choices and are deliberately not applied to DSEC server submissions.
