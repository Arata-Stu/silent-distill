# Third-party references

## Fast Feature Fields

- Repository: https://github.com/grasp-lyrl/fast-feature-fields
- License: Apache License 2.0

The dense downstream implementation references the dataset schemas, M3ED Cityscapes
19-to-11 class mapping, and event-supported optical-flow evaluation protocol in Fast Feature
Fields. Model and training code in this repository is independently implemented against the
existing `slassl` encoder, Hydra, TensorBoard, and manifest abstractions.

Unlike the centered event context used by the referenced F3 ground-truth flow loader, this
repository ends each event window at the target timestamp so that dense evaluation remains
causal. This protocol difference must be reported when comparing benchmark numbers.
