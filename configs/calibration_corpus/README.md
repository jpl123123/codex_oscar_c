# Calibration corpus (pinned)

`gpqa_diamond.csv` is the exact calibration corpus pinned by
`configs/calibration.json` (`dataset_sha256`
`41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`).

Source: the publicly hosted, unauthenticated copy used by OpenAI's
simple-evals: https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv

It is vendored so that air-gapped target machines (whose only inbound channel
is this repository) can run the model-matched calibration without outbound
network access. `ensure_corpus` verifies the pinned SHA256 before use; the
URL remains the fallback for machines with normal egress. Calibration uses
this text only as prompt input; no model K/V or statistics leave the NPU.
