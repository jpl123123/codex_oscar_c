"""Keep implementation coverage separate from hardware acceptance."""

# These entries must be removed only after concrete code and regression evidence
# exist, not by editing an environment switch or accepting a successful HTTP code.
# Implemented-but-NPU-unverified features (drafter windows, prefix staging
# restore, device error propagation, automatic calibration) are tracked in
# docs/checklist.md; they are no longer implementation gaps.
SERVICE_IMPLEMENTATION_GAPS = (
    "fixed-address graph metadata update and actual captured model integration",
    "whole-service TP4 probes and verified NPU resource release before formal serve",
)


def require_service_implementation() -> None:
    if SERVICE_IMPLEMENTATION_GAPS:
        raise RuntimeError("Service implementation is not complete: " + "; ".join(SERVICE_IMPLEMENTATION_GAPS))
