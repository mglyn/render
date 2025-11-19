# Lighting Mode TODO

1. Review current handling of LightingMode enum usage in `src/renderer/path_tracer.cu` and renderer control flow.
2. Define desired behavior for each mode (direct, indirect, MIS) including what sampling strategies run.
3. Update CUDA path tracer to branch based on mode, ensuring correct combination of BSDF and light sampling with MIS weights when required. ✅
4. Verify renderer UI/controls pass correct mode selection and that accumulation resets when mode changes (if needed). ✅
5. Test all three modes to confirm expected visual differences.
