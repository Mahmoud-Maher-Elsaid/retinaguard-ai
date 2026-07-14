# RetinaGuard Safety Gate

This stage adds an uncertainty-aware safety triage layer on top of the diabetic retinopathy pipeline.

The safety gate combines:

1. EfficientNet prediction confidence.
2. Normalized prediction entropy.
3. Top-2 class probability margin.
4. Fundus image quality metrics.
5. U-Net predicted lesion burden.
6. Lesion-grade consistency.

The output is not a clinical diagnosis. It is a research-oriented safety layer that decides whether a model prediction looks reliable, needs follow-up, requires routine referral, urgent referral, or manual review.
