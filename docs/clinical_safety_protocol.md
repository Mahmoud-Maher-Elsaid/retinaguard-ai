# RetinaGuard-AI Clinical Safety Protocol

Purpose:
This document describes the research safety logic used by RetinaGuard-AI.

Safety Gate inputs:
1. EfficientNet confidence
2. Prediction entropy
3. Top-2 probability margin
4. Image quality score
5. U-Net lesion burden
6. Lesion-grade consistency
7. Predicted DR severity

Triage decisions:
- safe_negative_prediction
- low_risk_follow_up
- follow_up_recommended
- routine_referral
- urgent_referral
- manual_review_required

Conservative review logic:
The system recommends manual review when image quality is poor, prediction uncertainty is high, lesion evidence does not match predicted grade, or confidence/probability margin is weak.

Disclaimer:
This project is a research prototype only. It must not be used for diagnosis or treatment decisions.
