# Retrieval quality vs downstream utility: rank agreement

- RuleBased @ k=1: tau_b=0.6667, best by nDCG = dense, best by reward = tfidf
- RuleBased @ k=3: tau_b=1.0, best by nDCG = dense, best by reward = dense
- RuleBased @ k=5: tau_b=1.0, best by nDCG = dense, best by reward = dense
- TFIDF_LogReg_Calibrated @ k=1: tau_b=0.0, best by nDCG = dense, best by reward = random
- TFIDF_LogReg_Calibrated @ k=3: tau_b=0.3333, best by nDCG = dense, best by reward = dense
- TFIDF_LogReg_Calibrated @ k=5: tau_b=0.0, best by nDCG = dense, best by reward = dense
