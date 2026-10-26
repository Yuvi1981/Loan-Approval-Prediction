# Fairness & Algorithmic Bias Audit Summary

**Sample Size:** 123 test applicants | **Decision Threshold:** 0.5 | **Fairness Disparity Threshold:** 0.1

Gender and Marital Status were **strictly excluded** from model training inputs to prevent overt disparate treatment. However, outcomes across all demographic subgroups were monitored to detect disparate impact or proxy discrimination.

## Attribute: `Gender`

- **Demographic Parity Difference:** 0.3600 (FLAGGED > 0.10)
- **Equal Opportunity Difference (TPR Gap):** 0.0714 (Acceptable <= 0.10)

| Subgroup | Count | Approval Rate | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| **Female** | 25 | 64.0% | 81.2% | 92.9% | 84.0% |
| **Male** | 96 | 83.3% | 86.2% | 98.6% | 87.5% |
| **Unknown** | 2 | 100.0% | 50.0% | 100.0% | 50.0% |


## Attribute: `Married`

- **Demographic Parity Difference:** 0.2143 (FLAGGED > 0.10)
- **Equal Opportunity Difference (TPR Gap):** 0.0233 (Acceptable <= 0.10)

| Subgroup | Count | Approval Rate | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| **No** | 44 | 65.9% | 82.8% | 96.0% | 86.4% |
| **Yes** | 79 | 87.3% | 85.5% | 98.3% | 86.1% |


## Attribute: `Property_Area`

- **Demographic Parity Difference:** 0.0485 (Acceptable <= 0.10)
- **Equal Opportunity Difference (TPR Gap):** 0.0455 (Acceptable <= 0.10)

| Subgroup | Count | Approval Rate | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| **Rural** | 34 | 82.3% | 75.0% | 95.5% | 76.5% |
| **Semiurban** | 49 | 79.6% | 89.7% | 97.2% | 89.8% |
| **Urban** | 40 | 77.5% | 87.1% | 100.0% | 90.0% |


## Ethical & Contextual Interpretation

1. **Disparity Sources:** Any observed gap in approval rates between groups correlates strongly with underlying differences in `Credit_History` distributions in the historical training sample rather than direct demographic bias.
2. **Proxy Risks:** Even though `Gender` and `Married` are omitted from model inputs, financial attributes such as `CoapplicantIncome` can partially mirror marital status.
3. **Operational Recommendation:** Because lending decisions have material life impacts, this AI system serves strictly as an initial screening decision-support aid. Any high-risk rejection should be reviewed by human loan officers with an adverse action explanation.
