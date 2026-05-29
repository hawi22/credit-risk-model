# Credit Risk Probability Model - Bati Bank

## Credit Scoring Business Understanding

### 1. Basel II Accord & Model Interpretability
Under Basel II, financial institutions must be able to explain the "why" behind risk weights. Our model needs to be interpretable so that regulators and the loan origination team can understand why a specific customer was assigned a high risk probability.

### 2. The Proxy Variable Necessity
Since the Xente dataset does not have a "default" label (whether someone paid back), we use RFM (Recency, Frequency, Monetary) analysis. 
- **Risk:** We might misidentify a brand-new customer who is actually creditworthy as "High Risk" simply because they haven't made many transactions yet.

### 3. Model Trade-offs
- **Logistic Regression:** High interpretability, lower complexity. Easier to meet Basel II standards.
- **Gradient Boosting (XGBoost):** Higher predictive power, lower interpretability. Requires additional tools (like SHAP) to explain decisions to regulators.