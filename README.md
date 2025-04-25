# 🩺 Economic Shifts & Public Health: Forecasting Diabetes & HBP in Canada

📊 A machine learning project analyzing the link between rising food prices and chronic health risks in Canada.  
Built to support **policy-making**, **public health strategy**, and **economic awareness** through **statistical validation**, **predictive modeling**, and **interactive forecasting**.

---

## 🔍 Problem Overview

Canadian food prices have surged over the past decade, leading to increased concern about their impact on public health. News reports suggest links between dietary affordability and chronic illnesses like **diabetes** and **high blood pressure (HBP)**—but few provide data-driven insights.

This project aims to answer:  
- How do **food prices** and **spending patterns** influence public health?
- Can we **predict disease prevalence** using economic indicators?
- How can we enable **interactive scenario analysis** for stakeholders?

---

## 💡 Solution Overview

We developed a **Streamlit-based forecasting app** powered by an optimized **XGBoost model** to simulate the impact of changes in food costs on disease rates.

Key capabilities:
- **Forecast diabetes & HBP prevalence (2023–2028)** across provinces
- **Test "what-if" scenarios** by adjusting food category CPIs (Consumer Price Indices)
- Use **interactive charts** and **live predictions** for deeper understanding

---

## 🧪 Statistical & ML Techniques

✅ **Hypothesis Testing**  
- Chi-Square Test: Association between food price tiers and health risk levels  
- ANOVA: Variance in disease prevalence across food categories  
- Test of association between diabetes and HBP  

✅ **Feature Selection Techniques**  
- Filter: Pearson Correlation  
- Wrapper: Recursive Feature Elimination  
- Embedded: Random Forest Feature Importance  

✅ **Models Evaluated**  
- ARIMA, SARIMA, Prophet  
- **XGBoost** (final model)  
- LSTM (tested for sequential data)

✅ **Model Metrics Used**  
- MSE, RMSE, MAE, R²  
- Final XGBoost Diabetes Model: RMSE = 32.67, R² = 0.82  
- Final XGBoost HBP Model: RMSE = 64.82, R² = 0.86  

---

## 🖥️ App Deployment

Built using **Streamlit** for rapid UI prototyping and interactivity:

- Province selection via dropdown  
- CPI value tuning via sliders for 10+ food categories  
- Table + chart outputs to compare baseline and adjusted forecasts  
- Real-time predictions (no reloading required)  
- Deployable via Streamlit Cloud / Heroku / local server  

🔗 Live App: https://healthnomics-forecast-98.streamlit.app/

---

## 🛠️ Technologies Used

- **Python** (Pandas, NumPy, Scikit-learn, XGBoost, Seaborn, Matplotlib)
- **Streamlit** for deployment  
- **Statsmodels**, **SciPy** for hypothesis testing  
- **Jupyter Notebook** for EDA & model evaluation  
- **Git** for version control  
- **Pickle** for model persistence  

---

## 🧠 Learning Outcomes

- Applied **ML model validation** principles including conceptual soundness, fairness, and robustness  
- Explored **bias mitigation** and **predictive performance tuning**  
- Learned how to translate complex models into **publicly accessible dashboards**  
- Developed skills in **collaborative Git workflows** and **streamlined deployment**

---
---

## 📫 Contact Me

📧 Email: mansaw1998@gmail.com  
🔗 LinkedIn: [linkedin.com/in/sawantmanish98](https://linkedin.com/in/sawantmanish98)  
🧑‍💻 GitHub: [github.com/MSawant98](https://github.com/MSawant98)

---

