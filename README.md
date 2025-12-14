# 🏡 AI Housing Price & Rent-Risk Predictor with LLM Explanations

A hybrid **Machine Learning + Generative AI** project that predicts housing prices and rent-risk levels using **Linear Regression** and **Logistic Regression** (built from scratch in NumPy), then uses an **LLM explanation layer** to translate predictions into natural-language insights.

This project demonstrates ML fundamentals from the *Supervised Machine Learning: Regression & Classification* course and extends them into a **modern AI product workflow** suitable for roles like:

- **Generative AI Engineer**
- **AI Product Developer**
- **LLM Research Engineer**
- **Machine Learning Engineer**

---

## ✨ Project Features

### 🔹 1. Multivariate Linear Regression (NumPy Only)
- Implemented gradient descent from scratch  
- Mean normalization & feature scaling  
- Cost function visualization  
- Predicts continuous housing prices (`MedHouseVal`)  

---

### 🔹 2. Logistic Regression for Rent-Risk Classification
- Binary label: high-value / high-rent-risk vs low-risk  
- Logistic regression + gradient descent  
- Classification accuracy evaluation  
- Decision boundary & probability contour plots  

---

### 🔹 3. LLM Explanation Layer (Generative AI)
- Converts raw predictions into human-readable explanations  
- Creates interpretable insights for both regression & classification outputs  
- API-compatible design (OpenAI, Anthropic, Cohere, etc.)  
- Includes placeholder stub so the notebook runs without an API key  

---

### 🔹 4. Clean Project Architecture

```text
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
│  ├─ 01_eda_and_preprocessing.ipynb
│  ├─ 02_regression_house_price.ipynb
│  ├─ 03_classification_rent_risk.ipynb
│  └─ 04_llm_explanations.ipynb
└─ src/
   ├─ data_loader.py
   ├─ features.py
   ├─ regression.py
   ├─ classification.py
   └─ explain_llm.py
```

---

## 📊 Models & Methods Used

### Regression
- Multivariate linear regression  
- Gradient descent optimization  
- Cost function minimization  
- Feature scaling (mean normalization)  
- Prediction visualization  

### Classification
- Logistic regression  
- Sigmoid function  
- Cross-entropy loss  
- Threshold tuning  
- Decision boundary visualization  

### LLM Integration
- Prompt engineering  
- Feature → structured prompt conversion  
- Natural language explanation generation  
- Ready for OpenAI / Anthropic / etc.  

---

## 🚀 Screenshots (Recommended)

*(Add images once generated in notebooks)*

- Cost vs iterations plot  
- Predicted vs actual housing prices  
- Logistic regression decision boundary  
- LLM explanation output screenshot  

Example:
```markdown
![Regression Cost Curve](assets/cost_curve.png)
![Decision Boundary](assets/decision_boundary.png)
![LLM Explanation](assets/llm_output.png)
```

---

## 🧪 How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add dataset
The California Housing dataset loads automatically using `sklearn.datasets`.  
If using a different dataset, place it in:
```bash
data/raw/
```

### 3. Run notebooks (in order)
1. `01_eda_and_preprocessing.ipynb`
2. `02_regression_house_price.ipynb`
3. `03_classification_rent_risk.ipynb`
4. `04_llm_explanations.ipynb`

### 4. (Optional) Add an LLM API key
```bash
export OPENAI_API_KEY="your_key_here"
```

Replace the stub in `explain_llm.py` with real API calls.

---

## 🎯 Intended For

- Portfolio projects  
- Résumé experience  
- AI job applications  
- Interview discussions  

---

## 📄 Résumé-Ready Bullet Points

- Built a hybrid Machine Learning + LLM system that predicts housing prices and rent-risk using NumPy-based linear and logistic regression, then generates natural-language explanations through a generative AI layer.
- Implemented gradient descent, data normalization, and model evaluation from scratch in Python and visualized learning dynamics through cost curves, classification boundaries, and prediction-vs-ground-truth plots.
- Designed modular ML pipelines and prompt-engineering interfaces to integrate tabular data predictions with LLM reasoning, demonstrating practical experience with explainable AI and AI product development workflows.

---

## 🧰 Tech Stack
- **Python**
- **NumPy, Pandas, Matplotlib**
- **Scikit-learn (dataset only)**
- **Jupyter Notebook**
- **LLM API (optional)**

---

## 📬 Future Improvements
- Add L2 regularization  
- Add polynomial features  
- Compare scratch models vs scikit-learn models  
- Add SHAP or feature importance  
- Convert LLM layer into a REST API (FastAPI)  
- Add a Streamlit UI for interactive predictions  

---

## 🌟 Author

**Samira Zare**  
Generative AI Engineer | Machine Learning Researcher  
Website: *samirzare.com*
