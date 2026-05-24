# Model Monitoring & Drift Detection System

A production-style MLOps pipeline that trains a machine learning model, simulates real-world data drift, detects performance degradation, and automatically retrains the model.


## What This Project Does

In real ML systems, models degrade over time because the world changes — customer behaviour shifts, economies fluctuate, new competitors emerge. This project simulates exactly that and builds a system to handle it automatically.


## Project Structure

<pre>
model-monitoring-project/
├── data/
│   ├── raw.csv          
│   └── drifted.csv      
├── src/
│   ├── generate_data.py 
│   ├── train.py         
│   ├── drift.py         
│   ├── evaluate.py      
│   └── retrain.py       
├── models/
│   ├── model.pkl        
│   └── encoders.pkl     
├── app.py               
└── README.md
</pre>

## Results

| Stage | Accuracy | F1 Score |
|---|---|---|
| Original model on original data | 93.6% | 90.6% |
| Original model on drifted data | 64.9% | 51.6% |
| Retrained model on drifted data | 91.7% | 89.7% |

---

## How It Works

### 1. Data Generation
Synthetic customer churn dataset with features like age, income, tenure, support calls and monthly charge.

### 2. Model Training
Random Forest classifier trained on original data with balanced class weights.

### 3. Drift Simulation
Three types of drift are applied:
- **Age shift** — customer base gets older
- **Income noise** — economic uncertainty
- **Concept drift** — 20% of churn labels flipped

### 4. Evaluation
Model is evaluated on both original and drifted data to measure degradation.

### 5. Automated Retraining
If accuracy drops below 80%, the system automatically retrains on combined old + new data.

### 6. Dashboard
Streamlit dashboard displays all metrics and drift status in real time.

---

## How To Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/model-monitoring-project.git
cd model-monitoring-project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib evidently streamlit mlflow

# Generate data
python3 src/generate_data.py

# Train model
python3 src/train.py

# Simulate drift
python3 src/drift.py

# Evaluate performance
python3 src/evaluate.py

# Retrain if needed
python3 src/retrain.py

# Launch dashboard
streamlit run app.py
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| pandas & numpy | Data manipulation |
| scikit-learn | Machine learning |
| matplotlib | Visualizations |
| Streamlit | Dashboard UI |

---

## Key Concepts Demonstrated

- Machine learning lifecycle management
- Data drift and concept drift
- Model performance monitoring
- Automated retraining pipelines
- MLOps best practices