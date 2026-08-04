# Chapter 3 — Machine Learning: From Data to Reliable Predictions

> **Part 3 — Machine Learning**  
> **Target time:** 10–12 hours  
> **Outcome:** You can frame a business problem as an ML task, select practical baseline algorithms, evaluate them honestly, and design a production prediction service.

## 1. What machine learning is—and when to use it

Machine learning (ML) learns a pattern from historical examples instead of requiring a developer to write every rule. The learned pattern becomes a model that estimates an outcome for new inputs.

```text
Historical records ─► clean and label ─► train candidate models ─► evaluate
                                                                      │
New record ─► validate features ─► deployed model ─► score ─► business action
                                                    │             │
                                                    └─ monitoring ─┘
```

### What ML solves

ML is useful when the input has many weak signals, the desired outcome is observable, and historical examples are reasonably representative. Examples include:

| Business question | ML task | Example output |
|---|---|---|
| Which payment needs review? | Classification | `review_probability = 0.81` |
| How much inventory will sell? | Regression | `next_week_units = 132` |
| Which customers behave similarly? | Clustering | `segment = price_sensitive` |
| What should appear first? | Ranking | ordered product list |
| Is this event unusual? | Anomaly detection | anomaly score |

### What ML does *not* solve well

Use a database query, deterministic rule, or normal software when the answer is already specified and must be exact. “Is a customer over the credit limit?” is often a rule. “Which of millions of customers is likely to churn next month?” is a modeling candidate.

> **Decision rule:** Start with the lowest-complexity solution that can meet the requirement. A clear business rule or SQL query is cheaper, easier to audit, and usually safer than an ML model.

### The ML product lifecycle

```text
Problem and metric
      │
      ▼
Data contract ─► baseline ─► offline evaluation ─► shadow/canary ─► production
      ▲                                                     │             │
      └────────── feedback, drift, labels, incidents ──────┴─────────────┘
```

An ML model is a hypothesis encoded in software. It is not complete when training ends; it needs monitoring, feedback, retraining criteria, and an owner.

---

## 2. Supervised learning

### Concept explanation

**Supervised learning** uses examples with an input and an expected answer, called a label. During training, the algorithm looks for a mapping that generalizes to new inputs.

```text
Features (X)                         Label (y)
──────────────────────────────────────────────────
account age, failed payments   ──►   manual review needed
house size, bedrooms, location ──►   sale price
email text features             ──►   spam / not spam
```

The two main supervised tasks are classification and regression.

### Classification

Classification predicts a category. Binary classification has two outcomes: fraud/not fraud, churn/no churn, urgent/not urgent. Multiclass classification has several: billing, shipping, product, account access.

Many classifiers emit a score or probability-like value. The application decides what action corresponds to that score.

```python
from sklearn.linear_model import LogisticRegression

# [failed_payments, account_age_days]
X_train = [[0, 600], [3, 5], [0, 300], [4, 2], [1, 15], [0, 900]]
y_train = [0, 1, 0, 1, 1, 0]  # 1 = manual review

model = LogisticRegression().fit(X_train, y_train)
probability = model.predict_proba([[2, 10]])[0, 1]
print(f"Manual review probability: {probability:.2f}")
```

### Regression

Regression predicts a continuous number: demand, delivery duration, revenue, or energy use. A regression output should include plausible limits and uncertainty handling. A forecast of negative shipments is technically numeric but operationally nonsensical.

```python
from sklearn.linear_model import LinearRegression

# advertising spend -> weekly orders, simplified example
X_train = [[1], [2], [3], [4]]
y_train = [12, 20, 31, 39]

model = LinearRegression().fit(X_train, y_train)
forecast = model.predict([[5]])[0]
print(max(0, forecast))  # domain rule: orders cannot be negative
```

### Why labels are often the hard part

The model code may take minutes; trustworthy labels can take months. Ask:

- What exactly does the label mean?
- Who created it, and could they be biased or inconsistent?
- Was it known at prediction time?
- Does the label represent the desired outcome, or just a proxy?

For example, “ticket closed” is not necessarily “customer issue solved.” Training on closure alone may reward a system that closes cases prematurely.

### Evaluation metrics for classification

| Metric | Answers | Useful when | Limitation |
|---|---|---|---|
| Accuracy | How often was it correct? | Balanced, low-stakes classes | Misleading for rare events |
| Precision | Of predicted positives, how many were right? | Expensive false positives | Can miss real positives |
| Recall | Of actual positives, how many were found? | Expensive false negatives | Can create many false alarms |
| F1 | Balance precision and recall | Need a single comparison metric | Hides business cost |
| ROC-AUC / PR-AUC | How well does it rank? | Compare candidates | Does not set an action threshold |

```text
                        Actual positive    Actual negative
Predicted positive       true positive      false positive
Predicted negative       false negative     true negative

precision = TP / (TP + FP)       recall = TP / (TP + FN)
```

For fraud, missed fraud can be costly, so recall matters. For a manual-review queue, too many false positives can overwhelm staff, so precision and capacity matter too.

### Production architecture: supervised model service

```text
Event/API request
      │
      ▼
schema validation ─► feature builder ─► model registry/version ─► score
      │                     │                     │                 │
      │                     ▼                     ▼                 ▼
      └────────────► feature logs          monitoring        rule/queue/action
                                                                    │
                                                           human feedback label
```

### Common mistakes

- Training with a future field that will not exist at inference time.
- Allowing a model to decide a sensitive action without review or policy constraints.
- Treating all labels as ground truth without auditing label quality.
- Reporting accuracy for a severely imbalanced dataset.

### Interview questions

1. Explain precision versus recall using a fraud-review queue.
2. What is label leakage? Give an example.
3. How would you decide the classification threshold for a support escalation model?
4. Why should a model version be recorded with every prediction?

---

## 3. Unsupervised learning

### Concept explanation

**Unsupervised learning** works without predefined labels. It finds patterns, groups, or unusual cases. It is exploratory: results require human interpretation.

### Clustering

Clustering groups similar records. A retailer might cluster customers by recency, frequency, monetary value, and product preferences. The cluster IDs have no inherent meaning; analysts must inspect them and assign useful names.

```text
customer features ─► scaling/cleaning ─► clustering algorithm ─► clusters
                                                            │
                                                   inspect and name segments
```

#### K-means

K-means chooses `k` centroids and repeatedly assigns points to the nearest centroid. It is fast and easy to explain, but assumes roughly compact, similarly sized clusters and requires selecting `k`.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# [orders_last_90_days, average_order_value]
customers = [[1, 25], [2, 30], [15, 150], [12, 120], [1, 18], [17, 160]]
X = StandardScaler().fit_transform(customers)
labels = KMeans(n_clusters=2, random_state=42, n_init="auto").fit_predict(X)
print(labels)
```

#### Other approaches

| Algorithm | Strength | Limitation | Typical use |
|---|---|---|---|
| K-means | Fast, simple baseline | Must choose `k`; shape assumptions | Customer segments |
| DBSCAN | Finds irregular shapes; marks noise | Sensitive to density settings | Geographic/event clusters |
| Hierarchical clustering | Shows nested relationships | Expensive at large scale | Exploratory analysis |
| PCA / UMAP | Reduces dimensions for analysis/visualization | Can distort detail | Inspecting embeddings/features |

### Anomaly detection

Anomaly detection identifies unusual observations: abnormal login patterns, equipment behavior, or data quality failures. “Unusual” is not equivalent to “bad” or “fraudulent.” It should trigger investigation or a safe workflow, not automatic punishment.

### Cost and operational implications

Unsupervised systems can reduce manual analysis but require ongoing interpretation. Data distributions drift, cluster memberships change, and a segmentation result can become a stale assumption. Recompute on a planned schedule and compare segment stability.

### Common mistakes

- Naming clusters “high value” without inspecting actual records.
- Selecting the number of clusters solely because a plot looks appealing.
- Using clustering results as legal, credit, or employment decisions without careful governance.
- Forgetting to scale features before distance-based clustering.

### Interview questions

1. How would you select the number of clusters for a customer-segmentation experiment?
2. Why must clusters be interpreted after training?
3. What is the difference between an anomaly and a fraud label?

---

## 4. Reinforcement learning in an ML workflow

Reinforcement learning (RL) learns actions from rewards over time. It is distinct from supervised learning because feedback may be delayed and the agent’s actions change future data.

```text
state ─► policy ─► action ─► environment ─► reward + next state
  ▲                                                        │
  └──────────────────────── learning update ──────────────┘
```

### Industry use cases

- Robotics and industrial control in constrained environments.
- Recommendation or bidding policies with rigorous experimentation.
- Resource allocation and scheduling simulations.
- Post-training preference optimization for foundation models.

### Limitations and safety

RL is rarely the starting point for a business product. Defining a reward can produce harmful shortcuts, online exploration can harm users, and offline historical logs may not predict policy changes accurately. Use simulation, hard constraints, human oversight, and limited pilots.

> **Example:** Optimizing a support bot only for “time to ticket close” can teach it to close tickets prematurely. A useful reward needs resolution quality, reopen rate, safety, and customer satisfaction—not a single easy proxy.

---

## 5. Practical algorithm guide

### Linear and logistic regression

**What they are:** Linear regression predicts a number. Logistic regression predicts a class probability using a linear combination of features.

**Advantages:** fast, interpretable, low-cost, strong baselines.

**Limitations:** may miss nonlinear interactions unless features are engineered.

**Use when:** you need transparency, a baseline, or a modest tabular problem.

### Decision trees and random forests

A decision tree makes a sequence of feature splits. A **random forest** trains many trees on varied samples/features and combines them. This reduces the instability of one tree.

```text
failed payments > 1?
  ├─ yes: account age < 30 days? ─► review
  └─ no:  order value > $5,000?  ─► review or allow
```

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300, min_samples_leaf=10, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)
```

**Advantages:** handles nonlinear relationships, mixed signals, and modest feature preparation; robust baseline for tabular data.

**Limitations:** larger model than one tree; can be harder to calibrate/interpret; may underperform boosting on some structured datasets.

### XGBoost and gradient-boosted trees

Boosting builds trees sequentially, with each tree correcting earlier errors. XGBoost is widely used for high-quality structured/tabular prediction. Similar families include LightGBM and CatBoost.

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss"
)
model.fit(X_train, y_train)
```

**Advantages:** frequently excellent performance on tabular data; handles interactions well.

**Limitations:** tuning can be time-consuming; overfitting is possible; explainability and calibration still require work.

### Support Vector Machine (SVM)

An SVM seeks a boundary with maximum separation (margin) between classes. With kernels, it can model nonlinear boundaries.

**Advantages:** can work well on medium-sized, high-dimensional data such as text features.

**Limitations:** scaling and parameter choices matter; training can become expensive on large datasets; probability outputs need extra care.

### K-nearest neighbors (KNN)

KNN predicts based on the closest labeled examples. It is simple and useful as a local-similarity baseline.

**Advantages:** almost no training; intuitive.

**Limitations:** slow query-time search at scale, sensitive to feature scaling/noise, and weak in high dimensions.

### Selection guide

| Situation | Start with | Why |
|---|---|---|
| Small, explainable tabular classification | Logistic regression | Transparent baseline |
| Strong tabular baseline | Random forest or XGBoost | Captures nonlinear patterns |
| Medium-sized sparse text features | Linear model or SVM | Efficient high-dimensional behavior |
| Local similarity, small dataset | KNN | Simple comparison point |
| Images, audio, raw language | Pretrained deep model | Representation learning matters |

> **Best practice:** Compare at least one simple baseline against a more complex candidate. If the complex model is not clearly better on held-out data and business constraints, do not deploy it.

---

## 6. Training and evaluation workflow

### Data splits

```text
raw records ─► time/entity-aware split ─► train (60–80%)
                                         ├► validation (10–20%)
                                         └► test (10–20%, sealed)
```

For a customer-level task, do not let the same customer appear in both training and test data if that would overstate performance. For time-dependent tasks, train on past data and test on future data.

### Baseline, candidate, and release gate

```python
def release_allowed(metrics: dict) -> bool:
    return (
        metrics["recall_at_review_capacity"] >= 0.85
        and metrics["false_positive_rate"] <= 0.03
        and metrics["p95_latency_ms"] <= 100
    )
```

A model can be statistically better but still fail deployment constraints, such as latency, operational review capacity, fairness, or cost.

### Production monitoring

Monitor four categories:

| Category | Example signals |
|---|---|
| Service health | errors, p95 latency, throughput, model load failures |
| Data health | missing features, schema changes, range shifts |
| Model health | score distribution, calibration, delayed label quality |
| Business health | fraud caught, review workload, resolution rate, cost/outcome |

**Data drift** means inputs changed; **concept drift** means the relationship between inputs and outcomes changed. Neither automatically means “retrain now,” but both demand investigation.

### Cost optimization

- Use a smaller/cheaper baseline model if it meets the measured requirement.
- Batch offline scoring and feature computation.
- Cache stable feature lookups, not stale predictions whose inputs change.
- Avoid real-time feature dependencies that add latency unless they materially improve outcomes.
- Measure cost per successful action, including human-review workload created by false positives.

---

## 7. Hands-on project — churn prediction service

Build a churn-risk service for a fictional subscription product.

### Required scope

1. Define churn precisely, such as “canceled within 30 days.”
2. Build a dataset with only features available before the prediction date.
3. Train logistic regression, random forest, and XGBoost candidates.
4. Compare precision, recall, PR-AUC, calibration, p95 inference latency, and estimated monthly cost.
5. Choose a threshold based on the capacity of a retention team.
6. Expose `POST /score` using FastAPI and Pydantic input validation.
7. Record model version, feature version, timestamp, score, and resulting human action.
8. Write tests for missing/invalid features and a simple drift alert.

### Suggested project architecture

```text
CSV/warehouse ─► validation + feature job ─► versioned training dataset
                                                   │
                                              training/evaluation
                                                   │
API ─► schema validation ─► feature transform ─► registered model ─► score + route
                                                                    │
                                             dashboard ◄── logs / delayed churn labels
```

### Project interview prompts

- Why did you select your release metric and threshold?
- What leakage risks existed in the churn dataset?
- How would you retrain safely after a pricing change?
- What action happens when feature values are missing?
- Why did the chosen model win after considering costs, not only AUC?

---

## Summary

Supervised learning predicts known labels; unsupervised learning discovers structure without labels; reinforcement learning optimizes sequential actions from rewards. Classification, regression, and clustering solve different problems and demand different metrics. For structured data, logistic regression, random forests, and gradient-boosted trees are practical tools that often outperform a rush to deep learning.

The production standard is not “highest benchmark score.” It is a clearly defined outcome, leakage-free data, representative evaluation, bounded decisions, versioned deployment, monitoring, and a business case that includes human and infrastructure costs.

## Further reading

- [scikit-learn — Supervised learning](https://scikit-learn.org/stable/supervised_learning.html)
- [scikit-learn — Unsupervised learning](https://scikit-learn.org/stable/unsupervised_learning.html)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [Google — Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Evidently — ML monitoring concepts](https://docs.evidentlyai.com/docs/overview)

---

**Next chapter:** Part 4 — Deep Learning: neural networks, PyTorch, TensorFlow, CNNs, RNNs, attention, transformers, and positional encoding.
