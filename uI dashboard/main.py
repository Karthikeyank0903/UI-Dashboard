import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
data = pd.read_csv('heart.csv')

# Split the data
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_test)

# Evaluate the Models
# Logistic Regression Metrics
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
precision_log_reg = precision_score(y_test, y_pred_log_reg)
recall_log_reg = recall_score(y_test, y_pred_log_reg)
roc_auc_log_reg = roc_auc_score(y_test, y_pred_log_reg)

# Decision Tree Metrics
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
precision_decision_tree = precision_score(y_test, y_pred_decision_tree)
recall_decision_tree = recall_score(y_test, y_pred_decision_tree)
roc_auc_decision_tree = roc_auc_score(y_test, y_pred_decision_tree)

# Print Metrics
print(f"Logistic Regression - Accuracy: {accuracy_log_reg}, Precision: {precision_log_reg}, Recall: {recall_log_reg}, ROC AUC: {roc_auc_log_reg}")
print(f"Decision Tree - Accuracy: {accuracy_decision_tree}, Precision: {precision_decision_tree}, Recall: {recall_decision_tree}, ROC AUC: {roc_auc_decision_tree}")

# Compute ROC curve and ROC area
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, log_reg.predict_proba(X_test)[:, 1])
fpr_decision_tree, tpr_decision_tree, _ = roc_curve(y_test, decision_tree.predict_proba(X_test)[:, 1])

# Create a DataFrame with metrics
metrics = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [accuracy_log_reg, accuracy_decision_tree],
    'Precision': [precision_log_reg, precision_decision_tree],
    'Recall': [recall_log_reg, recall_decision_tree],
    'ROC AUC': [roc_auc_log_reg, roc_auc_decision_tree]
})

# Save to CSV
metrics.to_csv('model_metrics.csv', index=False)

# Create a DataFrame with ROC data
roc_data = pd.DataFrame({
    'False Positive Rate': np.concatenate([fpr_log_reg, fpr_decision_tree]),
    'True Positive Rate': np.concatenate([tpr_log_reg, tpr_decision_tree]),
    'Model': ['Logistic Regression'] * len(fpr_log_reg) + ['Decision Tree'] * len(fpr_decision_tree)
})

# Save to CSV
roc_data.to_csv('roc_data.csv', index=False)

# Plot Model Performance Metrics
fig_metrics = px.bar(metrics, x='Model', y=['Accuracy', 'Precision', 'Recall', 'ROC AUC'],
                     barmode='group', title='Model Performance Metrics')

fig_metrics.show()

# Plot ROC Curves
fig_roc = go.Figure()

fig_roc.add_trace(go.Scatter(x=fpr_log_reg, y=tpr_log_reg, mode='lines', name='Logistic Regression (AUC = %0.2f)' % roc_auc_log_reg))
fig_roc.add_trace(go.Scatter(x=fpr_decision_tree, y=tpr_decision_tree, mode='lines', name='Decision Tree (AUC = %0.2f)' % roc_auc_decision_tree))

fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))

fig_roc.update_layout(title='ROC Curve',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate')

fig_roc.show()
