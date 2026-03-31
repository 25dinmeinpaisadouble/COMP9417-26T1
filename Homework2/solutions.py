import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

# PART G 

N_MAJORITY = 500
np.random.seed(42)

X0 = np.random.randn(N_MAJORITY, 2) + np.array([-3, 0])
y0 = np.zeros(N_MAJORITY)

X2 = np.random.randn(N_MAJORITY, 2) + np.array([3, 0])
y2 = np.ones(N_MAJORITY) * 2

middle_cluster_sizes = [500, 250, 100, 50, 25, 10, 5]
softmax_recalls = []
ovr_recalls = []

for n1 in middle_cluster_sizes:
    X1 = np.random.randn(n1, 2) + np.array([0, 0])
    y1 = np.ones(n1)
    
    X = np.vstack([X0, X1, X2])
    y = np.concatenate([y0, y1, y2])
    
    ovr = OneVsRestClassifier(LogisticRegression())
    ovr.fit(X, y)
    y_pred_ovr = ovr.predict(X)
    
    softmax = LogisticRegression(solver='lbfgs')
    softmax.fit(X, y)
    y_pred_softmax = softmax.predict(X)
    
    mask = (y == 1)
    ovr_recalls.append(np.mean(y_pred_ovr[mask] == 1))
    softmax_recalls.append(np.mean(y_pred_softmax[mask] == 1))
    
    print(f"N1={n1:3d}: OvR={ovr_recalls[-1]:.3f}, Softmax={softmax_recalls[-1]:.3f}")

# recall plot
plt.figure(figsize=(10, 6))
plt.plot(middle_cluster_sizes, softmax_recalls, 'b-o', label='Softmax', linewidth=2)
plt.plot(middle_cluster_sizes, ovr_recalls, 'r-s', label='OvR', linewidth=2)
plt.xscale('log')
plt.gca().invert_xaxis()
plt.xlabel('Class 1 Size (N1)')
plt.ylabel('Class 1 Recall')
plt.title('Class 1 Recall vs Class Imbalance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(middle_cluster_sizes, middle_cluster_sizes)
plt.tight_layout()
plt.savefig('part_g_recall_plot.png', dpi=150)
plt.show()

# data geometry plot
np.random.seed(42)
X0 = np.random.randn(N_MAJORITY, 2) + np.array([-3, 0])
X2 = np.random.randn(N_MAJORITY, 2) + np.array([3, 0])
X1 = np.random.randn(25, 2) + np.array([0, 0])

plt.figure(figsize=(10, 6))
plt.scatter(X0[:, 0], X0[:, 1], c='blue', label='Class 0 (N=500)', alpha=0.6, s=30)
plt.scatter(X1[:, 0], X1[:, 1], c='green', label='Class 1 (N=25)', alpha=0.6, s=30)
plt.scatter(X2[:, 0], X2[:, 1], c='red', label='Class 2 (N=500)', alpha=0.6, s=30)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Data Geometry')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('part_g_data_geometry.png', dpi=150)
plt.show()

# check weights at extreme imbalance
np.random.seed(42)
X0 = np.random.randn(N_MAJORITY, 2) + np.array([-3, 0])
y0 = np.zeros(N_MAJORITY)
X2 = np.random.randn(N_MAJORITY, 2) + np.array([3, 0])
y2 = np.ones(N_MAJORITY) * 2
X1 = np.random.randn(5, 2) + np.array([0, 0])
y1 = np.ones(5)

X = np.vstack([X0, X1, X2])
y = np.concatenate([y0, y1, y2])

ovr = OneVsRestClassifier(LogisticRegression())
ovr.fit(X, y)
print(f"\nOvR Class 1 intercept (N1=5): {ovr.estimators_[1].intercept_[0]:.2f}")

# PART H

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining class counts:")
for digit in range(10):
    print(f"  {digit}: {np.sum(y_train == digit)}")

C_values = np.linspace(0.01, 4, 20)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# softmax grid search
softmax = LogisticRegression(solver='lbfgs', max_iter=1000)
grid_softmax = GridSearchCV(softmax, {'C': C_values}, cv=cv, scoring='accuracy')
grid_softmax.fit(X_train_scaled, y_train)

# ovr grid search
ovr = OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=1000))
grid_ovr = GridSearchCV(ovr, {'estimator__C': C_values}, cv=cv, scoring='accuracy')
grid_ovr.fit(X_train_scaled, y_train)

print(f"\nSoftmax: best C={grid_softmax.best_params_['C']:.2f}, "
      f"CV acc={grid_softmax.best_score_:.3f} +/- {grid_softmax.cv_results_['std_test_score'][grid_softmax.best_index_]:.3f}")
print(f"OvR: best C={grid_ovr.best_params_['estimator__C']:.2f}, "
      f"CV acc={grid_ovr.best_score_:.3f} +/- {grid_ovr.cv_results_['std_test_score'][grid_ovr.best_index_]:.3f}")

# cv accuracy plot
plt.figure(figsize=(10, 6))

softmax_mean = grid_softmax.cv_results_['mean_test_score']
softmax_std = grid_softmax.cv_results_['std_test_score']
plt.plot(C_values, softmax_mean, 'b-o', label='Softmax', linewidth=2)
plt.fill_between(C_values, softmax_mean - softmax_std, softmax_mean + softmax_std, alpha=0.2, color='blue')

ovr_mean = grid_ovr.cv_results_['mean_test_score']
ovr_std = grid_ovr.cv_results_['std_test_score']
plt.plot(C_values, ovr_mean, 'r-s', label='OvR', linewidth=2)
plt.fill_between(C_values, ovr_mean - ovr_std, ovr_mean + ovr_std, alpha=0.2, color='red')

plt.xlabel('C')
plt.ylabel('Mean CV Accuracy')
plt.title('CV Accuracy vs Regularization (shaded = +/- 1 SD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('part_h_cv_accuracy.png', dpi=150)
plt.show()

# refit with best C and evaluate
best_softmax = LogisticRegression(solver='lbfgs', max_iter=1000, C=grid_softmax.best_params_['C'])
best_softmax.fit(X_train_scaled, y_train)
y_pred_softmax = best_softmax.predict(X_test_scaled)

best_ovr = OneVsRestClassifier(
    LogisticRegression(solver='lbfgs', max_iter=1000, C=grid_ovr.best_params_['estimator__C'])
)
best_ovr.fit(X_train_scaled, y_train)
y_pred_ovr = best_ovr.predict(X_test_scaled)

print(f"\nTest accuracy - Softmax: {np.mean(y_pred_softmax == y_test):.3f}")
print(f"Test accuracy - OvR: {np.mean(y_pred_ovr == y_test):.3f}")

# confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_softmax = confusion_matrix(y_test, y_pred_softmax)
ConfusionMatrixDisplay(cm_softmax, display_labels=range(10)).plot(ax=axes[0], cmap='Blues')
axes[0].set_title(f'Softmax (acc={np.mean(y_pred_softmax == y_test):.3f})')

cm_ovr = confusion_matrix(y_test, y_pred_ovr)
ConfusionMatrixDisplay(cm_ovr, display_labels=range(10)).plot(ax=axes[1], cmap='Reds')
axes[1].set_title(f'OvR (acc={np.mean(y_pred_ovr == y_test):.3f})')

plt.tight_layout()
plt.savefig('part_h_confusion_matrices.png', dpi=150)
plt.show()

# most confused pairs
print("\nMost confused pairs:")
for name, cm in [('Softmax', cm_softmax), ('OvR', cm_ovr)]:
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    confusions = []
    for i in range(10):
        for j in range(10):
            if cm_copy[i, j] > 0:
                confusions.append((cm_copy[i, j], i, j))
    confusions.sort(reverse=True)
    print(f"  {name}: {[(t, p) for c, t, p in confusions[:3]]}")