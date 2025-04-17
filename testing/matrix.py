import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load CSV predictions
csv_path = 'xception_test_predictions_20.csv'
df = pd.read_csv(csv_path)

# Extract true and predicted labels
y_true = df['true_label']
y_pred = df['predicted_label']

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
labels = ['Real', 'Fake']

# Display confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Xception Test Set')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()