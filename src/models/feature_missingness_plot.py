import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# Load the JSON data
json_path = Path(__file__).parent.parent.parent / "reports" / "feature_missingness_study_results.json"

with open(json_path, 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data['data'], columns=data['columns'])

# Apply Gaussian smoothing to the ROC AUC values
df['metric_roc_auc_weighted_smooth'] = gaussian_filter1d(df['metric_roc_auc_weighted'], sigma=1.5)

# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='step', y='metric_roc_auc_weighted_smooth', marker='o', linewidth=2.5, markersize=6)

# Rename axes
plt.xlabel('Number of features dropped', fontsize=12)
plt.ylabel('ROC AUC weighted', fontsize=12)
plt.title('Feature Missingness Study: ROC AUC Weighted by Features Dropped', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
output_path = Path(__file__).parent.parent.parent / "reports" / "feature_missingness_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")

plt.show()
