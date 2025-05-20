import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import Levenshtein
from matplotlib.patches import FancyBboxPatch

# Set Seaborn style
sns.set_style("whitegrid")

# Configure fonts
plt.rcParams['font.family'] = 'Lohit Devanagari'  # For Devanagari text
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']  # Fallback for English text
plt.rcParams['font.size'] = 12

# Load the .tsv file
df = pd.read_csv("predictions/prediction_attention.tsv", sep="\t", names=["english", "predicted", "actual"])

# Randomly sample 5 rows
sampled_df = df.sample(10, random_state=150).reset_index(drop=True)

# Compute edit distances
sampled_df["edit_distance"] = sampled_df.apply(
    lambda row: Levenshtein.distance(row["predicted"], row["actual"]), axis=1
)

# Create figure
fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')

# Use gradient color
colors = sns.color_palette("Blues", n_colors=len(sampled_df))
bars = ax.barh(sampled_df["english"], sampled_df["edit_distance"], color=colors, edgecolor='navy', linewidth=0.5)

# Add shadow effect
for bar in bars:
    bar.set_zorder(1)
    shadow = FancyBboxPatch(
        (bar.get_x(), bar.get_y()), bar.get_width(), bar.get_height(),
        boxstyle="round,pad=0.02", fc=(0, 0, 0, 0.1), ec='none', zorder=0
    )
    ax.add_patch(shadow)

# Annotate bars
for bar, pred, actual, edit_dist in zip(bars, sampled_df["predicted"], sampled_df["actual"], sampled_df["edit_distance"]):
    ax.text(
        bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
        f"Pred: {pred} | Actual: {actual}",
        va='center', ha='left', fontsize=10, color='black',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
    )

# Customize axes
ax.set_xlabel("Edit Distance", fontsize=14, weight='bold')
ax.set_title("Edit Distance Between Predicted and Actual Hindi Words", fontsize=16, weight='bold', pad=15)
ax.set_ylabel("English Words", fontsize=14, weight='bold')

# Add grid lines
ax.grid(True, axis='x', linestyle='--', alpha=0.7)
ax.grid(False, axis='y')

# Adjust margins and spines
ax.margins(y=0.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')

# Adjust layout
plt.tight_layout(pad=2.0)

# Save plot
plt.savefig("edit_distance_comparison.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()