# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 17:14:58 2026

@author: ADMIN
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.metrics import confusion_matrix, mean_absolute_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# === Set folder name ===
name = 'SHAP-TRW-noLonLat'

# === Create directories ===
# Set parent directory (replace with your actual path)
file_dir = "E:/GlobalTreeRing/Out-Table-and-Figure/"
# Get today's date
today = datetime.today().strftime('%Y-%m-%d')
# Concatenate full path
SHAP_dir = os.path.join(file_dir, today)
# Create folder (if not exists)
os.makedirs(SHAP_dir, exist_ok=True)
print(f"✅ Folder created: {SHAP_dir}")

# === Adjust variables ===
columns_to_select = ["phylo_depth", 
                     "Elevation",
                     "TMP_mean", "PRE_mean", "SR_mean", "CO2_mean", "VPD_mean",
                     "TMP_cv", "PRE_cv", "SR_cv", "CO2_cv", "VPD_cv",
                     "TRW_mean", "TRW_cv"]

# === Step 1: Load Data ===
# Create output folder if it doesn't exist
output_folder = os.path.join(SHAP_dir, name)  # Create target folder under SHAP_dir
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Assuming data format is like:
# | temp | precip | co2 | ... | response_type |
df = pd.read_csv('E:/GlobalTreeRing/. Result/Result5-DrivingForces/Driving_TRW_yr_merged_depth.csv')
df = df.dropna()
df = df.drop(['Unnamed: 0'], axis=1)  # Drop a specific column, axis=1 means column

# Filter data
df_selected = df[df.columns[df.columns.str.contains('|'.join(columns_to_select))]]
response_type = 'duration'

# === Data Preprocessing: Treat duration as ordinal classification ===
print("Target variable distribution:")
print(df[response_type].value_counts().sort_index())

# Ensure duration is integer type
df[response_type] = df[response_type].astype(int)

# Create ordinal label mapping
duration_mapping = {
    2: 'Short (2)',
    3: 'Medium (3)', 
    4: 'Long (4)',
    5: 'Very Long (5)'
}

# Create labels for better interpretation
df['duration_label'] = df[response_type].map(duration_mapping)

# === Step 2: Split Feature and Target Columns ===
X = df_selected
y = df[response_type]  # Keep as numeric, but treat as ordinal classification
y_labels = df['duration_label']  # Labels for visualization
labs = df['site_name']

# === Step 3: Split Train and Test Sets ===
X_train, X_test, y_train, y_test, labs_train, labs_test, y_labels_train, y_labels_test = train_test_split(
    X, y, labs, y_labels, test_size=0.3, random_state=42, stratify=y
)

# Save data
data_output_folder = os.path.join(SHAP_dir, name, "DATA")
if not os.path.exists(data_output_folder):
    os.makedirs(data_output_folder)

X_train.to_csv(data_output_folder + "/X_train_" + name + ".csv", index=True)
X_test.to_csv(data_output_folder + "/X_test_" + name + ".csv", index=True)
y_train.to_csv(data_output_folder + "/y_train_" + name + ".csv", index=True)
y_test.to_csv(data_output_folder + "/y_test_" + name + ".csv", index=True)
labs_train.to_csv(data_output_folder + "/labs_train_" + name + ".csv", index=True)
labs_test.to_csv(data_output_folder + "/labs_test_" + name + ".csv", index=True)

# === Step 4: Data Standardization ===
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Save standardized data
X_train_scaled.to_csv(data_output_folder + "/X_train_scaled_" + name + ".csv", index=True)
X_test_scaled.to_csv(data_output_folder + "/X_test_scaled_" + name + ".csv", index=True)

# === Step 5: Random Forest Classifier Training ===
# Use classifier, but consider ordinal nature via logic/metrics later if needed
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# View class order
print(f"Model classes: {sorted(model.classes_)}")

# Briefly check variable importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind='bar', title='Feature Importances', figsize=(12, 6))
plt.tight_layout()
plt.savefig(output_folder + "/feature_importances_"+name+".png", dpi=300, bbox_inches='tight')
plt.close()

# === Step 6: Model Prediction ===
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Save prediction results
predictions_df = pd.DataFrame({
    'lab': labs_test,
    'True_Values': y_test,
    'True_Labels': y_labels_test,
    'Predictions': y_pred,
    'Pred_Labels': [duration_mapping[pred] for pred in y_pred]
})

# Add prediction probability for each class
for i, class_val in enumerate(sorted(model.classes_)):
    predictions_df[f'Prob_Class_{class_val}'] = y_pred_proba[:, i]

predictions_df.to_csv(output_folder + "/predictions_"+name+".csv", index=False)

# === Step 7: Model Evaluation (Considering Ordinal Nature) ===
# Basic classification metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Metric specific to ordinal classification: Mean Absolute Error (considers distance between classes)
mae = mean_absolute_error(y_test, y_pred)

# Calculate adjacent accuracy (allowing +/- 1 error)
adjacent_correct = np.sum(np.abs(y_test - y_pred) <= 1) / len(y_test)

evaluation_metrics = {
    'Accuracy': accuracy,
    'F1_Score_weighted': f1,
    'Precision_weighted': precision,
    'Recall_weighted': recall,
    'Mean_Absolute_Error': mae,
    'Adjacent_Accuracy': adjacent_correct  # Accuracy allowing +/- 1 error
}

eval_df = pd.DataFrame([evaluation_metrics])
eval_df.to_csv(output_folder + "/model_evaluation_metrics_"+name+".csv", index=False)
print("Model Evaluation Results:")
print(eval_df)

# === Step 8: Confusion Matrix Visualization ===
cm = confusion_matrix(y_test, y_pred, labels=sorted(model.classes_))
plt.figure(figsize=(8, 6))

# Create labels
class_labels = [duration_mapping[x] for x in sorted(model.classes_)]

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Duration')
plt.ylabel('True Duration')
plt.title('Confusion Matrix for Duration Prediction')
plt.tight_layout()
plt.savefig(output_folder + "/confusion_matrix_"+name+".png", dpi=300, bbox_inches='tight')
plt.close()

# === Step 9: Extra Visualization for Ordinal Classification ===
# Prediction probability distribution
plt.figure(figsize=(12, 8))
for i, class_val in enumerate(sorted(model.classes_)):
    plt.subplot(2, 2, i+1)
    
    # Get prediction probability distribution for true samples of this class
    true_class_mask = y_test == class_val
    if np.sum(true_class_mask) > 0:
        probs = y_pred_proba[true_class_mask, i]
        plt.hist(probs, bins=20, alpha=0.7, color=f'C{i}')
        plt.title(f'Prediction Probability for True Class {duration_mapping[class_val]}')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig(output_folder + "/probability_distributions_"+name+".png", dpi=300, bbox_inches='tight')
plt.close()

# SHAP Interpretability Analysis - Correctly Handling 3D Array Version ====================================================================
print("Starting SHAP analysis...")

# === Step 1: Initialize SHAP Explainer ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

print("=== SHAP Value Structure Confirmation ===")
print(f"SHAP values shape: {shap_values.shape}")  # Should be (694, 16, 4)
print(f"Model classes: {sorted(model.classes_)}")

# === Step 2: Correctly Save SHAP Values ===
# For 3D array (n_samples, n_features, n_classes), extract by class
for class_idx, class_val in enumerate(sorted(model.classes_)):
    print(f"\nProcessing class {class_val} (Index {class_idx})...")
    
    # Extract SHAP values for this class from 3D array: shape (694, 16)
    shap_values_class = shap_values[:, :, class_idx]
    print(f"SHAP values shape for class {class_val}: {shap_values_class.shape}")
    
    # Convert SHAP values to DataFrame
    shap_values_class_df = pd.DataFrame(shap_values_class, columns=X.columns)
    
    # Reset index to match X_test index
    shap_values_class_df.index = X_test.index
    
    # Add related info
    shap_values_class_df['lab'] = labs_test.values
    shap_values_class_df['true_values'] = y_test.values
    shap_values_class_df['predictions'] = y_pred
    shap_values_class_df['class_probability'] = y_pred_proba[:, class_idx]
    shap_values_class_df['base_value'] = explainer.expected_value[class_idx]
    
    # Save to CSV file
    shap_values_class_df.to_csv(
        os.path.join(output_folder, f"shap_values_duration_{class_val}_{name}.csv"),
        index=True
    )
    print(f"Saved SHAP values for class {class_val}, DataFrame shape: {shap_values_class_df.shape}")

print("✅ SHAP values saving completed")

# === SHAP Visualization Analysis ===
shap_viz_folder = os.path.join(output_folder, "SHAP_Visualizations")
if not os.path.exists(shap_viz_folder):
    os.makedirs(shap_viz_folder)

print("Starting to generate SHAP visualization charts...")

# Convert 3D SHAP array to list format for SHAP visualization functions
shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
print(f"Converted to list format, containing {len(shap_values_list)} classes")

# ===== 1. Generate Summary Plot for Each Class =====
print("Generating Summary Plots...")
for class_idx, class_val in enumerate(sorted(model.classes_)):
    plt.figure(figsize=(12, 8))
    
    shap.summary_plot(
        shap_values_list[class_idx], 
        X_test_scaled, 
        feature_names=X.columns,
        show=False,
        max_display=len(X.columns)
    )
    
    plt.title(f'SHAP Summary Plot - Duration {duration_mapping[class_val]}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(shap_viz_folder, f"summary_plot_duration_{class_val}_{name}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Summary Plot for class {class_val}")

# ===== 2. Comprehensive Summary Plot (Mean Absolute Effect of All Classes) =====
print("Generating Comprehensive Summary Plot...")
plt.figure(figsize=(12, 8))

# Calculate mean absolute value of SHAP values across all classes
mean_abs_shap = np.mean(np.abs(shap_values), axis=2)
print(f"Mean absolute SHAP values shape: {mean_abs_shap.shape}")

shap.summary_plot(
    mean_abs_shap, 
    X_test_scaled, 
    feature_names=X.columns,
    show=False,
    max_display=len(X.columns)
)
plt.title('SHAP Summary Plot - All Duration Classes (Mean Absolute Impact)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(shap_viz_folder, f"summary_plot_all_classes_{name}.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# ===== 3. Bar Plot (Feature Importance) =====
print("Generating Bar Plots...")
for class_idx, class_val in enumerate(sorted(model.classes_)):
    plt.figure(figsize=(12, 8))
    
    shap.summary_plot(
        shap_values_list[class_idx], 
        X_test_scaled, 
        feature_names=X.columns,
        plot_type="bar",
        show=False,
        max_display=len(X.columns)
    )
    
    plt.title(f'SHAP Feature Importance - Duration {duration_mapping[class_val]}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(shap_viz_folder, f"bar_plot_duration_{class_val}_{name}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
# ===== 3-2. Comprehensive Bar Plot (Mean Absolute Effect of All Classes) =====
print("Generating Comprehensive Summary Plot...")
plt.figure(figsize=(12, 8))

# Calculate mean absolute value of SHAP values across all classes
mean_abs_shap = np.mean(np.abs(shap_values), axis=2)
print(f"Mean absolute SHAP values shape: {mean_abs_shap.shape}")

shap.summary_plot(
    mean_abs_shap, 
    X_test_scaled, 
    feature_names=X.columns,
    plot_type="bar",
    show=False,
    max_display=len(X.columns)
)
plt.title('SHAP bar Plot - All Duration Classes (Mean Absolute Impact)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(shap_viz_folder, f"bar_plot_all_classes_{name}.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# New Section: Extract data and save as CSV
# ==========================================
# 1. Calculate mean absolute SHAP values
# Assuming shap_values shape is (n_samples, n_features, n_classes)
# axis=2 means averaging over "classes" (comprehensive impact of all Lag durations)
# Result mean_abs_shap_samples shape is (n_samples, n_features)
mean_abs_shap_samples = np.mean(np.abs(shap_values), axis=2)

# 2. Plotting (Your original code)
plt.figure(figsize=(12, 8))
shap.summary_plot(
    mean_abs_shap_samples, 
    X_test_scaled, 
    feature_names=X.columns,
    plot_type="bar",
    show=False,
    max_display=len(X.columns)
)
plt.title('SHAP Bar Plot - All Duration Classes (Mean Absolute Impact)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(shap_viz_folder, f"bar_plot_all_classes_{name}.png"), 
            dpi=300, bbox_inches='tight')
plt.close()

# 3. Calculate Global Importance for each feature
# Average over axis=0 (all samples) to get a final importance value per feature
feature_importance_values = np.mean(mean_abs_shap_samples, axis=0)

# 4. Create DataFrame
df_importance = pd.DataFrame({
    'Feature': X.columns,
    'Mean_SHAP_Value': feature_importance_values
})

# 5. Sort by importance descending (consistent with plot order)
df_importance = df_importance.sort_values(by='Mean_SHAP_Value', ascending=False)

# 6. Save to CSV file
csv_path = os.path.join(shap_viz_folder, f"feature_importance_data_{name}.csv")
df_importance.to_csv(csv_path, index=False)

print(f"Image saved: bar_plot_all_classes_{name}.png")
print(f"Data saved: {csv_path}")

# Print top 5 to check
print("\nTop 5 Important Drivers:")
print(df_importance.head(5))


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ==========================================
# 1. Define Variable Groups and Colors (Based on your classification)
# ==========================================

# Define grouping dictionary
feature_groups = {
    "Phylogenetic": ["phylo_depth"],
    "Spatial": ["Elevation"],
    "Climate Mean": ["TMP_mean", "PRE_mean", "SR_mean", "VPD_mean", "CO2_mean"],
    "Climate Variability": ["TMP_cv", "PRE_cv", "SR_cv", "VPD_cv", "CO2_cv"],
    "Vegetation State": ["TRW_mean", "TRW_cv"]
}

# Define color scheme (You can modify color codes)
group_colors = {
  "Phylogenetic": "#a29bfe",      
  "Spatial": "#ff7f0e",       
  "Climate Mean":  "#fdcb6e", 
  "Climate Variability":   "#00b894", #"#00cec9",
  "Vegetation State": "#e377c2"     
}


# ==========================================
# 1. Define Settings (Keep unchanged)
# ==========================================
# Define grouping dictionary
feature_groups = {
    "Phylogenetic": ["phylo_depth"],
    "Spatial geographic": ["Elevation"],
    "Moisture-related": ["PRE_mean", "PRE_cv", "VPD_mean", "VPD_cv"],
    "Engery-related": ["TMP_mean", "TMP_cv", "SR_mean", "SR_cv", "CO2_mean", "CO2_cv"],
    "Vegetation state": ["TRW_mean", "TRW_cv"]
}

# Define color scheme
group_colors = {
  "Phylogenetic": "#a29bfe",
  "Spatial geographic": "#ff7f0e",
  "Moisture-related": "#00b894",
  "Engery-related": "#fdcb6e",
  "Vegetation state": "#e377c2"
}


# ==========================================
# 2. Data Processing
# ==========================================

# A. Calculate Average Feature Importance (Keep unchanged)
if len(shap_values.shape) == 3:
    mean_shap = np.mean(np.mean(np.abs(shap_values), axis=2), axis=0)
else:
    mean_shap = np.mean(np.abs(shap_values), axis=0)

# B. Create DataFrame (Keep unchanged)
df_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': mean_shap
})

# C. Map Categories (Keep unchanged)
feat_to_group = {}
for group, feats in feature_groups.items():
    for f in feats:
        feat_to_group[f] = group

df_imp['Category'] = df_imp['Feature'].map(feat_to_group).fillna('Other')

# --- [New Step] D. Rename Feature Labels for Plotting ---
def rename_feature_label(feature_name):
    if feature_name == "phylo_depth":
        return "Phylogenetic depth"
    elif feature_name == "Elevation":
        return "Elevation"
    elif feature_name.endswith("_cv"):
        # Replace "VPD_cv" with "VPD Variation"
        base_name = feature_name.replace("_cv", "")
        return f"{base_name} Variation"
    elif feature_name.endswith("_mean"):
        # Replace "VPD_mean" with "Average VPD"
        base_name = feature_name.replace("_mean", "")
        return f"Average {base_name}"
    else:
        return feature_name

# Apply renaming function to create new column
df_imp['Feature_Label'] = df_imp['Feature'].apply(rename_feature_label)
# ---------------------------------------

# Sort by importance (Keep unchanged)
df_imp = df_imp.sort_values(by='Importance', ascending=True)

# Save to CSV (Suggested to save version with new labels)
csv_path = os.path.join(shap_viz_folder, f"feature_importance_data2_{name}.csv")
df_imp.to_csv(csv_path, index=False)


# ==========================================
# 3. Plot 1: All Features Sorted by Importance (Modified - with divider line)
# ==========================================

# 1. Create canvas
fig, ax = plt.subplots(figsize=(8, 6))

# 2. Draw horizontal bar chart
bars = ax.barh(df_imp['Feature_Label'], df_imp['Importance'])

# 3. Color the bars
for bar, original_feature in zip(bars, df_imp['Feature']):
    category = df_imp.loc[df_imp['Feature'] == original_feature, 'Category'].values[0]
    bar.set_color(group_colors.get(category, '#95a5a6'))

# 4. Add legend
handles = [plt.Rectangle((0,0),1,1, color=color) for color in group_colors.values()]
labels = group_colors.keys()
ax.legend(handles, labels, title="Driver Groups", loc="lower right")

# 5. Set axis labels
ax.set_xlabel("Average Impact (Mean |SHAP value|)")

# 6. Set title (a)
ax.text(-0.15, 1.05, '(a)', transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='bottom', ha='left')

# 7. Grid lines
ax.grid(axis='x', linestyle='--', alpha=0.5)

# --- [New Modification] Add a line below the 4th ranked feature ---
# df_imp is sorted ascending, so largest is at the end (index = len - 1)
# The 4th largest index position is len(df_imp) - 4
# Line should be drawn between 4th and 5th largest, so subtract 0.5
if len(df_imp) >= 4:
    line_y = len(df_imp) - 4 - 0.5
    # axhline draws horizontal line
    # color='black', linestyle='--', linewidth=1.5
    ax.axhline(y=line_y, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

# 8. Save and show
plt.tight_layout()
save_path_1 = os.path.join(shap_viz_folder, f"feature_importance_colored_{name}.png")
plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
plt.show()


# ===== 4. Dependence Plot (Colored by phylo_depth) =====
print("Generating Dependence Plots...")

# Select most important features for dependence plot analysis
n_top_features = 6

for class_idx, class_val in enumerate(sorted(model.classes_)):
    # Calculate feature importance for this class
    feature_importance = np.mean(np.abs(shap_values_list[class_idx]), axis=0)
    top_features_idx = np.argsort(feature_importance)[::-1][:n_top_features]
    top_features = [X.columns[i] for i in top_features_idx]
    
    print(f"Duration {class_val} - Top features: {top_features}")
    
    for feature in top_features:
        if feature == 'phylo_depth':
            # If feature itself is phylo_depth, choose another important feature as color
            other_features = [f for f in top_features if f != 'phylo_depth']
            color_feature = other_features[0] if other_features else 'auto'
        else:
            color_feature = 'phylo_depth' if 'phylo_depth' in X.columns else 'auto'
        
        plt.figure(figsize=(10, 6))
        
        try:
            shap.dependence_plot(
                feature, 
                shap_values_list[class_idx], 
                X_test_scaled, 
                feature_names=X.columns,
                interaction_index=color_feature,
                show=False
            )
            
            plt.title(f'SHAP Dependence Plot - {feature}\nDuration {duration_mapping[class_val]}', 
                     fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(shap_viz_folder, 
                                     f"dependence_{feature}_duration_{class_val}_{name}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Generated dependence plot for {feature} (Class {class_val})")
            
        except Exception as e:
            print(f"Could not generate dependence plot for feature {feature} (Duration {class_val}): {e}")
            plt.close()

# ===== 5. Waterfall Plot (Showing individual sample predictions) =====
print("Generating Waterfall Plots...")

# Select representative samples for each class
n_samples_per_class = 2

for class_idx, class_val in enumerate(sorted(model.classes_)):
    # Select samples with highest prediction probability for this class
    class_probs = y_pred_proba[:, class_idx]
    top_samples_idx = np.argsort(class_probs)[::-1][:n_samples_per_class]
    
    for sample_idx in top_samples_idx:
        plt.figure(figsize=(12, 8))
        
        try:
            # Create Explanation object
            explanation = shap.Explanation(
                values=shap_values_list[class_idx][sample_idx], 
                base_values=explainer.expected_value[class_idx],
                data=X_test_scaled.iloc[sample_idx].values,
                feature_names=X.columns.tolist()
            )
            
            shap.waterfall_plot(explanation, show=False, max_display=12)
            
            plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}\n' + 
                     f'Duration {duration_mapping[class_val]} (Prob: {class_probs[sample_idx]:.3f})', 
                     fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(shap_viz_folder, 
                                     f"waterfall_sample_{sample_idx}_duration_{class_val}_{name}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Could not generate waterfall plot for sample {sample_idx} (Duration {class_val}): {e}")
            plt.close()

# ===== 6. Special Analysis for phylo_depth =====
if 'phylo_depth' in X.columns:
    print("Generating special analysis for phylo_depth...")
    
    phylo_idx = list(X.columns).index('phylo_depth')
    phylo_values = X_test_scaled['phylo_depth'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for class_idx, class_val in enumerate(sorted(model.classes_)):
        phylo_shap = shap_values[:, phylo_idx, class_idx]  # Extract from 3D array
        
        # Use true duration values as color
        scatter = axes[class_idx].scatter(phylo_values, phylo_shap, 
                                        c=y_test.values, cmap='viridis', 
                                        alpha=0.6, s=50)
        
        axes[class_idx].set_xlabel('Phylogenetic Depth (standardized)')
        axes[class_idx].set_ylabel('SHAP Value')
        axes[class_idx].set_title(f'phylo_depth SHAP - Duration {duration_mapping[class_val]}')
        
        # Add trend line
        try:
            z = np.polyfit(phylo_values, phylo_shap, 1)
            p = np.poly1d(z)
            axes[class_idx].plot(phylo_values, p(phylo_values), "r--", alpha=0.8)
            
            # Calculate correlation coefficient
            corr = np.corrcoef(phylo_values, phylo_shap)[0, 1]
            axes[class_idx].text(0.05, 0.95, f'Corr: {corr:.3f}', 
                                transform=axes[class_idx].transAxes, 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        except:
            pass
    
    plt.tight_layout()
    plt.savefig(os.path.join(shap_viz_folder, f"phylo_depth_analysis_{name}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated phylo_depth special analysis plot")

# ===== 7. Generate Feature Importance Summary =====
print("Generating feature importance summary...")
summary_data = []

for class_idx, class_val in enumerate(sorted(model.classes_)):
    feature_importance = np.mean(np.abs(shap_values_list[class_idx]), axis=0)
    
    for feat_idx, feature in enumerate(X.columns):
        summary_data.append({
            'Duration_Class': class_val,
            'Duration_Label': duration_mapping[class_val],
            'Feature': feature,
            'Mean_Abs_SHAP': feature_importance[feat_idx],
            'Mean_SHAP': np.mean(shap_values_list[class_idx][:, feat_idx])
        })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(shap_viz_folder, f"feature_importance_summary_{name}.csv"), index=False)

print("✅ Ordinal classification SHAP analysis completed!")
print(f"✅ All files saved to: {shap_viz_folder}")

# Show most important features for each duration class
print("\nMost important features for each duration category:")
for class_val in sorted(model.classes_):
    class_summary = summary_df[summary_df['Duration_Class'] == class_val].nlargest(5, 'Mean_Abs_SHAP')
    print(f"\n{duration_mapping[class_val]}:")
    print(class_summary[['Feature', 'Mean_Abs_SHAP']].to_string(index=False))

# Specifically check performance of phylo_depth in each class
if 'phylo_depth' in X.columns:
    print("\nSHAP importance of phylo_depth in each class:")
    phylo_summary = summary_df[summary_df['Feature'] == 'phylo_depth']
    print(phylo_summary[['Duration_Label', 'Mean_Abs_SHAP', 'Mean_SHAP']].to_string(index=False))


# Ridgeline Plot - Single Plot =======================================
# Enhanced SHAP Cross-Class Comparison Visualization - Improved Color Mapping
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import string

# Set global font size and family
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})


def create_single_feature_comparison_plot(shap_values, X_test_scaled, model, 
                                          duration_mapping, feature_name, 
                                          color_mapping='percentile',  # New parameter
                                          save_path=None, show_plot=True):
    """
    Create SHAP comparison plot for a single feature across different duration classes
    
    Parameters:
    -----------
    color_mapping : str, default 'percentile'
        Color mapping method:
        - 'percentile': Based on 25%-75% percentiles, ensures uniform color distribution
        - 'symmetric': Symmetric mapping based on median
        - 'standard': Standard mapping (original method)
        - 'robust': Robust mapping based on 1.5 * IQR
    """
    
    # Ensure feature exists
    if feature_name not in X_test_scaled.columns:
        print(f"Feature {feature_name} not found in data")
        return None
    
    # Get feature index
    feature_idx = list(X_test_scaled.columns).index(feature_name)
    
    # Prepare data
    classes = sorted(model.classes_)
    n_classes = len(classes)
    
    # Extract SHAP values and feature values
    shap_data = []
    feature_values = X_test_scaled[feature_name].values
    
    for class_idx, class_val in enumerate(classes):
        # Extract feature SHAP values for this class from 3D SHAP array
        feature_shap_values = shap_values[:, feature_idx, class_idx]
        
        # Calculate statistics
        mean_shap = np.mean(feature_shap_values)
        
        # Calculate importance rank of this feature in this class
        class_feature_importance = np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)
        feature_rank = np.where(np.argsort(class_feature_importance)[::-1] == feature_idx)[0][0] + 1
        
        for i, (shap_val, feature_val) in enumerate(zip(feature_shap_values, feature_values)):
            shap_data.append({
                'duration_class': class_val,
                'duration_label': duration_mapping[class_val],
                'shap_value': shap_val,
                'feature_value': feature_val,
                'mean_shap': mean_shap,
                'feature_rank': feature_rank,
                'sample_idx': i
            })
    
    # Convert to DataFrame
    df_plot = pd.DataFrame(shap_data)
    
    # Set color range based on selected mapping method
    feature_vals = df_plot['feature_value'].values
    
    if color_mapping == 'percentile':
        # Mapping based on 25%-75% percentiles, ensures uniform color distribution
        vmin = np.percentile(feature_vals, 25)
        vmax = np.percentile(feature_vals, 75)
        print(f"Using percentile mapping: 25%ile={vmin:.3f}, 75%ile={vmax:.3f}")
        
    elif color_mapping == 'symmetric':
        # Symmetric mapping based on median
        median_val = np.median(feature_vals)
        mad = np.median(np.abs(feature_vals - median_val))  # Median absolute deviation
        vmin = median_val - 2 * mad
        vmax = median_val + 2 * mad
        print(f"Using symmetric mapping: Median={median_val:.3f}, Range=[{vmin:.3f}, {vmax:.3f}]")
        
    elif color_mapping == 'robust':
        # Robust mapping based on IQR
        q25, q75 = np.percentile(feature_vals, [25, 75])
        iqr = q75 - q25
        vmin = q25 - 0.5 * iqr
        vmax = q75 + 0.5 * iqr
        print(f"Using robust mapping: IQR Range=[{vmin:.3f}, {vmax:.3f}]")
        
    else:  # 'standard'
        # Standard mapping (original method)
        vmin, vmax = feature_vals.min(), feature_vals.max()
        print(f"Using standard mapping: Min={vmin:.3f}, Max={vmax:.3f}")
    
    # Expand range if vmin and vmax are too close
    if abs(vmax - vmin) < 1e-6:
        center = (vmin + vmax) / 2
        vmin = center - 0.1
        vmax = center + 0.1
        print(f"Range too small, expanded to: [{vmin:.3f}, {vmax:.3f}]")
    
    # Create chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set y-axis positions
    y_positions = np.arange(n_classes)
    y_labels = [f"Duration {duration_mapping[cls]}\n(Rank: #{df_plot[df_plot['duration_class']==cls]['feature_rank'].iloc[0]})" 
                for cls in classes]
    
    # Create normalize object
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot scatter for each class
    for i, class_val in enumerate(classes):
        class_data = df_plot[df_plot['duration_class'] == class_val]
        
        # Add some vertical jitter to avoid overlap
        y_pos = np.full(len(class_data), i) + np.random.normal(0, 0.08, len(class_data))
        
        # Set color based on feature value (Red for high, Blue for low)
        scatter = ax.scatter(class_data['shap_value'], y_pos, 
                           c=class_data['feature_value'], 
                           cmap='coolwarm',  # Red-Yellow-Blue colormap
                           norm=norm,  # Use custom normalization
                           alpha=0.6, 
                           s=25,
                           edgecolors=None, 
                           linewidth=0.5)
        
        # Add mean vertical line
        mean_shap = class_data['mean_shap'].iloc[0]
        ax.axvline(x=mean_shap, ymin=(i-0.35)/n_classes, ymax=(i+0.35)/n_classes, 
                  color='black', linewidth=4, alpha=0.9)
        
        # Add mean label
        ax.text(mean_shap, i + 0.38, f'Mean: {mean_shap:.3f}', 
               ha='center', va='bottom', fontsize=14, fontweight='bold',
               color='black', family='Arial')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label(f'{feature_name} Feature Value', fontsize=16, fontweight='bold', family='Arial')
    cbar.ax.tick_params(labelsize=14)
    
    # Add key percentile markers on colorbar
    if color_mapping == 'percentile':
        # Add 25%, 50%, 75% percentile markers
        percentiles = [25, 50, 75]
        perc_values = [np.percentile(feature_vals, p) for p in percentiles]
        for p, val in zip(percentiles, perc_values):
            if vmin <= val <= vmax:
                cbar.ax.axhline(y=val, color='white', linewidth=2, alpha=0.8)
                cbar.ax.text(1.1, val, f'{p}%', va='center', ha='left', 
                           fontsize=12, color='black', fontweight='bold')
    
    # Set axis properties
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=15, family='Arial')
    ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=18, fontweight='bold', family='Arial')
    ax.set_ylabel('Duration Classes', fontsize=18, fontweight='bold', family='Arial')
    
    # Add zero line
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.6, linewidth=2)
    
    # Set title
    mapping_desc = {
        'percentile': '25%-75% Percentile',
        'symmetric': 'Symmetric (Median±2MAD)',
        'robust': 'Robust (IQR±0.5*IQR)',
        'standard': 'Standard (Min-Max)'
    }
    
    ax.set_title(f'{feature_name} SHAP Values Across Duration Classes\n' + 
                f'Color mapping: {mapping_desc[color_mapping]} | Black lines: mean SHAP values',
                fontsize=18, fontweight='bold', pad=25, family='Arial')
    
    # Add grid
    ax.grid(True, alpha=0.4, axis='x')
    
    # Set background color zones
    for i in range(n_classes):
        if i % 2 == 0:
            ax.add_patch(Rectangle((ax.get_xlim()[0], i-0.4), 
                                 ax.get_xlim()[1] - ax.get_xlim()[0], 0.8, 
                                 facecolor='lightgray', alpha=0.15, zorder=0))
    
    # Add color mapping info text
    info_text = f"Color Range: [{vmin:.3f}, {vmax:.3f}]\nMapping: {mapping_desc[color_mapping]}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=12, family='Arial')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Return statistics
    stats_info = {
        'feature_stats': {
            'min': feature_vals.min(),
            'max': feature_vals.max(),
            'mean': feature_vals.mean(),
            'median': np.median(feature_vals),
            'q25': np.percentile(feature_vals, 25),
            'q75': np.percentile(feature_vals, 75),
        },
        'color_mapping': color_mapping,
        'color_range': (vmin, vmax),
        'data': df_plot
    }
    
    return stats_info


# ===================
# Ridgeline Plot - Multi-Plot =======
# Enhanced SHAP Cross-Class Comparison Visualization - Multi-Feature Panel Version
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import string

# Set global font size and family
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
})


from matplotlib.colors import Normalize
def create_single_subplot_comparison(ax, shap_values, X_test_scaled, model, 
                                   duration_mapping, feature_name, 
                                   text_a = 20, text_b = 20, text_c = 26,
                                   color_mapping='percentile', subplot_label='',
                                   feature_title_mapping=None):
    """
    Create SHAP comparison plot for a single feature on a given subplot
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Subplot axis object
    subplot_label : str
        Subplot label (e.g., 'a', 'b', 'c', etc.)
    Other parameters same as original function
    """
    
    # Ensure feature exists
    if feature_name not in X_test_scaled.columns:
        print(f"Feature {feature_name} not found in data")
        return None
    
    # Get feature index
    feature_idx = list(X_test_scaled.columns).index(feature_name)
    
    # Prepare data
    classes = sorted(model.classes_)
    n_classes = len(classes)
    
    # Extract SHAP values and feature values
    shap_data = []
    feature_values = X_test_scaled[feature_name].values
    
    for class_idx, class_val in enumerate(classes):
        # Extract feature SHAP values for this class from 3D SHAP array
        feature_shap_values = shap_values[:, feature_idx, class_idx]
        
        # Calculate statistics
        mean_shap = np.mean(feature_shap_values)
        
        # Calculate importance rank of this feature in this class
        class_feature_importance = np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)
        feature_rank = np.where(np.argsort(class_feature_importance)[::-1] == feature_idx)[0][0] + 1
        
        for i, (shap_val, feature_val) in enumerate(zip(feature_shap_values, feature_values)):
            shap_data.append({
                'duration_class': class_val,
                'duration_label': duration_mapping[class_val],
                'shap_value': shap_val,
                'feature_value': feature_val,
                'mean_shap': mean_shap,
                'feature_rank': feature_rank,
                'sample_idx': i
            })
    
    # Convert to DataFrame
    df_plot = pd.DataFrame(shap_data)
    
    # Set color range based on selected mapping method
    feature_vals = df_plot['feature_value'].values
    
    if color_mapping == 'percentile':
        # Mapping based on 25%-75% percentiles, ensures uniform color distribution
        vmin = np.percentile(feature_vals, 25)
        vmax = np.percentile(feature_vals, 75)
        
    elif color_mapping == 'symmetric':
        # Symmetric mapping based on median
        median_val = np.median(feature_vals)
        mad = np.median(np.abs(feature_vals - median_val))  # Median absolute deviation
        vmin = median_val - 2 * mad
        vmax = median_val + 2 * mad
        
    elif color_mapping == 'robust':
        # Robust mapping based on IQR
        q25, q75 = np.percentile(feature_vals, [25, 75])
        iqr = q75 - q25
        vmin = q25 - 0.5 * iqr
        vmax = q75 + 0.5 * iqr
        
    else:  # 'standard'
        # Standard mapping (original method)
        vmin, vmax = feature_vals.min(), feature_vals.max()
    
    # Expand range if vmin and vmax are too close
    if abs(vmax - vmin) < 1e-6:
        center = (vmin + vmax) / 2
        vmin = center - 0.1
        vmax = center + 0.1
    
    # Set y-axis positions
    y_positions = np.arange(n_classes)
    y_labels = [f"Duration {duration_mapping[cls]}\n(Rank: #{df_plot[df_plot['duration_class']==cls]['feature_rank'].iloc[0]})" 
                for cls in classes]
    
    # Create normalize object
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    
    # Plot scatter for each class
    for i, class_val in enumerate(classes):
        class_data = df_plot[df_plot['duration_class'] == class_val]
        
        # Add some vertical jitter to avoid overlap
        y_pos = np.full(len(class_data), i) + np.random.normal(0, 0.08, len(class_data))
        
        # Set color based on feature value (Red for high, Blue for low)
        scatter = ax.scatter(class_data['shap_value'], y_pos, 
                           c=class_data['feature_value'], 
                           cmap='coolwarm',  # Red-Yellow-Blue colormap
                           norm=norm,  # Use custom normalization
                           alpha=0.6, 
                           s=15,  # Reduce point size for multi-plot
                           edgecolors=None, 
                           linewidth=0.3)
        
        # Add mean vertical line
        mean_shap = class_data['mean_shap'].iloc[0]
        ax.axvline(x=mean_shap, ymin=(i-0.35)/n_classes, ymax=(i+0.35)/n_classes, 
                  color='black', linewidth=3, alpha=0.9)
        
        # Add mean label
        # ax.text(mean_shap - 0.001, i + 0.1, f'{mean_shap:.2f}', 
        #        ha='center', va='bottom', fontsize=text_b, fontweight='bold',
        #        color='black', family='Arial')
    
    # Set axis properties
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=text_b, family='Arial')
    ax.set_xlabel('SHAP Value', fontsize=text_b, fontweight='bold', family='Arial')
    # ax.set_ylabel('Duration Classes', fontsize=text_b, fontweight='bold', family='Arial')
    
    # Enlarge x-axis tick font
    ax.tick_params(axis='x', labelsize=text_b)
    
    # Add zero line
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    
    # Set title
    # ax.set_title(f'{feature_name}', fontsize=text_c, fontweight='bold', pad=8, family='Arial')
    # Use mapped name as title
    display_name = feature_title_mapping.get(feature_name, feature_name) if feature_title_mapping else feature_name
    ax.set_title(display_name, fontsize=text_c, fontweight='bold', pad=10, family='Arial')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Set background color zones
    for i in range(n_classes):
        if i % 2 == 0:
            ax.add_patch(Rectangle((ax.get_xlim()[0], i-0.4), 
                                 ax.get_xlim()[1] - ax.get_xlim()[0], 0.8, 
                                 facecolor='lightgray', alpha=0.1, zorder=0))
    
    # Add subplot label
    # if subplot_label:
    #     ax.text(0.02, 0.98, subplot_label, transform=ax.transAxes, 
    #             fontsize=text_b, fontweight='bold', va='top', ha='left',
    #             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add subplot label: Outside the box, top left, no border
    if subplot_label:
        ax.annotate(subplot_label, xy=(0, 1), xytext=(-30, 10),
                    textcoords='offset points',  # Offset relative to plot boundary
                    xycoords='axes fraction',    # Relative coordinates (0~1)
                    fontsize=text_c, fontweight='bold', ha='right', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='none'))  # No border
    
    return scatter, norm, vmin, vmax


def create_multi_feature_comparison_plot(shap_values, X_test_scaled, model, 
                                         duration_mapping, feature_list, 
                                         color_mapping='percentile',
                                         text_a = 20, text_b = 20, text_c = 26,
                                         fig_width_cm = 40, 
                                         fig_height_cm = 50, 
                                         feature_title_mapping = None,
                                         save_path=None, show_plot=True):
    """
    Create SHAP comparison multi-plot for multiple features (3 rows x 2 cols)
    
    Parameters:
    -----------
    feature_list : list
        List of features to plot, should contain 6 feature names
    """
    
    if len(feature_list) != 6:
        print(f"Error: Need 6 features, but provided {len(feature_list)}")
        return None
    
    # Create 3x2 subplots
    # === Set figure size using cm ===
    fig_width_in = fig_width_cm / 2.54
    fig_height_in = fig_height_cm / 2.54

    fig, axes = plt.subplots(3, 2, figsize=(fig_width_in, fig_height_in))
    axes = axes.flatten()
    
    # Subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    # Create subplot for each feature
    scatters = []
    norms = []
    
    for i, (feature_name, label) in enumerate(zip(feature_list, subplot_labels)):
        print(f"Processing feature {i+1}/6: {feature_name}")
        
        # scatter, norm, vmin, vmax = create_single_subplot_comparison(
        #     axes[i], shap_values, X_test_scaled, model, 
        #     duration_mapping, feature_name, 
        #     color_mapping=color_mapping, subplot_label=label
        # )
        
        scatter, norm, vmin, vmax = create_single_subplot_comparison(
            axes[i], shap_values, X_test_scaled, model, 
            duration_mapping, feature_name, 
            color_mapping=color_mapping, subplot_label=label,
            feature_title_mapping=feature_title_mapping
        )
        
        scatters.append(scatter)
        norms.append((norm, vmin, vmax))
    
    # Adjust subplot spacing
    plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=7.0)
    
    # Add general title
    mapping_desc = {
        'percentile': '25%-75% Percentile',
        'symmetric': 'Symmetric (Median±2MAD)',
        'robust': 'Robust (IQR±0.5*IQR)',
        'standard': 'Standard (Min-Max)'
    }
    
    fig.suptitle(f'SHAP Values Across Duration Classes for Selected Features\n' + 
                f'Color mapping: {mapping_desc[color_mapping]} | Black lines: mean SHAP values',
                fontsize=text_c, fontweight='bold', y=1.02)
    
    # Add colorbar for each subplot
    for i, (scatter, (norm, vmin, vmax)) in enumerate(zip(scatters, norms)):
        # Get full data range for this feature
        feature_vals = X_test_scaled[feature_list[i]].values
        data_min, data_max = feature_vals.min(), feature_vals.max()
        
        # Create custom segmented colormap
        from matplotlib.colors import ListedColormap, BoundaryNorm
        import matplotlib.cm as cm
        
        # Get endpoints of coolwarm colormap
        cmap_base = cm.get_cmap('coolwarm')
        low_color = cmap_base(0.0)    # Bluest color
        high_color = cmap_base(1.0)   # Reddest color
        
        # Create segmented color list
        n_segments = 100
        colors = []
        
        # Low value segment: Solid Blue
        n_low = int(n_segments * (vmin - data_min) / (data_max - data_min))
        colors.extend([low_color] * max(1, n_low))
        
        # Middle segment: Normal Gradient
        n_mid = int(n_segments * (vmax - vmin) / (data_max - data_min))
        mid_colors = [cmap_base(x) for x in np.linspace(0, 1, max(1, n_mid))]
        colors.extend(mid_colors)
        
        # High value segment: Solid Red  
        n_high = n_segments - len(colors)
        colors.extend([high_color] * max(1, n_high))
        
        # Create custom colormap
        custom_cmap = ListedColormap(colors)
        full_norm = Normalize(vmin=data_min, vmax=data_max)
        
        # Create colorbar
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="3%", pad=0.05)
        
        # Create colorbar using custom colormap
        sm = cm.ScalarMappable(norm=full_norm, cmap=custom_cmap)
        cbar = plt.colorbar(sm, cax=cax)
        # cbar.set_label(f'Feature Value', fontsize=text_b, fontweight='bold')
        cbar.ax.tick_params(labelsize=text_b)
        
        # Set ticks and labels for colorbar
        cbar.set_ticks([data_min, vmin, np.median(feature_vals), vmax, data_max])
        cbar.set_ticklabels([f'{data_min:.2f}', f'{vmin:.2f}', 
                           f'{np.median(feature_vals):.2f}', 
                           f'{vmax:.2f}', f'{data_max:.2f}'])
        
        # Add quantile lines
        if color_mapping == 'percentile':
            # Mark 25% and 75% quantile boundaries
            cbar.ax.axhline(y=vmin, color='white', linewidth=2, alpha=0.9)
            cbar.ax.axhline(y=vmax, color='white', linewidth=2, alpha=0.9)
            
            # Add section description (Simplified)
            # cbar.ax.text(1.1, data_min + (vmin-data_min)*0.5, 'Low\n(Solid)', 
            #             transform=cbar.ax.transData, rotation=90, 
            #             ha='center', va='center', fontsize=7, alpha=0.8)
            
            # cbar.ax.text(1.1, vmin + (vmax-vmin)*0.5, 'Gradient\nRange', 
            #             transform=cbar.ax.transData, rotation=90, 
            #             ha='center', va='center', fontsize=7, alpha=0.8)
            
            # cbar.ax.text(1.1, vmax + (data_max-vmax)*0.5, 'High\n(Solid)', 
            #             transform=cbar.ax.transData, rotation=90, 
            #             ha='center', va='center', fontsize=7, alpha=0.8)
    
    # Save image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig



# === Main Execution Code ===

# Create output folders
cross_class_folder = os.path.join(output_folder, "Cross_Class_SHAP_Analysis")
single_plots_folder = os.path.join(cross_class_folder, "Single_Feature_Plots")
multi_plots_folder = os.path.join(cross_class_folder, "Multi_Feature_Plots")

for folder in [cross_class_folder, single_plots_folder, multi_plots_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

print("=== Starting generation of cross-class SHAP comparison plots for all features ===\n")

# 1. Generate single plots for all features
all_features = list(X.columns)
print(f"Processing {len(all_features)} features")

for i, feature in enumerate(all_features):
    print(f"Processing feature {i+1}/{len(all_features)}: {feature}")
    
    save_path = os.path.join(single_plots_folder, f"{feature}_cross_class_comparison_{name}.png")
    
    create_single_feature_comparison_plot(
        shap_values, X_test, model, duration_mapping, 
        feature, color_mapping='percentile', save_path=save_path, show_plot=False
    )

print(f"\n✅ All {len(all_features)} single feature plots generated!")



# Default using percentile mapping (Recommended)
result = create_single_feature_comparison_plot(
    shap_values, X_test, model, duration_mapping, 
    feature_name='your_feature', 
    color_mapping='percentile'  # 25%-75% percentile mapping
)

# Symmetric mapping (Suitable for normally distributed features)
result = create_single_feature_comparison_plot(
    shap_values, X_test_scaled, model, duration_mapping, 
    feature_name='your_feature', 
    color_mapping='symmetric'  # Symmetric mapping based on median
)

# Robust mapping (Suitable for features with outliers)
result = create_single_feature_comparison_plot(
    shap_values, X_test_scaled, model, duration_mapping, 
    feature_name='your_feature', 
    color_mapping='robust'  # Robust mapping based on IQR
)

# Standard mapping (Original method)
result = create_single_feature_comparison_plot(
    shap_values, X_test_scaled, model, duration_mapping, 
    feature_name='your_feature', 
    color_mapping='standard'  # Min-Max mapping
)





# === Usage Example ===

# Specify 6 features to plot
target_features = ['phylo_depth', 'TRW_cv', 'Elevation', 'PRE_cv', 'VPD_mean', 'VPD_cv']

# Create multi-plot
save_path = os.path.join(multi_plots_folder, f"selected_6_features_comparison_{name}.png")

print("=== Starting generation of 6-feature multi-plot ===\n")

feature_title_mapping = {
    'phylo_depth': 'Phylogenetic depth',
    'TRW_cv': 'TRW variation',
    'Elevation': 'Elevation',
    'VPD_mean': 'Average VPD',
    'VPD_cv': 'VPD variation',
    'PRE_cv': 'PRE variation'
}

# Call plotting function
create_multi_feature_comparison_plot(
    shap_values, X_test, model, duration_mapping, 
    target_features, color_mapping='percentile', 
    text_b=20, text_c=26,
    fig_width_cm = 55, 
    fig_height_cm = 50, 
    save_path=save_path, show_plot=True,
    feature_title_mapping=feature_title_mapping  # <- Added line
)

print(f"\n✅ 6-feature multi-plot generated!")











import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm, LinearSegmentedColormap
import matplotlib.cm as cm
import os
from scipy.stats import pearsonr  # [New] For calculating correlation coefficient

# ==========================================
# 0. Global Settings and Color Definitions
# ==========================================
feature_groups = {
    "Phylogenetic": ["phylo_depth"],
    "Spatial geographic": ["Elevation"],
    "Moisture-related": ["PRE_mean", "PRE_cv", "VPD_mean", "VPD_cv"],
    "Energy-related": ["TMP_mean", "TMP_cv", "SR_mean", "SR_cv", "CO2_mean", "CO2_cv"],
    "Vegetation state": ["TRW_mean", "TRW_cv"]
}

group_colors = {
  "Phylogenetic": "#f8b3d2",        
  "Spatial geographic": "#c9a3db",  
  "Moisture-related": "#5ab2fb",    
  "Energy-related": "#ffd08e",      
  "Vegetation state": "#90b768",
}


# Feature name beautification mapping (includes CO2 subscript)
feature_title_mapping = {
    'phylo_depth': 'Phylogenetic depth',
    'Elevation': 'Elevation',
    
    'TRW_cv': 'TRW variation',
    'TRW_mean': 'Average TRW',
    
    'VPD_cv': 'VPD variation',
    'VPD_mean': 'Average VPD',
    
    'TMP_cv': 'TMP variation',
    'TMP_mean': 'Average TMP',
    
    'SR_cv': 'SR variation',
    'SR_mean': 'Average SR',
    
    'CO2_cv': 'CO$_2$ Variation',
    'CO2_mean': 'Average CO$_2$',
    
    'PRE_mean': 'Average PRE',
    'PRE_cv': 'PRE variation'
}

# Global font settings
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 20,            
    'axes.titlesize': 24,       
    'axes.labelsize': 22,       
    'xtick.labelsize': 18,      
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

# ==========================================
# 1. Data Preparation
# ==========================================
if len(shap_values.shape) == 3:
    mean_shap = np.mean(np.mean(np.abs(shap_values), axis=2), axis=0)
else:
    mean_shap = np.mean(np.abs(shap_values), axis=0)

df_imp = pd.DataFrame({'Feature': X.columns, 'Importance': mean_shap})
feat_to_group = {f: g for g, feats in feature_groups.items() for f in feats}
df_imp['Category'] = df_imp['Feature'].map(feat_to_group).fillna('Other')
df_imp['Feature_Label'] = df_imp['Feature'].map(feature_title_mapping).fillna(df_imp['Feature'])
df_imp = df_imp.sort_values(by='Importance', ascending=True)

target_feature_list = []
phylo_feat = "phylo_depth"
target_feature_list.append(phylo_feat)
sorted_features_desc = df_imp.sort_values(by='Importance', ascending=False)['Feature'].tolist()
count = 0
for f in sorted_features_desc:
    if f != phylo_feat:
        target_feature_list.append(f)
        count += 1
    if count >= 4:
        break
print(f"Selected 5 features for ridgeline plot: {target_feature_list}")


# ==========================================
# 2. Plotting Function Definition
# ==========================================

def draw_bar_chart(ax, df_data, subplot_label='(a)'):
    """Draw feature importance bar chart (Plot a)"""
    bars = ax.barh(df_data['Feature_Label'], df_data['Importance'])
    
    for bar, original_feature in zip(bars, df_data['Feature']):
        category = df_data.loc[df_data['Feature'] == original_feature, 'Category'].values[0]
        bar.set_color(group_colors.get(category, '#95a5a6'))

    handles = [plt.Rectangle((0,0),1,1, color=color) for color in group_colors.values()]
    labels = group_colors.keys()
    ax.legend(handles, labels, title="Driver Groups", loc="lower right", fontsize=16) 

    ax.set_xlabel("Average Impact (Mean |SHAP value|)", fontweight='bold', fontsize=22)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18)

    ax.text(-0.25, 1.05, subplot_label, transform=ax.transAxes,
            fontsize=32, fontweight='bold', va='bottom', ha='left')
    
    if len(df_data) >= 4:
        line_y = len(df_data) - 4 - 0.5
        ax.axhline(y=line_y, color='black', linestyle='--', linewidth=1.5, alpha=0.8)

    ax.grid(axis='x', linestyle='--', alpha=0.5)


def create_single_subplot_comparison(ax, shap_values, X_test_scaled, model, 
                                   duration_mapping, feature_name, 
                                   text_a = 24, text_b = 20, text_c = 24,
                                   color_mapping='percentile', subplot_label='',
                                   feature_title_mapping=None):
    
    if feature_name not in X_test_scaled.columns:
        return None, None, None, None
    
    feature_idx = list(X_test_scaled.columns).index(feature_name)
    classes = sorted(model.classes_)
    n_classes = len(classes)
    
    shap_data = []
    feature_values = X_test_scaled[feature_name].values
    
    # [New] Dictionary to store stats per class (rank, r, p)
    class_stats = {}

    for class_idx, class_val in enumerate(classes):
        feature_shap_values = shap_values[:, feature_idx, class_idx]
        mean_shap = np.mean(feature_shap_values)
        class_feature_importance = np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)
        feature_rank = np.where(np.argsort(class_feature_importance)[::-1] == feature_idx)[0][0] + 1
        
        # [New] Calculate correlation (Feature Value vs SHAP Value)
        r, p = pearsonr(feature_values, feature_shap_values)
        
        # Store stats
        class_stats[class_val] = {
            'rank': feature_rank,
            'r': r,
            'p': p
        }

        for i, (shap_val, feature_val) in enumerate(zip(feature_shap_values, feature_values)):
            shap_data.append({
                'duration_class': class_val,
                'duration_label': duration_mapping[class_val],
                'shap_value': shap_val,
                'feature_value': feature_val,
                'mean_shap': mean_shap,
                'sample_idx': i
            })
    
    df_plot = pd.DataFrame(shap_data)
    feature_vals = df_plot['feature_value'].values
    
    if color_mapping == 'percentile':
        vmin = np.percentile(feature_vals, 25)
        vmax = np.percentile(feature_vals, 75)
    else:
        vmin, vmax = feature_vals.min(), feature_vals.max()
        
    if abs(vmax - vmin) < 1e-6:
        center = (vmin + vmax) / 2
        vmin = center - 0.1
        vmax = center + 0.1
    
    # [Modified] Build Y-axis labels including r and p
    y_positions = np.arange(n_classes)
    y_labels = []
    for cls in classes:
        stats = class_stats[cls]
        # Format P value
        if stats['p'] < 0.001:
            p_str = "p<0.001"
        else:
            p_str = f"p={stats['p']:.3f}"
        
        # Build label string: "Class\n#Rank | r=0.xx p=..."
        label_str = f"{duration_mapping[cls]}\n#{stats['rank']} | r={stats['r']:.2f} {p_str}"
        y_labels.append(label_str)

    
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    
    # Low - White - High gradient
    low_hex = "#67B0A2" 
    high_hex = "#CD555A"
    custom_gradient_cmap = LinearSegmentedColormap.from_list(
        "custom_teal_white_pink", 
        [low_hex, "#ffffff", high_hex]
    )

    for i, class_val in enumerate(classes):
        class_data = df_plot[df_plot['duration_class'] == class_val]
        y_pos = np.full(len(class_data), i) + np.random.normal(0, 0.08, len(class_data))
        
        scatter = ax.scatter(class_data['shap_value'], y_pos, 
                           c=class_data['feature_value'], 
                           cmap=custom_gradient_cmap, 
                           norm=norm, 
                           alpha=0.6, 
                           s=20, 
                           edgecolors=None, 
                           linewidth=0.3)
        
        mean_shap = class_data['mean_shap'].iloc[0]
        ax.axvline(x=mean_shap, ymin=(i-0.35)/n_classes, ymax=(i+0.35)/n_classes, 
                  color='black', linewidth=3, alpha=0.9)
    
    # Apply new labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=text_b, family='Arial')
    ax.set_xlabel('SHAP Value', fontsize=text_b, fontweight='bold', family='Arial')
    ax.tick_params(axis='x', labelsize=text_b)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)
    
    display_name = feature_title_mapping.get(feature_name, feature_name) if feature_title_mapping else feature_name
    ax.set_title(display_name, fontsize=text_c, fontweight='bold', pad=12, family='Arial')
    
    ax.grid(True, alpha=0.3, axis='x')
    
    for i in range(n_classes):
        if i % 2 == 0:
            ax.add_patch(Rectangle((ax.get_xlim()[0], i-0.4), 
                                   ax.get_xlim()[1] - ax.get_xlim()[0], 0.8, 
                                   facecolor='lightgray', alpha=0.1, zorder=0))
    
    if subplot_label:
        ax.annotate(subplot_label, xy=(0, 1), xytext=(-35, 20), 
                    textcoords='offset points', xycoords='axes fraction',
                    fontsize=32, fontweight='bold', ha='right', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='none'))
    
    # --- Custom solid-color extended colorbar logic ---
    feature_vals_full = X_test_scaled[feature_name].values
    data_min, data_max = feature_vals_full.min(), feature_vals_full.max()
    
    low_color = custom_gradient_cmap(0.0) 
    high_color = custom_gradient_cmap(1.0)
    
    n_segments = 100
    colors = []
    
    n_low = int(n_segments * (vmin - data_min) / (data_max - data_min))
    colors.extend([low_color] * max(1, n_low))
    
    n_mid = int(n_segments * (vmax - vmin) / (data_max - data_min))
    mid_colors = [custom_gradient_cmap(x) for x in np.linspace(0, 1, max(1, n_mid))]
    colors.extend(mid_colors)
    
    n_high = n_segments - len(colors)
    colors.extend([high_color] * max(1, n_high))
    
    custom_cmap_extended = ListedColormap(colors)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    sm = cm.ScalarMappable(norm=Normalize(vmin=data_min, vmax=data_max), cmap=custom_cmap_extended)
    cbar = plt.colorbar(sm, cax=cax)
    
    cbar.ax.tick_params(labelsize=22) 
    
    ticks = [data_min, vmin, np.median(feature_vals_full), vmax, data_max]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{x:.2f}' for x in ticks])
    
    if color_mapping == 'percentile':
        cbar.ax.axhline(y=vmin, color='white', linewidth=2, alpha=0.9)
        cbar.ax.axhline(y=vmax, color='white', linewidth=2, alpha=0.9)

    return scatter, norm, vmin, vmax

# ==========================================
# 3. Generate 2x3 Layout (Adjusted)
# ==========================================

def create_final_combined_figure_2x3_compact():
    fig, axes = plt.subplots(2, 3, figsize=(26, 17))
    axes = axes.flatten()
    
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    print("=== Starting drawing final combined plot (With Correlation Stats) ===")
    
    print("Drawing Plot (a)")
    draw_bar_chart(axes[0], df_imp, subplot_label=labels[0])
    
    font_params = {'text_b': 20, 'text_c': 24} 

    for i, feature in enumerate(target_feature_list):
        ax_idx = i + 1
        label = labels[ax_idx]
        print(f"Drawing Plot {label}: {feature}")
        
        create_single_subplot_comparison(
            ax=axes[ax_idx], 
            shap_values=shap_values, 
            X_test_scaled=X_test, 
            model=model, 
            duration_mapping=duration_mapping, 
            feature_name=feature, 
            color_mapping='percentile', 
            subplot_label=label,
            feature_title_mapping=feature_title_mapping,
            **font_params
        )
        
    plt.tight_layout(pad=2.0, w_pad=3.0, h_pad=3.0)
    
    save_folder = os.path.join(output_folder, "Final_Combined_Plots")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    save_png = os.path.join(save_folder, f"Final_Combined_SHAP_2x3_Stats_{name}.png")
    save_pdf = os.path.join(save_folder, f"Final_Combined_SHAP_2x3_Stats_{name}.pdf")
    
    plt.savefig(save_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_pdf, dpi=300, bbox_inches='tight')
    print(f"✅ Saved stats version PNG: {save_png}")
    plt.show()

# Execute
create_final_combined_figure_2x3_compact()




























# =============================================================================
# Correlation
# =============================================================================

import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. Data Preparation
# ==========================================
# Assuming X is your feature data (DataFrame), y is Duration (Series, numeric: 2, 3, 4, 5...)
# Note: y must be numeric (e.g., 3.13) or ordered integer (2, 3, 4, 5), not classification strings

# For demonstration, merge X and y into one DataFrame
df_analysis = X.copy()
df_analysis['Duration'] = y  # Ensure y name is 'Duration'

# Define list of all features to analyze
target_col = 'Duration'
predictors = X.columns.tolist()

# Store results
partial_corr_results = []

print("Calculating partial correlation analysis (controlling for all other variables)...")

# ==========================================
# 2. Loop to calculate partial correlations
# ==========================================
for var in predictors:
    # Current variable as x
    # Target variable as y
    # All other variables as covar (control variables)
    covariates = [p for p in predictors if p != var]
    
    try:
        # Calculate partial correlation
        stats = pg.partial_corr(
            data=df_analysis, 
            x=var, 
            y=target_col, 
            covar=covariates,
            method='pearson' # or 'spearman' if data is not normally distributed
        )
        
        # Extract results
        r_val = stats['r'].values[0]       # Correlation coefficient
        p_val = stats['p-val'].values[0]   # P-value
        ci_low = stats['CI95%'].values[0][0] # 95% CI lower bound
        ci_high = stats['CI95%'].values[0][1] # 95% CI upper bound
        
        partial_corr_results.append({
            'Feature': var,
            'Partial_Corr': r_val,
            'P_Value': p_val,
            'Significance': '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
        })
        
    except Exception as e:
        print(f"Calculation failed for variable {var}: {e}")

# Convert to DataFrame and sort
df_pcorr = pd.DataFrame(partial_corr_results)
df_pcorr = df_pcorr.sort_values(by='Partial_Corr', ascending=False) # Sort by correlation descending

print("Calculation complete! Top 5 results:")
print(df_pcorr.head())

# Save results
# df_pcorr.to_csv("Partial_Correlation_Results.csv", index=False)

# ==========================================
# 3. Visualization: Beautiful Lollipop Chart
# ==========================================

plt.figure(figsize=(10, 10))

# Create color map: Positive red, Negative blue
# Use normalization to map -1~1 to colors
norm = plt.Normalize(-1, 1) # Or adjust based on your data range, e.g., df_pcorr['Partial_Corr'].min() to max()
colors = plt.cm.coolwarm(norm(df_pcorr['Partial_Corr'].values))

# Plot lines
plt.hlines(y=df_pcorr['Feature'], xmin=0, xmax=df_pcorr['Partial_Corr'], 
           color=colors, alpha=0.6, linewidth=2)

# Plot points
for i, (r, p) in enumerate(zip(df_pcorr['Partial_Corr'], df_pcorr['P_Value'])):
    # Adjust point size or border based on significance
    edge_color = 'black' if p < 0.05 else 'none' # Add black border if significant
    plt.scatter(r, i, color=colors[i], s=100, edgecolors=edge_color, zorder=3)

# Add center line
plt.axvline(0, color='gray', linestyle='--', linewidth=1)

# Add significance markers
for i, (r, sig) in enumerate(zip(df_pcorr['Partial_Corr'], df_pcorr['Significance'])):
    if sig != 'ns':
        # Adjust asterisk position based on positive/negative correlation
        offset = 0.01 if r > 0 else -0.02
        ha = 'left' if r > 0 else 'right'
        plt.text(r + offset, i, sig, va='center', ha=ha, fontsize=12, fontweight='bold')

plt.xlabel('Partial Correlation Coefficient (r)', fontsize=12)
plt.title('Partial Correlation with Duration\n(Controlling for all other variables)', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

# Save image
plt.savefig(os.path.join(shap_viz_folder, "Partial_Correlation_Plot.png"), dpi=300)
plt.show()





# =============================================================================
# Structural Equation Model (SEM)
# =============================================================================

import pandas as pd
import numpy as np
from semopy import Model, calc_stats
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# 0. Preparation
# ==========================================
print("\n" + "="*60)
print("=== Step 3: SEM Model Based on New Hand-drawn DAG ===")
print("="*60)

# 1. Variable Mapping
vars_map = {
    'Dur': 'duration',       # Outcome
    'Phy': 'phylo_depth',    # Mediator (Middle)
    'Ele': 'Elevation',      # Exogenous Source
    'Vc':  'VPD_cv',         # Mediator (Top)
    'Vm': 'VPD_mean',
    'Gs':  'Num_GS_Months',  # Mediator (Bottom 1)
    'Trw': 'TRW_cv'          # Mediator (Bottom 2)
}
cols_sem = list(vars_map.values())

# Check Data
missing_cols = [c for c in cols_sem if c not in df.columns]
if missing_cols:
    print(f"❌ Error: Data missing columns: {missing_cols}")
else:
    # Extract and drop NA
    df_sem = df[cols_sem].dropna()
    
    # 2. Data Standardization
    # Must do, to compare coefficient sizes of different paths
    scaler = StandardScaler()
    df_sem_scaled = pd.DataFrame(scaler.fit_transform(df_sem), columns=df_sem.columns)
    
    # ==========================================
    # 3. Define Model (Model Syntax)
    # ==========================================
    model_desc = """
    # --- 1. Environmental Driver Layer (Source) ---
    VPD_cv ~ Elevation
    VPD_mean ~ Elevation
    
    # --- 2. Biological Filter Layer (Filter) ---
    phylo_depth ~ VPD_cv
    phylo_depth ~ VPD_mean
    
    # --- 3. Physiological Response Layer (Response) ---
    # [Key Modification 1]: TRW_cv is determined not only by environment but also by "ancestral genetics" (Phylo)
    #  Ancient species (Phylo) typically have more conservative or different growth strategies, directly affecting TRW_cv
    TRW_cv ~ VPD_cv + VPD_mean + phylo_depth
    
    # [Key Modification 2]: Growth Season (GS) depends not only on Elevation but also on environmental mood (VPD_cv)
    #  Large environmental fluctuations (VPD_cv) might limit growth season length
    
    # --- 4. Outcome Convergence Layer (Outcome) ---
    duration ~ VPD_cv + VPD_mean + TRW_cv + phylo_depth
    
    # --- 5. Residual Covariance (Fine-tuning) ---
    # Allow unexplained covariance between two physiological indicators
    VPD_cv ~~ VPD_mean
    """
    
    print("Fitting SEM model...")
    
    try:
        # ==========================================
        # 4. Model Fitting
        # ==========================================
        model = Model(model_desc)
        res = model.fit(df_sem_scaled)
        
        # ==========================================
        # 5. Get Complete Parameter Table
        # ==========================================
        inspect = model.inspect(std_est=True)
        
        print("\n" + "-"*30)
        print("【Table 1: Complete Parameter Estimates (Parameter Estimates)】")
        print("-" * 30)
        print(inspect.to_string())
        
        # Save parameters
        out_path_params = os.path.join(output_folder, f"SEM_NewDesign_Params_{name}.csv")
        inspect.to_csv(out_path_params)

        # ==========================================
        # 6. Get Goodness of Fit
        # ==========================================
        stats = calc_stats(model)
        
        print("\n" + "-"*30)
        print("【Table 2: Model Goodness of Fit (Fit Indices)】")
        print("-" * 30)
        print(stats.T)
        
        # Save fit metrics
        out_path_stats = os.path.join(output_folder, f"SEM_NewDesign_Fit_{name}.csv")
        stats.to_csv(out_path_stats)

        # ==========================================
        # 7. Manually Extract R2
        # ==========================================
        print("\n" + "-"*30)
        print("【Table 3: R-Square for Endogenous Variables】")
        print("-" * 30)
        
        variances = inspect[(inspect['op'] == '~~') & (inspect['lval'] == inspect['rval'])]
        endogenous_vars = ['duration', 'TRW_cv', 'Num_GS_Months', 'phylo_depth', 'VPD_cv']
        
        for var in endogenous_vars:
            try:
                error_var = variances[variances['lval'] == var]['Estimate'].values[0]
                r2 = 1 - error_var
                print(f"   - {var:<15}: R2 = {r2:.4f} ({r2*100:.1f}%)")
            except IndexError:
                pass
        
        # ==========================================
        # 8. Key Path Verification (Key Insight Check)
        # ==========================================
        print("\n" + "="*40)
        print("=== Key Conclusion Verification (Key Path Check) ===")
        print("="*40)
        
        def get_beta(target, source):
            try:
                row = inspect[(inspect['lval']==target) & (inspect['rval']==source) & (inspect['op']=='~')]
                return row['Estimate'].values[0], row['p-value'].values[0]
            except:
                return 0, 1

        # 1. Compare Direct Drivers (Who drives Duration?)
        b_trw, p_trw = get_beta('duration', 'TRW_cv')
        b_vc, p_vc  = get_beta('duration', 'VPD_cv')
        b_phy, p_phy = get_beta('duration', 'phylo_depth')
        b_ele, p_ele = get_beta('duration', 'Elevation')
        b_vm, p_vm = get_beta('duration', 'VPD_mean')
        
        print(f"1. Direct Impact on Duration:")
        print(f"   - TRW_cv (Growth):  {b_trw:.3f} (p={p_trw:.3f})")
        print(f"   - VPD_cv (Risk):    {b_vc:.3f} (p={p_vc:.3f})")
        print(f"   - VPD_mean:         {b_vm:.3f} (p={p_vm:.3f})")
        print(f"   - Phylo  (History): {b_phy:.3f} (p={p_phy:.3f})")
        print(f"   - Elevation:        {b_ele:.3f} (p={p_ele:.3f})")

        
        if abs(b_trw) > abs(b_vc) and abs(b_trw) > abs(b_phy):
             print("   ✅ TRW_cv is the strongest direct driver.")

        # 2. Verify Environmental Filtering on Phylogeny (VPD_cv -> Phylo)
        b_vc_phy, p_vc_phy = get_beta('phylo_depth', 'VPD_cv')
        print(f"\n2. Environmental Variation Filters Phylogeny (VPD_cv -> Phylo):")
        print(f"   - Beta: {b_vc_phy:.3f} (p={p_vc_phy:.3f})")
        if p_vc_phy < 0.05:
            print("   ✅ Verification Significant: Environmental fluctuations affect phylogenetic distribution.")

    except Exception as e:
        print(f"❌ SEM fitting failed: {e}")
        import traceback
        traceback.print_exc()

print("="*60 + "\n")