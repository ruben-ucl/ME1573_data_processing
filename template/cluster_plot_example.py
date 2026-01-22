import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from matplotlib import cm

# Set random seed for reproducibility
np.random.seed(42)

# Create three different datasets
# Dataset 1: Two clear clusters along a correlation line
cluster1_x = np.random.normal(5, 0.5, 100)
cluster1_y = 2 * cluster1_x + np.random.normal(0, 1, 100)
cluster2_x = np.random.normal(15, 0.5, 100)
cluster2_y = 2 * cluster2_x + np.random.normal(0, 1, 100)
data_2clusters = np.column_stack([
    np.concatenate([cluster1_x, cluster2_x]),
    np.concatenate([cluster1_y, cluster2_y])
])

# Dataset 2: Three clear clusters along a correlation line
cluster1_x = np.random.normal(5, 0.5, 100)
cluster1_y = 2 * cluster1_x + np.random.normal(0, 1, 100)
cluster2_x = np.random.normal(10, 0.5, 100)
cluster2_y = 2 * cluster2_x + np.random.normal(0, 1, 100)
cluster3_x = np.random.normal(15, 0.5, 100)
cluster3_y = 2 * cluster3_x + np.random.normal(0, 1, 100)
data_3clusters = np.column_stack([
    np.concatenate([cluster1_x, cluster2_x, cluster3_x]),
    np.concatenate([cluster1_y, cluster2_y, cluster3_y])
])

# Dataset 3: Uniform distribution (no real clusters)
x_uniform = np.random.uniform(5, 15, 200)
y_uniform = 2 * x_uniform + np.random.normal(0, 2, 200)
data_uniform = np.column_stack([x_uniform, y_uniform])

def plot_silhouette_analysis(data, dataset_name, k_values=[2, 3, 4]):
    """Create comprehensive silhouette analysis."""
    n_rows = 2
    n_cols = len(k_values)
    fig = plt.figure(figsize=(5 * n_cols, 10))
    
    for idx, k in enumerate(k_values):
        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        
        # Calculate silhouette scores
        silhouette_avg = silhouette_score(data, cluster_labels)
        sample_silhouette_values = silhouette_samples(data, cluster_labels)
        
        # ===== SILHOUETTE PLOT (top row) =====
        ax1 = plt.subplot(n_rows, n_cols, idx + 1)
        ax1.set_xlim([-0.2, 1])
        ax1.set_ylim([0, len(data) + (k + 1) * 10])
        
        y_lower = 10
        for i in range(k):
            # Aggregate silhouette scores for samples in cluster i
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / k)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            
            # Label the silhouette plots with cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), 
                    fontsize=12, fontweight='bold')
            
            # Compute new y_lower for next plot
            y_lower = y_upper + 10
        
        ax1.set_title(f'Silhouette Plot (k={k})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Silhouette Coefficient Value', fontsize=12)
        ax1.set_ylabel('Cluster Label', fontsize=12)
        
        # The vertical line for average silhouette score
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2)
        ax1.text(silhouette_avg + 0.05, len(data) * 0.9, 
                f'Avg: {silhouette_avg:.3f}',
                color='red', fontsize=11, fontweight='bold')
        
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # ===== SCATTER PLOT (bottom row) =====
        ax2 = plt.subplot(n_rows, n_cols, n_cols + idx + 1)
        colors = cm.nipy_spectral(cluster_labels.astype(float) / k)
        ax2.scatter(data[:, 0], data[:, 1], marker='o', s=30, 
                   lw=0, alpha=0.7, c=colors, edgecolor='k')
        
        # Mark cluster centres
        centers = kmeans.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                   c="white", alpha=1, s=200, edgecolor='k', linewidth=2)
        
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker=f'${i}$', alpha=1,
                       s=50, edgecolor='k', linewidth=1)
        
        ax2.set_title(f'Clustered Data (k={k})', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Signal 1', fontsize=12)
        ax2.set_ylabel('Signal 2', fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{dataset_name}\nSilhouette Analysis for Different Numbers of Clusters',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig

# Generate plots for all three datasets
fig1 = plot_silhouette_analysis(data_2clusters, 'Dataset 1: Two Clusters', k_values=[2, 3, 4])
fig2 = plot_silhouette_analysis(data_3clusters, 'Dataset 2: Three Clusters', k_values=[2, 3, 4])
fig3 = plot_silhouette_analysis(data_uniform, 'Dataset 3: Uniform (No Clusters)', k_values=[2, 3, 4])

plt.show()

# Print summary statistics
print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

for data, name in [(data_2clusters, "Two Clusters"),
                    (data_3clusters, "Three Clusters"),
                    (data_uniform, "Uniform")]:
    print(f"\n{name}:")
    print("-" * 40)
    
    for k in [2, 3, 4]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        
        print(f"  k={k}: Silhouette Score = {score:.3f}", end="")
        
        # Indicate optimal k
        if k == 2 and name == "Two Clusters":
            print(" ← OPTIMAL")
        elif k == 3 and name == "Three Clusters":
            print(" ← OPTIMAL")
        elif k == 2 and name == "Uniform":
            print(" (best of poor options)")
        else:
            print()

print("\n" + "=" * 60)