"""
K-means clustering and Silhouette analysis in the latent space to identify metastable pathway channels

"""

import glob
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial' 
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

resultdir = 'results/train_nsamples10001_batchsize250_lr8e-05_c1'
X = np.load(resultdir + '/trained_hidden_vectors_nsamples10001_batchsize250_lr8e-05_c1.npy')
range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(16, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=12)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

  
    plt.savefig(resultdir + "/hidden_layer_" + str(n_clusters) + "kmeans_centers_silhouette_analysis.png")



### plot the train/test scores and clustered pathways weighted by flux in the latent space ###

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial' 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import cm

lasso = 'microcinJ25'
resultdir = 'results/train_nsamples10001_batchsize250_lr8e-05_c1'


train_score = np.load(resultdir + '/training_scores_save.npy',allow_pickle=True) 
test_score = np.load(resultdir + '/testing_scores_save.npy',allow_pickle=True)
train = [train_score.tolist()[i][0] for i in range(100)] 
test = [test_score.tolist()[i][0] for i in range(100)]

fig = plt.figure(dpi=300, figsize=(10, 7))  
ax = fig.add_subplot(111) 
X = list(range(100)) 
plt.plot(X, train, color='navy', linewidth = 3,label='training loss') 
plt.plot(X, test, color='orange', linewidth = 3,label='testing loss')

ax.spines['bottom'].set_linewidth(2.0)
ax.spines['left'].set_linewidth(2.0)
ax.spines['top'].set_linewidth(2.0)
ax.spines['right'].set_linewidth(2.0)

plt.xlabel('Training Epochs', fontsize = 22)
plt.ylabel('Loss', fontsize = 22)                                                           
plt.legend(loc="upper right")
plt.legend(fontsize="x-large")
plt.savefig(resultdir + '/train_test.png')
plt.close()


centers = np.load(resultdir + '/trained_hidden_vectors_nsamples10001_batchsize250_lr8e-05_c1.npy')
flux = np.load(lasso + '_TPT_pathways_flux.npy')

n_clusters = 2
n_samples = 10001
batch_size = 250
learning_rate = 8e-5

km_cluster = KMeans(n_clusters=n_clusters, max_iter=2000, random_state=42)
km_cluster.fit(centers)
print(km_cluster.labels_)
np.savetxt(resultdir + "/paths_" + str(n_clusters) + "kmeans_clustering_labels.txt".format(n_samples),
           km_cluster.labels_)

silhouette_avg = silhouette_score(centers, km_cluster.labels_)
print(silhouette_avg)
km_lump = np.zeros(n_clusters)
for i in range(n_samples):
    km_lump[int(km_cluster.labels_[i])] += flux[i]

km_lump = km_lump / np.sum(km_lump)
print(km_lump)


flux_normalized = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))  # Min-max normalization
flux_transformed = np.power(flux_normalized, 0.6)
min_alpha = 0.01  # Minimum visibility (15% opacity)
flux_alpha = min_alpha + (1 - min_alpha) * flux_transformed

min_size = 1  # Minimum point size
max_size = 100  # Maximum point size
sizes = min_size + (max_size - min_size) * flux_normalized

# Define the colors for each cluster (one color per cluster)
colors = [plt.cm.tab20c(1), plt.cm.Set1(7), plt.cm.tab20c(8)]

sorted_indices = np.argsort(flux_alpha)

# Create the figure and axis
fig1 = plt.figure(dpi=600, figsize=(7, 7))
ax = fig1.add_subplot(111)

# Plot each point, adjusting color based on flux value within the cluster
for idx in sorted_indices:  # Iterate through points in sorted order
    cluster_idx = int(km_cluster.labels_[idx])  # Get the cluster index for this point
    color_intensity = flux_alpha[idx]  # Use the normalized flux value for color intensity
    cluster_color = colors[cluster_idx]  # Get the color corresponding to the cluster
    point_size = sizes[idx]
    ax.scatter(
        centers[idx, 0],
        centers[idx, 1],
        color=cluster_color,
        alpha=color_intensity,
        s=point_size,
        edgecolors=cluster_color,
        linewidth=1.0
    )

# Customize the axes
ax.spines['bottom'].set_linewidth(2.0)
ax.spines['left'].set_linewidth(2.0)
ax.spines['top'].set_linewidth(2.0)
ax.spines['right'].set_linewidth(2.0)

# Labels
plt.xticks(fontsize=16) # Set x-axis tick label font size
plt.yticks(fontsize=16) # Set y-axis tick label font size
plt.xlabel('VAE Latent Dimension 1', fontsize=22)
plt.ylabel('VAE Latent Dimension 2', fontsize=22)

# Save the figure
plt.savefig("hidden_layer_" + str(n_clusters) + "kmeans_centers_flux_weighted.png", bbox_inches='tight')