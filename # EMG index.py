# EMG index


import numpy as np
from sklearn.decomposition import PCA

# FPCA on the EMG data

    # Load the data - named emg_data_matrix at the moment

    # Assuming emg_data_matrix is a 2D numpy array with rows representing time and columns representing muscle channels
    # Perform PCA separately on each muscle channel
harmonic_scores = np.zeros_like(emg_data_matrix)
for i in range(emg_data_matrix.shape[1]):  # Iterate over each muscle channel
    muscle_channel_data = emg_data_matrix[:, i]
    pca = PCA(n_components=3)  # Choose the number of components - in the article the first 3 describes more than 70%
    pca.fit(muscle_channel_data)
    harmonic_scores[:, i] = pca.transform(muscle_channel_data)

# Harmonic scores - the contribution of each functional component (harmonic) to the variability in the EMG data. 

# Other solution for FPCA
from fda import FDataGrid
from fda.fdarep import basis
from fda.fdarep import bases

# Assuming emg_data is a 2D numpy array with rows representing time and columns representing EMG signals
# Create a FDataGrid object
fd_grid = FDataGrid(data_matrix=emg_data.T, sample_points=time_vector)

# Choose a basis for functional PCA (e.g., B-spline basis)
bspline_basis = basis.BSpline(n_basis=10, domain_range=(time_min, time_max))
bspline_fd = bases.Basis(fd_grid.sample_points[0], bspline_basis)

# Perform functional PCA
fd_pca = fd_grid.pca(0.95, method="standard", basis=bspline_fd)

# Get the harmonic scores
harmonic_scores = fd_pca.harmonics_scores()


# Mean and std

mean = np.mean(harmonic_scores)
std = np.std(harmonic_scores)

# Function to calculate Euclidean distance
def euclidean_distance(harmonic_scores, mean, std):
    normalized_scores = (harmonic_scores - mean) / std
    distance = np.sqrt(np.sum(normalized_scores**2))
    return distance

# Function to calculate EMG index for a single subject
def calculate_emg_index(harmonic_scores, mean, std):
    distances = np.zeros(len(harmonic_scores))
    for i, scores in enumerate(harmonic_scores):
        distances[i] = euclidean_distance(scores, mean[i], std[i])
    return np.mean(distances)

# Sample data (replace with your actual data)
# Suppose you have 3 harmonic scores for each muscle-muscle plot for each subject
num_subjects_TD = 10  # number of subjects with typical development
num_subjects_CP = 8   # number of subjects with cerebral palsy
num_harmonics = 3     # number of harmonics considered
num_muscle_plots = 8  # number of muscle-muscle plots

# Generate random data for demonstration
harmonic_scores_TD = np.random.rand(num_subjects_TD, num_muscle_plots, num_harmonics)
harmonic_scores_CP = np.random.rand(num_subjects_CP, num_muscle_plots, num_harmonics)

# Calculate mean and standard deviation for TD group
mean_TD = np.mean(harmonic_scores_TD, axis=(0, 1))
std_TD = np.std(harmonic_scores_TD, axis=(0, 1))

# Normalize CP data using TD group
normalized_scores_CP = (harmonic_scores_CP - mean_TD) / std_TD

# Calculate EMG index for each subject in CP group
emg_indices = np.zeros(num_subjects_CP)
for i, subject_scores in enumerate(normalized_scores_CP):
    emg_indices[i] = calculate_emg_index(subject_scores, mean_TD, std_TD)

# Calculate overall index based on symmetry and co-activation
overall_index = np.sum(emg_indices)

print("Overall EMG index:", overall_index)