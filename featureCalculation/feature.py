from skimage.morphology import skeletonize
import numpy as np
from scipy.spatial.distance import euclidean
def calculateVesselMetrics(prediction_np, pixel_spacing_mm):
    total_pixels = prediction_np.size
    vessel_pixels = np.sum(prediction_np)
    density = vessel_pixels / total_pixels
    skeleton = skeletonize(prediction_np)
    vessel_length_mm = np.sum(skeleton) * pixel_spacing_mm
    return density, vessel_length_mm

def calculateTortuosity(skeleton):
    y, x = np.where(skeleton == 1)
    coordinates = list(zip(y, x))
    if len(coordinates) < 2:
        return 0
    L = 0
    for i in range(1, len(coordinates)):
        L += euclidean(coordinates[i - 1], coordinates[i])
    D = euclidean(coordinates[0], coordinates[-1])
    T = L / D if D != 0 else 0
    return T