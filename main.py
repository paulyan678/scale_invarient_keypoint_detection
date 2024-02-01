import cv2
import numpy as np
import scipy.ndimage as ndi 
import matplotlib.pyplot as plt

def open_image(path):
    img = cv2.imread(path)
    if img is None:
        raise Exception("Image not found")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def get_DOG(img, sigma, k, s):
    pre_blur = cv2.GaussianBlur(img.astype(np.float64), (0, 0), sigma)
    DOG = np.zeros((img.shape[0], img.shape[1], s), dtype=np.float64)
    for i in range(1, s + 1):
        cur_blur = cv2.GaussianBlur(img.astype(np.float64), (0, 0), k**i * sigma)
        DOG[:, :, i-1] = np.abs(cur_blur - pre_blur)
        pre_blur = cur_blur
    return DOG

def plot_all_DOG(DOG):
    for i in range(DOG.shape[2]):
        cv2.imshow("DOG", DOG[:, :, i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
def plot_res(filter, radius, rows, cols):

    fig, ax = plt.subplots()
    ax.imshow(filter, cmap='gray')
    for row, col, r in zip(rows, cols, radius):
        circle = plt.Circle((col, row), r, fill=False, color='red')
        ax.add_patch(circle)
    plt.show()


def calc_radius(sigma, k):
    return (sigma*k*np.sqrt(2*np.log(k**2))) / (np.sqrt(k**2 - 1))

def local_maxima_3D(data, order=5):
    """Detects local maxima in a 3D array
    
    Parameters
    ---------
    data : 3d ndarray
    order : int
        How many points on each side to use for the comparison
    
    Returns
    -------
    coords : ndarray
        coordinates of the local maxima
    values : ndarray
        values of the local maxima
    """
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0
    
    filtered = ndi.maximum_filter(data, footprint=footprint)
    mask_local_maxima = data > filtered
    coords = np.asarray(np.where(mask_local_maxima)).T
    values = data[mask_local_maxima]
    
    return coords, values

gray = open_image("/Users/paulyan/Documents/csc_420/scale_invarient_keypoint_detection/butterfly.jpg")
sigma = 1.6
s = 20
k = 1.1
dog = get_DOG(gray, sigma, k, s)
# i, j, q = np.unravel_index(np.argsort(dog.flatten())[-10:], dog.shape)
# print(i, j, q)
coords, _ = local_maxima_3D(dog)
rows = coords[:, 0]
cols = coords[:, 1]
q = coords[:, 2]
radius = calc_radius(k**(q-1)*sigma, k)
# print(coords)
plot_res(gray, radius, rows, cols)


gray = gray[0:200, 0:200]
dog = get_DOG(gray, sigma, k, s)
# i, j, q = np.unravel_index(np.argsort(dog.flatten())[-10:], dog.shape)
# print(i, j, q)
coords, _ = local_maxima_3D(dog)
rows = coords[:, 0]
cols = coords[:, 1]
q = coords[:, 2]
radius = calc_radius(k**(q-1)*sigma, k)
# print(coords)
plot_res(gray, radius, rows, cols)
