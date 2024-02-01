import numpy as np
import matplotlib.pyplot as plt


def generate_gaussian_filter(n, sigma):
    x = np.arange(-n//2 + 1, n//2 + 1.)
    y = np.arange(-n//2 + 1, n//2 + 1.)

    X, Y = np.meshgrid(x, y)
    return np.exp(-(X**2 + Y**2)/(2*sigma**2))/(2*np.pi*sigma**2)

def plot(filter, circle):
    fig, ax = plt.subplots()
    ax.imshow(filter, cmap='gray')
    ax.add_patch(circle)
    plt.show()

def calc_radius(sigma, k):
    return (sigma*k*np.sqrt(2*np.log(k**2))) / (np.sqrt(k**2 - 1))




k = 3
sigma = 10
n = 100
narrow = generate_gaussian_filter(n, sigma)
wide = generate_gaussian_filter(n, k*sigma)
radius = calc_radius(sigma, k)
circle = plt.Circle((n//2, n//2), radius, fill=False, color='red')
plot(wide - narrow, circle)
