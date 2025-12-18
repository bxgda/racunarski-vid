# AUTOMATSKO OTKLANJANJE PERIODIČNOG ŠUMA PRIMENOM FURIJEOVE TRANSFORMACIJE

# kod radi tako što učita sliku i automatski detektuje piksele na magnitudi spektra koji prave nepotrebne šumove

import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft(image):
    return np.fft.fftshift(np.fft.fft2(image))

def inverse_fft(magnitude_log, phase):
    magnitude = np.exp(magnitude_log)
    fft_shifted = magnitude * phase
    fft_unshifted = np.fft.ifftshift(fft_shifted)
    return np.abs(np.fft.ifft2(fft_unshifted))

def show_plot(image, title, gray=False):
    if gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def load_image(path, color_system):
    img = cv2.imread(path)
    return cv2.cvtColor(img, color_system)

def detect_periodic_peaks(image_log, ignore_radius=10, threshold_factor=5):
    """
    automatska detekcija periodicnog suma.
    ignore_radius: broj piksela oko centra koji se ne dira
    """
    h, w = image_log.shape
    cy, cx = h // 2, w // 2

    # maskiraj centar
    mask = np.ones_like(image_log, dtype=bool)
    mask[cy - ignore_radius:cy + ignore_radius + 1, cx - ignore_radius:cx + ignore_radius + 1] = False

    # koristi medijanu i std za threshold
    masked_values = image_log[mask]
    median_val = np.median(masked_values)
    std_val = np.std(masked_values)

    # automatski prag
    threshold = median_val + (np.max(masked_values) - median_val) / std_val

    # pikovi iznad praga i van centra
    peaks = (image_log > threshold) & mask
    return peaks

def process_image(image):
    fft_img = fft(image)
    mag = np.abs(fft_img)
    phase = fft_img / (mag + 1e-8)
    mag_log = np.log(mag + 1e-8)

    show_plot(mag_log, "Log-magnituda spektra", True)

    # automatski detektuj pikove
    peaks = detect_periodic_peaks(mag_log, ignore_radius=20)
    mag_log[peaks] = 0

    show_plot(mag_log, "Magnituda posle uklanjanja pikova", True)

    return inverse_fft(mag_log, phase)

if __name__ == "__main__":
    image = load_image("./slika_4.png", cv2.COLOR_BGR2GRAY)
    show_plot(image, "Originalna slika", True)

    result = process_image(image)
    show_plot(result, "Slika bez periodicnog suma", True)
    cv2.imwrite("./slika_0_bez_periodicnog_suma.png", result)
