# OTKLANJANJE PERIODIČNOG ŠUMA PRIMENOM FURIJEOVE TRANSFORMACIJE

# kod radi tako što se na slici magnitude spektra nađu amplitude (pikseli) koje prave nepotrebne frekvencije
# i njihovim "nuliranjem" brišemo nepotrebne frekvencije odnosno otklanjamo šum sa slike

import cv2
import numpy as np
import matplotlib.pyplot as plt

slika = cv2.imread("slika_4.png", 0)
cv2.imshow("slika_4.png", slika)
cv2.waitKey(0)

furijeova_transformacija = np.fft.fft2(slika)
ft_pomeraj = np.fft.fftshift(furijeova_transformacija)

magnituda_spektra = np.log(np.abs(ft_pomeraj) + 1)

plt.imshow(magnituda_spektra)
plt.title('magnituda spektra sa sumom')
plt.savefig('ft_magnituda_spektra_PRE.png')
plt.show()

koordinate = [(306,306), (206, 306), (306,206), (206,206)]
for xy in koordinate:
    ft_pomeraj[xy] = 0

magnituda_spektra = np.log(np.abs(ft_pomeraj) + 1)

plt.imshow(magnituda_spektra)
plt.title('magnituda spektra bez suma')
plt.savefig('ft_magnituda_spektra_POSLE.png')
plt.show()

ft_pomeraj_nazad = np.fft.ifftshift(ft_pomeraj)
image_finished = np.fft.ifft2(ft_pomeraj_nazad).real

cv2.imshow('obradjena slika', image_finished.astype(np.uint8))
cv2.imwrite('slika_4_bez_suma.png', image_finished)
cv2.waitKey(0)
cv2.destroyAllWindows()
