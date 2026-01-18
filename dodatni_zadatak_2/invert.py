import cv2
import os

# ovaj fajl sluzi da mogu da napravim sliku sa crnom pozadinom i belim oblikom jer sam mahom na netu nalazio slike sa
# belom pozadinom i crnim oblikom pa da bi model istrenirao da slikama kakve ce moci da se ocekuju na ulazu
# menjamo sve slike kako bi bile uniformne

# za slovo P promeniti putanju da bude dataset/P/ i ime_fajla 'p{i}.jpg'

n = 43
putanja_foldera = 'dataset/Ostalo/'

for i in range(0, n + 1):
    ime_fajla = f'o{i}.jpg'
    putanja = os.path.join(putanja_foldera, ime_fajla)

    # omdah ucitavamo u gray scale
    slika = cv2.imread(putanja, 0)

    if slika is not None:
        # odmah kako radimo binarizaciju radimo i inverziju
        _, thresh = cv2.threshold(slika, 127, 255, cv2.THRESH_BINARY_INV)

        # cuvamo sliku
        cv2.imwrite(putanja, thresh)
        print(f"invertovana slika: {ime_fajla}")
    else:
        print(f"nije pronadjena slika {ime_fajla}")
