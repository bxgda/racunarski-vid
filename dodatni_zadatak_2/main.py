import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ucitavanje modela
model = load_model('model_slovo_p.h5')

def detekcija_slova_p(image_path):
    slika = cv2.imread(image_path)

    if slika is None:
        print("nije pronadjena slika")
        return

    slika_za_prikaz = slika.copy()
    crno_bela_slika = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

    # ulazna slika se vec poklapa sa slikama na kojima je istreniran model ali svakako ovo radimo za svaki slucaj
    zamucena_slika = cv2.GaussianBlur(crno_bela_slika, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(zamucena_slika, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # promalazenje kontura
    konture, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    visina_slike, sirina_slike = slika.shape[:2]
    maksimalna_povrsina = (visina_slike * sirina_slike) * 0.5

    detektovano_p = []

    # prvi prolaz gde prikupljamo svako potencijalno P
    for k in konture:
        povrsina = cv2.contourArea(k)
        x, y, w, h = cv2.boundingRect(k)

        # filtriranje suma i prevelikih objekata
        if povrsina > maksimalna_povrsina or povrsina < 100:
            continue

        # isecanje sa malim padding-om (tako je lakse nasem modelu)
        pad = 2
        y1, y2 = max(0, y - pad), min(visina_slike, y + h + pad)
        x1, x2 = max(0, x - pad), min(sirina_slike, x + w + pad)
        slika_konture = slika[y1:y2, x1:x2]

        if slika_konture.size == 0: continue

        # priprema za model
        prepravljena_slika_konture = cv2.resize(slika_konture, (32, 32)) / 255.0
        prepravljena_slika_konture = np.expand_dims(prepravljena_slika_konture, axis=0)

        # predvidjanje modela
        predikcija = model.predict(prepravljena_slika_konture, verbose=0)[0][0]

        # ako je predikcija veca od 0.9 onda mi kazemo da je model nasao P
        if predikcija > 0.9:
            detektovano_p.append([x, y, w, h, predikcija])

    # crtanje rezultata
    for (x, y, w, h, prob) in detektovano_p:
        cv2.rectangle(slika_za_prikaz, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # trebalo mi je dok nisam lepo istrenirao model da vidim koji oblik ima koliku predikciju
        cv2.putText(slika_za_prikaz, f"{prob:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        print(f"detektovano P na koordinatama: ({x}, {y}) sa predikcijom od: {prob:.4f}")

    print(f"ukupno detektovanih slova P: {len(detektovano_p)}")

    # prikaz i cuvanje
    cv2.imshow('rezultat', slika_za_prikaz)
    cv2.imwrite('izlaz.png', slika_za_prikaz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detekcija_slova_p('ulaz.png')
