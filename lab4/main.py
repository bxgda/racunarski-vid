import cv2
import numpy as np
import imutils


class DetektorLjubimaca:
    def __init__(self, putanja_prototxt, putanja_model, putanja_labele):
        # ucitavamo mrezu
        self.mreza = cv2.dnn.readNetFromCaffe(putanja_prototxt, putanja_model)

        # otvaramo fajl sa labelama i pravi listu
        sa_fajla = open(putanja_labele).read().strip().split("\n")
        self.imena_klasa = [red.split(" ", 1)[-1].split(",")[0].lower() for red in sa_fajla]

    def skeniraj_sliku(self, glavna_slika, velicina_prozora=180):
        # pravimo kopiju da ne bi menjali original
        privremena = glavna_slika.copy()

        # glavna petlja za sliding window koja se zavrsava kada je prozor postao premali
        while True:
            # izlazimo ako je prozor premali postao
            if privremena.shape[0] < 100 or privremena.shape[1] < 350:
                break

            odnos_skaliranja = glavna_slika.shape[1] / privremena.shape[1]

            # prolazimo kroz sliku
            for vrh in range(0, privremena.shape[0], velicina_prozora):
                for levo in range(0, privremena.shape[1], velicina_prozora):

                    # pravimo trenutni deo
                    kvadrat = privremena[vrh:vrh + velicina_prozora, levo:levo + velicina_prozora]

                    # provera da nismo dosli do kraja ivica
                    if kvadrat.shape[0] == velicina_prozora and kvadrat.shape[1] == velicina_prozora:
                        # pretvaramo sliku u blob format, skaliramo i oduzimamo srednje vrednosti rgb kanala
                        ulazni_blob = cv2.dnn.blobFromImage(kvadrat, 1, (224, 224), (104, 117, 123))
                        self.mreza.setInput(ulazni_blob)
                        rezultati = self.mreza.forward()

                        # nalazimo indeks sa  najvisom verovatnocom
                        pozicija = np.argmax(rezultati[0])
                        sigurnost = rezultati[0][pozicija]

                        # ako je sigurnost prepoznavanja objekta veca od 0.5 radimo dalje
                        if sigurnost > 0.5:
                            # ovo je model u sustini prepoznao i sasd nalazimo bas tu rec
                            tekst_klase = self.imena_klasa[pozicija]

                            # racunamo x i y koordinatu trenutne slike ali nazad na originalnu zbog odnosa skaliranja
                            x_start = int(levo * odnos_skaliranja)
                            y_start = int(vrh * odnos_skaliranja)
                            dimenzija = int(velicina_prozora * odnos_skaliranja)

                            # ako je kuce nacrtamo zuti kvadrat a ako je macka onda crveni
                            if "dog" in tekst_klase:
                                self._nacrtaj(glavna_slika, "DOG", (x_start, y_start, dimenzija), (0, 255, 255))
                            elif "cat" in tekst_klase:
                                self._nacrtaj(glavna_slika, "CAT", (x_start, y_start, dimenzija), (0, 0, 255))

            # racunamo novu sirinu koja je duplo manja i smanjujemo sliku
            nova_sirina = int(privremena.shape[1] / 2.0)
            privremena = imutils.resize(privremena, width=nova_sirina)

    def _nacrtaj(self, slika, tekst, kordinate, boja):
        x, y, d = kordinate
        cv2.rectangle(slika, (x + 2, y + 2), (x + d - 2, y + d - 2), boja, 2)
        cv2.putText(slika, tekst, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, boja, 2)


def pronadji_beli_okvir(img, sirina_cilj=1440, visina_cilj=720):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binarna = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
    pronadjene_konture, _ = cv2.findContours(binarna, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    finalne_koordinate = (0, 0)
    minimalno_odstupanje = sirina_cilj + visina_cilj

    # prolazimo kroz sve pronadjene konture
    for k in pronadjene_konture:
        # proveravamo da li je povrsina veca od 50% ciljne povrsine
        if cv2.contourArea(k) > (sirina_cilj * visina_cilj * 0.5):
            # racunamo duzinu konture (znaci da je kontura zatvorena)
            duzina = cv2.arcLength(k, True)
            # aproksimiramo konturu sa poligonom
            uglovi = cv2.approxPolyDP(k, 0.1 * duzina, True)
            # ako ima 4 ugla znaci da smo blizu
            if len(uglovi) == 4:
                # izvlacimo x koordinate uglova
                sve_x = [t[0][0] for t in uglovi]
                # izvlacimo y koordinate uglova
                sve_y = [t[0][1] for t in uglovi]

                sirina_k = max(sve_x) - min(sve_x)
                visina_k = max(sve_y) - min(sve_y)

                razlika = abs(sirina_cilj - sirina_k) + abs(visina_cilj - visina_k)

                if razlika < minimalno_odstupanje:
                    minimalno_odstupanje = razlika
                    finalne_koordinate = (min(sve_y), min(sve_x))
    # cuvamo koordinate gornjeg levog ugla i vracamo iseceni deo slike
    y, x = finalne_koordinate
    return img[y:y + visina_cilj, x:x + sirina_cilj]


# --- IZVRSAVANJE ---
if __name__ == "__main__":
    F_MODEL = "bvlc_googlenet.caffemodel"
    F_PROTO = "bvlc_googlenet.prototxt"
    F_LABELE = "synset_words.txt"
    F_ULAZ = "download.png"

    izvorna = cv2.imread(F_ULAZ)
    if izvorna is None:
        izvorna = cv2.imread("download.png")

    detektor = DetektorLjubimaca(F_PROTO, F_MODEL, F_LABELE)
    region_konture = pronadji_beli_okvir(izvorna)

    detektor.skeniraj_sliku(region_konture)

    cv2.imshow("Finalno", region_konture)
    cv2.imwrite("output.jpg", region_konture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()