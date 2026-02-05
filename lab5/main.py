import cv2 as cv
from cv2 import aruco
import numpy as np
import glob

MARKER_SIZE = 0.04
MARKER_SEPARATION = 0.01

# definisanje recnika i detektora
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
aruco_params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# kreiranje ocekivane table
board = aruco.GridBoard((5, 7), MARKER_SIZE, MARKER_SEPARATION, aruco_dict)


def kalibracija(putanja):

    print("pokrenuta kalibracija.........")

    # gde su tacke u 3d prostoru
    prostor_tacke = []
    # gde su tacke pale na 2d senzor kamere
    slika_tacke = []

    slike = glob.glob(putanja)
    if not slike:
        print(f"greska pri ucitavanju slika na putanji: {putanja}")
        return None, None, None, None

    visina, sirina = 0, 0
    for putanja_slike in slike:
        slika = cv.imread(putanja_slike)
        if slika is None: continue
        sivo = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
        visina, sirina = sivo.shape[:2]

        uglovi, id_markera, _ = detector.detectMarkers(sivo)

        if id_markera is not None and len(id_markera) > 0:
            prostor_t, slika_t = board.matchImagePoints(uglovi, id_markera)
            if prostor_t is not None and slika_t is not None:
                prostor_tacke.append(prostor_t)
                slika_tacke.append(slika_t)

    # pravi se geometrija izmedju kamere i prostora i resava se problem distorzije
    ocena_kalibracije, matrica, distorzija, rotacija, translacija = cv.calibrateCamera(prostor_tacke, slika_tacke, (sirina, visina), None, None)

    # racunamo novu matricu kamere koja kontrolise proces ispravljanja distorzije kako ne bi bile crne ivice
    nova_matrica, region = cv.getOptimalNewCameraMatrix(matrica, distorzija, (sirina, visina), 1, (sirina, visina))
    print(f"kalibracija zavrsena... ocena kalibracije: {ocena_kalibracije}")
    return matrica, distorzija, nova_matrica, region


def crtanje_markera(slika, uglovi, id_markera):

    if id_markera is None:
        return slika

    # prolazimo kroz sve markere
    for (marker_ugao, marker_id) in zip(uglovi, id_markera):
        # uglovi su oblika (1, 4, 2), pretvaramo u (4, 2)
        tacke = marker_ugao.reshape((4, 2)).astype(np.int32)

        # crtanje takne zelene linije oko svakog markera
        cv.polylines(slika, [tacke], True, (0, 255, 0), 1)

        # id u sredini svakog markera u proseku x i y koordinata
        centar_x = int(np.mean(tacke[:, 0]))
        centar_y = int(np.mean(tacke[:, 1]))
        cv.putText(slika, f"id={marker_id[0]}", (centar_x - 10, centar_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # crveni kvadratic u uglu
        # u aruco sistemu redosled je 0 - gore levo, 1 - gore desno, 2 - dole desno, 3 - dole levo
        bottom_right = tacke[2]
        # crtanje malog kvadrata
        pt1 = (bottom_right[0] - 2, bottom_right[1] - 2)
        pt2 = (bottom_right[0] + 2, bottom_right[1] + 2)
        cv.rectangle(slika, pt1, pt2, (0, 0, 255), -1)

    return slika


def crtanje_osa(slika, matrica, distorzija, rotacija, translacija, duzina):

    tacke_osa = np.float32([[0, 0, 0], [duzina, 0, 0], [0, duzina, 0], [0, 0, -duzina]]).reshape(-1, 3)

    # na osnovu svih prosledjenih parametara se racuna na kojim pikselima na ekranu se tacke pojavljuju
    tacke_slika, _ = cv.projectPoints(tacke_osa, rotacija, translacija, matrica, distorzija)
    tacke_slika = tacke_slika.astype(int)

    # ravel samo "izravna" niz
    pocetak = tuple(tacke_slika[0].ravel())
    tacka_x = tuple(tacke_slika[1].ravel())
    tacka_y = tuple(tacke_slika[2].ravel())
    tacka_z = tuple(tacke_slika[3].ravel())

    # crtanje linija
    cv.line(slika, pocetak, tacka_x, (0, 0, 255), 3)
    cv.line(slika, pocetak, tacka_y, (0, 255, 0), 3)
    cv.line(slika, pocetak, tacka_z, (255, 0, 0), 3)

    return slika


def run_video(video_path, matrica, distorzija, nova_matrica):
    video = cv.VideoCapture(video_path)
    if not video.isOpened():
        print("video nije pronadjen")
        return

    print("pokretanje videa... q => EXIT")

    while True:
        ret, frame = video.read()
        if not ret: break

        sivo = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        uglovi, marker_id, _ = detector.detectMarkers(sivo)

        if marker_id is not None:
            crtanje_markera(frame, uglovi, marker_id)

            # procena poze
            prostor_tacke, slika_tacke = board.matchImagePoints(uglovi, marker_id)
            if len(slika_tacke) > 0:
                uspeh, rotacija, translacija = cv.solvePnP(prostor_tacke, slika_tacke, matrica, distorzija)
                if uspeh:
                    crtanje_osa(frame, matrica, distorzija, rotacija, translacija, 0.15)

        # Undistort i prikaz
        frame_undistorted = cv.undistort(frame, matrica, distorzija, None, nova_matrica)
        cv.imshow('Aruco Pose Estimation', cv.resize(frame_undistorted, (0, 0), fx=0.7, fy=0.7))

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    matrica, distorzija, nova_matrica, region = kalibracija('Aruco/*.jpg')

    if matrica is not None:
        run_video('Aruco/Aruco_board.mp4', matrica, distorzija, nova_matrica)