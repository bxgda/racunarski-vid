from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ovo resava problem malog broja varijacija slika i pravimo dodatne varijacije
dodatne_varijacije_slika = ImageDataGenerator(
    rescale=1. / 255,               # menja vrednost piksela na 0 i 1
    rotation_range=360,             # nasumicno rotira sliku za 360 stepeni
    width_shift_range=0.15,         # nasumicno pomera sliku levo-desno za 15%
    height_shift_range=0.15,        # nasumicno pomera sliku gore-dole za 15%
    shear_range=0.2,                # nasumicno krivi sliku za 20%
    zoom_range=0.2,                 # nasumicno zumira ili odzumira sliku za 20%
    fill_mode='nearest'             # odredjuje kako da popuni "prazne" piksele koji nastanu
                                    # tokom rotacije ili pomeranja i u ovom slucaju ce uvek crna da bude
)

# flow_from_directory automatski pravi klase na osnovu foldera koji se nalaze u 'dataset', P je jedna klasa i Ostalo druga
trening_generator = dodatne_varijacije_slika.flow_from_directory(
    'dataset',                      # putanja do glavnog foldera
    target_size=(32, 32),           # sve slike modifikujemo na 32x32 zbog uniformnosti
    batch_size=8,                   # uzima se 8 po 8 slika
    class_mode='binary',            # posto imamo samo 2 klase uzimamo binarnu klasifikaciju
    subset='training'
)

# arhitektura modela
model = models.Sequential([
    # definise velicinu ulazne slike i kanala boja (3 za svaki slucaj)
    layers.Input(shape=(32, 32, 3)),

    # ovo je skener koji trazi ivice i oblike, 32 filtera se primenjuju
    layers.Conv2D(32, (3, 3), activation='relu'),

    # smanjuje dimenzije slike kako bi zadrzao samo najbitnije informacije
    layers.MaxPooling2D((2, 2)),

    # gasi 25% veza tokom treninga da modelne bi naucio slike "napamet" nego da uci opste karakteristike
    layers.Dropout(0.25),

    # jos jedan skener sa vise filtera
    layers.Conv2D(64, (3, 3), activation='relu'),

    # pretvara 2D sliku u 1D niz brojeva
    layers.Flatten(),

    # povezani sloj od 64 neurona koji donose zakljucke na osnovu pronadjenih oblika
    layers.Dense(64, activation='relu'),

    # jos agresivnija zastita od ucenja napamet, sada gasimo 50% neurona
    layers.Dropout(0.5),

    # sigmoid aktivacija gde model kaze da li je siguran da je nesto P ili ne
    layers.Dense(1, activation='sigmoid')
])

# kompilacija modela
model.compile(
    optimizer='adam',               # adam je algoritam koji optimizuje brzinu ucenja
    loss='binary_crossentropy',     # standardna funkcija gubitka za binarnu klasifikaciju
    metrics=['accuracy']            # za pracenje tacnosti tokom treninga
)

# trening: model ce proci 600 puta kroz sve slike (pokusavao sam sa menj epoha ali tek ovaj broj zadovoljava potrebe ulazne slike)
model.fit(trening_generator, epochs=600)

# cuvamo model
model.save('model_slovo_p.h5')
