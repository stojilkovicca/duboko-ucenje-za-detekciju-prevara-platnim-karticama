# Detekcija finansijskih prevara pomoću neuronskih mreža

Tema projekta je automatska detekcija prevarnih transakcija kreditnih kartica pomoću neuronskih mreža. Sistem na osnovu numeričkih obeležja transakcije predviđa da li je ona legitimna ili lažna, na primer:

Ulaz: Vektor od 30 obeležja transakcije (V1–V28, Amount, Time)

Izlaz: 0 — regularna transakcija / 1 — prevara

Projekat je implementiran u jednom Jupyter Notebook fajlu:

`projekat_DUN.ipynb`

Notebook obuhvata kompletan tok rada: učitavanje podataka, analizu i vizualizaciju, preprocesiranje, treniranje tri modela, hiperparametarsku optimizaciju, evaluaciju i čuvanje najboljeg modela.

---

## 1. Opis problema

Detekcija prevarnih transakcija kreditnih kartica predstavlja problem binarne klasifikacije na visoko neuravnoteženom datasetu.

U realnom platnom sistemu, svaka transakcija se procenjuje i klasifikuje kao legitimna ili lažna. Cilj ovog projekta je da se napravi model koji automatski prepoznaje prevarne transakcije na osnovu numeričkih obeležja.

Primer:

Obeležja transakcije:

```
V1=-1.36, V2=0.07, ..., Amount=149.62
```

Predviđena klasa:

```
0 — Regularna transakcija
```

Ovakav sistem može da pomogne u:

- automatskom blokiranju sumnjivih transakcija,
- smanjenju finansijskih gubitaka,
- bržoj reakciji na pokušaje prevare,
- zaštiti korisnika kreditnih kartica.

Model kao ulaz dobija vektor numeričkih obeležja, a kao izlaz vraća binarnu odluku: regularna ili prevarna transakcija.

---

## 2. Podaci

### Izvor podataka

Za projekat je korišćen standardni benchmark dataset:

**Credit Card Fraud Detection Dataset**

Dataset nije uključen u repozitorijum i može se preuzeti sa Kaggle-a:

```
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

Dataset se nalazi u fajlu:

```
creditcard.csv
```

### Struktura podataka

Dataset sadrži ukupno:

- **284.807** transakcija
- **31** kolona

Korišćene kolone:

| Kolona | Opis |
|--------|------|
| V1 – V28 | PCA transformisana obeležja (anonimizovana) |
| Amount | Iznos transakcije |
| Time | Vreme od prve transakcije u datasetu |
| Class | Ciljna promenljiva: 0 = regularna, 1 = prevara |

### Neuravnoteženost klasa

Dataset je izrazito neuravnotežen:

| Klasa | Broj primera | Procenat |
|-------|-------------|----------|
| 0 — Regularna | 284.315 | 99,83% |
| 1 — Prevara | 492 | 0,17% |

Zbog ovako ekstremne neuravnoteženosti, sama accuracy nije dovoljna metrika za evaluaciju. Model koji sve proglasi regularnim ima 99,83% tačnost, ali ne detektuje nijednu prevaru. Zbog toga su korišćene i metrike Precision, Recall, F1-score i ROC-AUC.

### Preprocesiranje podataka

Pre treniranja modela urađeni su sledeći koraci:

- učitavanje CSV fajla,
- provera nedostajućih vrednosti (nisu pronadjene nedostajuce vrednosti),
- standardizacija kolona `Amount` i `Time` pomoću `StandardScaler`-a,
- dodavanje standardizovanih kolona scaled_Amount i scaled_Time,
- uklanjanje originalnih kolona Amount i Time,
- podela na features (`X`) i ciljnu promenljivu (`y`),
- stratifikovana podela na train i test skup.

### Podela podataka

Podaci su podeljeni stratifikovano na:

| Skup | Procenat | Broj primera |
|------|----------|-------------|
| Trening skup | 80% | 227.845 |
| Test skup | 20% | 56.962 |

Korišćena je stratifikovana podela kako bi raspodela klasa ostala ista u oba skupa.

---

## 3. Arhitektura modela

U projektu su implementirana i upoređena tri modela:

- osnovni feedforward neuronski model,
- isti model sa class weights (balansiranje klasa),
- model sa hiperparametarskom optimizacijom putem Keras Tuner-a.

### Model 1 — Osnovni model

Arhitektura modela:

```
Ulaz (30 obeležja) → Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dropout(0.3) → Dense(1, sigmoid) → Izlaz
```

Korišćena konfiguracija:

- optimizer: `Adam`
- loss: `binary_crossentropy`
- metrics: `accuracy`
- batch_size: `256`
- epochs: `20` (uz EarlyStopping)

### Model 2 — Class Weights

Identična arhitektura kao Model 1. Razlika je u tome što su tokom treninga dodeljene težine klasama proporcionalne njihovoj zastupljenosti u datasetu, pomoću `compute_class_weight('balanced')`.

Cilj je bio da model više "kazni" grešku na manjinskoj klasi (prevare) i tako podigne Recall.

### Model 3 — Keras Tuner (RandomSearch)

Kao treći model korišćena je hiperparametarska optimizacija putem `keras_tuner.RandomSearch`.

Korišćeni slojevi:

| Parametar / sloj | Objašnjenje |
|-----------------|-------------|
| Dense | Potpuno povezani sloj koji uči obrasce i veze između ulaznih podataka |
| ReLU | Aktivaciona funkcija koja omogućava modelu da uči složene nelinearne odnose |
| Dropout | Slučajno isključuje deo neurona tokom treninga i smanjuje overfitting |
| Sigmoid | Izlazna aktivaciona funkcija koja vraća verovatnoću pripadnosti klasi prevara |
| Adam | Optimizacioni algoritam koji podešava težine neuronske mreže tokom učenja |
| binary_crossentropy | Funkcija greške pogodna za probleme binarne klasifikacije |
| Accuracy | Metrika koja pokazuje procenat tačno klasifikovanih primera |

Pretraživani hiperparametri:

| Hiperparametar | Opseg vrednosti |
|----------------|----------------|
| Neuroni u 1. sloju | 32, 64, 96, 128 |
| Neuroni u 2. sloju | 16, 32, 48, 64 |
| Dropout 1 | 0.2, 0.3, 0.4 |
| Dropout 2 | 0.2, 0.3, 0.4 |
| Learning rate | 0.001, 0.0005, 0.0001 |

Konfiguracija pretrage:

- max_trials: `10`
- executions_per_trial: `1`
- objective: `val_loss`

Kao kao kriterijum izbora najboljeg modela korišćena je minimalna vrednost validacione greške (val_loss).

---

## 4. Trening

Za treniranje svih modela korišćen je TensorFlow/Keras u Google Colab okruženju.

Modeli su trenirani nad trening skupom, dok je 20% trening skupa korišćeno kao validacioni skup tokom treniranja.

Korišćeni elementi treninga:

- optimizer: `Adam`
- loss: `binary_crossentropy`
- batch_size: `256`
- epochs: `20`
- EarlyStopping (monitor: `val_loss`, patience: `3`, restore_best_weights: `True`)

### EarlyStopping

Korišćen je EarlyStopping callback kako bi se treniranje automatski zaustavilo kada se validacioni gubitak više ne poboljšava.

Korišćena konfiguracija:

- monitor = val_loss
- patience = 3
- restore_best_weights = True

Na taj način se smanjuje rizik od overfitting-a i sprečava nepotrebno treniranje modela nakon prestanka poboljšanja.

### Grafikoni treninga

Za svaki model prikazani su:

- trening i validaciona tačnost,
- trening i validacioni gubitak.

Grafikoni su korišćeni za praćenje procesa učenja i detekciju mogućeg overfitting-a.

Fiksiran je seed (42) na svim nivoima (`random`, `numpy`, `tensorflow`) radi reproduktibilnosti rezultata.

---

## 5. Analiza osetljivosti i hiperparametarska optimizacija

U projektu je urađena analiza osetljivosti kroz eksperimente sa različitim hiperparametrima kako bi se ispitalo kako promene arhitekture utiču na kvalitet klasifikacije. Pored osnovnog modela, analiziran je i uticaj class weights pristupa, kao i različitih kombinacija hiperparametara dobijenih pomoću Keras Tuner RandomSearch algoritma.

Menjani su sledeći hiperparametri:

- broj neurona u prvom Dense sloju,
- broj neurona u drugom Dense sloju,
- vrednost Dropout regularizacije u oba sloja,
- learning rate optimizatora.

Pretraživane konfiguracije putem Keras Tuner RandomSearch (10 pokušaja):

| Hiperparametar | Min vrednost | Max vrednost | Korak |
|----------------|-------------|-------------|-------|
| Neuroni u 1. sloju | 32 | 128 | 32 |
| Neuroni u 2. sloju | 16 | 64 | 16 |
| Dropout 1 | 0.2 | 0.4 | — |
| Dropout 2 | 0.2 | 0.4 | — |
| Learning rate | 0.0001 | 0.001 | — |

Keras Tuner je pronašao sledeće optimalne vrednosti minimizacijom `val_loss` na validacionom skupu:

| Hiperparametar | Pronađena vrednost |
|----------------|-------------------|
| Neuroni u 1. sloju | 128 |
| Dropout 1 | 0.3 |
| Neuroni u 2. sloju | 64 |
| Dropout 2 | 0.2 |
| Learning rate | 0.0001 |

Rezultati pokazuju da izbor hiperparametara ima značajan uticaj na performanse modela, zbog čega je neophodno pažljivo podešavanje arhitekture i praćenje validacionih rezultata.

Na osnovu pronađenih vrednosti formiran je konačni model (Model 3), koji je kasnije korišćen za završnu evaluaciju i proglašen najboljim modelom u projektu.

---

## 6. Rezultati evaluacije

Za evaluaciju modela korišćen je test skup koji nije korišćen tokom treniranja.

Korišćene metrike:

| Metrika | Objašnjenje |
|---------|-------------|
| Accuracy | Procenat ukupno tačno klasifikovanih transakcija |
| Precision | Od svih označenih kao prevara, koliko je stvarno prevara |
| Recall | Od svih pravih prevara, koliko je model uhvatio |
| F1-score | Harmonijska sredina Precision i Recall |
| ROC-AUC | Sposobnost modela da rangira prevare iznad regularnih transakcija |

### Poređenje svih modela

| Metrika | Model 1 (Osnovni) | Model 2 (Class Weights) | Model 3 (Keras Tuner) |
|---------|:-----------------:|:-----------------------:|:---------------------:|
| Accuracy | 99.94% | 97.91% | 99.94% |
| Precision | 81.63% | 7.08% | 81.00% |
| Recall | 81.63% | 91.84% | 82.65% |
| F1-score | 81.63% | 13.14% | 81.82% |
| ROC-AUC | 98.07% | 98.09% | 98.05% |

### Analiza modela

**Model 1 (Osnovni):** Savršeno balansiran Precision i Recall (81.63% / 81.63%). Solidan polazni model bez ikakvih tehnika za tretiranje neuravnoteženosti.

**Model 2 (Class Weights):** Recall je skočio na 91.84%, ali Precision je pao na svega 7.08% — model generiše veliki broj lažno pozitivnih klasifikacija. F1-score od 13.14% potvrđuje da ovaj model nije upotrebljiv u praksi.

**Model 3 (Keras Tuner):** Neznatno bolji Recall od Modela 1 (82.65% vs 81.63%) uz zadržan visok Precision (81.00%). Metodološki najjači pristup zahvaljujući sistematskoj pretrazi hiperparametara.

## Konfuziona matrica

Za najbolji model prikazana je konfuziona matrica koja omogućava detaljniji uvid u tačne i pogrešne klasifikacije.

Dobijeni rezultati:

- TN = 56845
- FP = 19
- FN = 17
- TP = 81

pokazuju da model veoma uspešno razlikuje regularne i prevarne transakcije.

Broj lažno pozitivnih klasifikacija (19) i broj lažno negativnih klasifikacija (17) ostali su veoma niski. Posebno su značajne lažno negativne klasifikacije, jer predstavljaju prevarne transakcije koje model nije uspeo da prepozna.

Dobijena konfuziona matrica potvrđuje visok Precision (81.00%) i Recall (82.65%), kao i dobru sposobnost modela da detektuje prevarne transakcije uz mali broj grešaka.

### Poređenje ROC krivih

Slične ROC-AUC vrednosti ukazuju da sva tri modela ostvaruju veoma sličnu sposobnost razdvajanja klasa. Razlike u Precision i Recall metrikama posledica su različitog tretiranja neuravnoteženosti i različitog položaja odlučne granice.

---

## 7. Diskusija

Rezultati pokazuju da se detekcija prevarnih transakcija kreditnih kartica može uspešno rešiti feedforward neuronskim mrežama čak i bez naprednih tehnika balansiranja klasa.

Model 1 (Osnovni) ostvario je iznenađujuće dobre rezultate bez ikakvih posebnih tehnika za tretiranje neuravnoteženosti. To pokazuje da je dataset, uprkos ekstremnoj neuravnoteženosti, dovoljno informativan — PCA transformisana obeležja V1–V28 nose dovoljno signala za razdvajanje klasa.

Model 2 sa class weights pokazuje klasičan trade-off između Recall-a i Precision-a. Dodela visokih težina manjinskoj klasi naterala je model da bude preoprezan i da gotovo sve transakcije proglašava prevarom. U realnom scenariju to bi značilo masovno blokiranje legitimnih transakcija, što je neprihvatljivo. Važno zapažanje je da class weights nisu promenili šta je model naučio — ROC-AUC ostaje identičan kao kod Modela 1 (~98%), što znači da su naučene reprezentacije iste. Razlika nastaje isključivo u pomeranju odlučne granice.

Model 3 sa Keras Tuner-om potvrđuje da sistematska pretraga hiperparametara donosi merljivo poboljšanje. Poboljšanje Recall-a sa 81.63% na 82.65% uz zadržan Precision nije dramatično, ali je metodološki važno — pokazuje da nasumično odabrani hiperparametri nisu optimalni i da postoji prostor za poboljšanje kroz pretragu.

Još jedan važan nalaz je identičnost ROC krivih sva tri modela. Sve tri krive su praktično nerazlučive na grafiku (AUC: 0.9807, 0.9809, 0.9805), što direktno dokazuje da razlika u tabelarnim metrikama nije posledica razlike u naučenom znanju, već razlike u načinu tretiranja neuravnoteženosti i položaju odlučne granice.

Moguća unapređenja projekta su navedena u posebnoj sekciji na kraju dokumenta.

---

## 8. Zaključak

Najbolji model je:

**Model 3 (Keras Tuner)**

Ostvareni rezultati na test skupu:

- accuracy = 99.94%
- precision = 81.00%
- recall = 82.65%
- F1-score = 81.82%
- ROC-AUC = 98.05%

Model 3 je izabran kao konačni model jer ostvaruje najbolji kompromis između Precision i Recall metrika, uz najveći F1-score među testiranim modelima.

Dobijeni rezultati pokazuju da neuronske mreže predstavljaju efikasan alat za detekciju finansijskih prevara i mogu imati značajnu primenu u realnim finansijskim sistemima.

Sačuvani fajlovi:

- `credit_card_fraud_model.keras` — sačuvani najbolji model
- `scaler.pkl` — sačuvani StandardScaler za preprocesiranje novih podataka

---

## 9. Moguća unapređenja

- primena SMOTE tehnike za sintetičko balansiranje klasa,
- primena autoenkodera za anomaly detection pristup,
- istraživanje optimalnog praga odluke umesto fiksnog 0.5,
- testiranje XGBoost modela radi poređenja sa neuronskim mrežama,
- korišćenje Bayesian optimizacije umesto RandomSearch-a,
- detaljnija analiza pogrešno klasifikovanih primera,
- dodavanje Batch Normalization slojeva.

---

## 10. Pokretanje projekta

Projekat je namenjen za pokretanje u Google Colab okruženju.

### 1. Otvoriti notebook

Otvoriti fajl:

```
projekat_DUN.ipynb
```

### 2. Obezbediti dataset

Potrebno je obezbediti fajl `creditcard.csv` i prilagoditi putanju u notebook-u.

Dataset se može preuzeti sa Kaggle-a:

```
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

### 3. Pokrenuti sve ćelije

U Google Colab-u izabrati:

```
Runtime → Run all
```

Notebook će zatim:

- učitati dataset,
- preprocesirati podatke,
- trenirati sva tri modela,
- prikazati evaluacione metrike i grafike,
- sačuvati najbolji model.

### Izlazni fajlovi

| Fajl | Opis |
|------|------|
| `credit_card_fraud_model.keras` | Sačuvani Keras model (Model 3) |
| `scaler.pkl` | Sačuvani StandardScaler objekat |

---

## 11. Korišćene biblioteke

| Biblioteka | Uloga |
|------------|-------|
| Python | Programski jezik |
| TensorFlow / Keras | Izgradnja i trening neuronskih mreža |
| Keras Tuner | Hiperparametarska optimizacija |
| scikit-learn | Preprocesiranje i evaluacione metrike |
| pandas | Upravljanje podacima |
| NumPy | Numeričke operacije |
| Matplotlib | Vizualizacija |
| Seaborn | Vizualizacija |
| joblib | Čuvanje scaler objekta |
---

## Autor

Aleksandar Stojlković, 2022/0102

Fakultet organizacionih nauka

Predmet: Duboko učenje i neuronske mreže

2025/26
