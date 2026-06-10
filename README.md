# Detekcija finansijskih prevara neuronskim mrežama

Projekat iz predmeta **Duboko učenje i neuronske mreže**.

Cilj projekta je razvoj modela zasnovanog na neuronskim mrežama za detekciju prevarnih transakcija platnim karticama. Korišćen je skup podataka sa Kaggle platforme, a implementacija je realizovana u Python-u korišćenjem TensorFlow/Keras biblioteka.

---

# 1. Opis problema

Prevare platnim karticama predstavljaju značajan problem u finansijskom sektoru. Zbog velikog broja transakcija i veoma malog broja prevara, ručno otkrivanje sumnjivih transakcija nije praktično.

Cilj ovog projekta je izgradnja modela dubokog učenja koji će na osnovu karakteristika transakcije klasifikovati da li se radi o regularnoj transakciji ili prevari.

Radi se o problemu binarne klasifikacije:

- Klasa 0 – regularna transakcija
- Klasa 1 – prevarna transakcija

---

# 2. Podaci

Korišćen je javno dostupan skup podataka:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Skup podataka sadrži:

- 284807 transakcija
- 30 ulaznih karakteristika
- 492 prevarne transakcije

Karakteristike V1-V28 predstavljaju rezultate PCA transformacije radi zaštite privatnosti korisnika.

Dodatne karakteristike:

- Time – vreme proteklo od prve transakcije
- Amount – iznos transakcije
- Class – ciljna promenljiva

Skup podataka je izrazito neuravnotežen, pošto prevarne transakcije čine približno 0.17% svih transakcija.

Pre procesiranja podataka izvršeno je:

- provera nedostajućih vrednosti,
- analiza distribucije podataka,
- standardizacija atributa Time i Amount,
- uklanjanje originalnih kolona Time i Amount,
- podela na trening i test skup.

Za podelu podataka korišćen je odnos:

- 80% trening skup
- 20% test skup

uz stratifikaciju ciljnih klasa.

---

# 3. Arhitektura modela

U projektu su analizirana tri modela.

## Model 1 – Osnovni model

Arhitektura:

- Dense(64)
- Dropout(0.3)
- Dense(32)
- Dropout(0.3)
- Dense(1)

Aktivacione funkcije:

- ReLU u skrivenim slojevima
- Sigmoid u izlaznom sloju

Korišćen optimizer:

- Adam

Funkcija greške:

- Binary Crossentropy

---

## Model 2 – Model sa class weights

Kod drugog modela uvedeni su class weights radi povećanja osetljivosti na prevarne transakcije.

---

## Model 3 – Hiperparametarski optimizovan model

Za automatsku optimizaciju hiperparametara korišćen je Keras Tuner.

Najbolja pronađena arhitektura:

- Dense(128)
- Dropout(0.3)
- Dense(64)
- Dropout(0.2)
- Dense(1)

Learning rate:

- 0.0001

---

# 4. Trening

Za treniranje modela korišćeni su:

- batch size = 256
- maksimalno 20 epoha
- validation split = 20%

Radi sprečavanja preprilagođavanja korišćen je Early Stopping mehanizam:

- monitor = val_loss
- patience = 3

Radi reproduktivnosti rezultata korišćeni su fiksirani seed-ovi.

---

# 5. Analiza osetljivosti i hiperparametarska optimizacija

Najpre je kreiran osnovni model.

Nakon toga analiziran je uticaj class weights pristupa. Uočeno je povećanje recall metrike, ali i značajan pad precision metrike, što je dovelo do velikog broja lažno pozitivnih klasifikacija.

Zatim je primenjena automatska hiperparametarska optimizacija pomoću Keras Tunera. Analizirane su različite kombinacije:

- broja neurona,
- dropout parametara,
- learning rate parametra.

Nakon 10 pokušaja pronađena je arhitektura koja ostvaruje najbolje performanse.

---

# 6. Rezultati evaluacije

Poređenje modela:

| Metrika | Model 1 | Model 2 | Model 3 |
|----------|---------:|---------:|---------:|
| Accuracy | 99.94% | 97.91% | 99.94% |
| Precision | 81.63% | 7.08% | 81.00% |
| Recall | 81.63% | 91.84% | 82.65% |
| F1-score | 81.63% | 13.14% | 81.82% |
| ROC-AUC | 98.07% | 98.09% | 98.05% |

Kao konačni model izabran je Model 3 dobijen primenom Keras Tunera.

---

# 7. Diskusija

Rezultati pokazuju da povećanje recall metrike može dovesti do značajnog smanjenja precision metrike.

Model sa class weights pristupom uspeo je da detektuje veći broj prevara, ali uz veliki broj lažno pozitivnih klasifikacija.

Automatska optimizacija hiperparametara omogućila je pronalaženje modela koji ostvaruje najbolji kompromis između precision i recall metrika.

---

# 8. Zaključak

U radu je prikazana primena dubokog učenja za detekciju finansijskih prevara.

Implementirana su tri različita modela i izvršeno je njihovo poređenje. Najbolje rezultate ostvario je model dobijen primenom Keras Tunera, koji je zbog najvećeg F1-score-a izabran kao konačni model.

Rezultati pokazuju da neuronske mreže mogu predstavljati efikasan alat za detekciju prevarnih transakcija i imati značajnu primenu u finansijskom sektoru.

---

## Autor

Aleksandar Stojlković

Fakultet organizacionih nauka

Predmet: Duboko učenje i neuronske mreže

2025/26
