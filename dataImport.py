import pandas as pd

openFile = pd.read_csv('data.csv')

#U ovom fajlu se vrsi i analiza podataka radi utvrdjivanja da li postoje neke anomalije

seasons = {1, 2, 3, 4} #kolona season i njene podrazumijevane vrijednosti
months = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} #kolona mnth i njene podrazumijevane vrijednosti
hours = {i for i in range(0, 24)}  #kolona hr...
work_and_holiday = {0, 1} #kolone holiday i workingday...
weekdays = {0, 1, 2, 3, 4, 5, 6} #kolona weekday...
weather_categories = set()
#provjera dostupnih vrijednosti za weathrsit(koje kategorije postoje)

for vrijeme in openFile['weathersit']:
    weather_categories.add(vrijeme)



#Temperature
temperaturesS = set() #veoma niske tempretaure
temperaturesB = set() #veoma visoke temperature
#provjera vrijednosti temperature
broj_visokih_temp = 0
for i, (temp1, temp2) in enumerate(zip(openFile['temp'], openFile['atemp'])):
    if temp1 < 0 and temp2 < 0:
        temperaturesS.add(i)
    if temp1 > 0.5 and temp2 > 0.5: #isprobano vise vrijednosti i ima ih dosta za preko 0.5, ne ide preko 1
        #temperaturesB.add(i) #da se vidi na kojim su pozcijama
        broj_visokih_temp += 1



#provjera za humidity
broj_hum = 0
vlaga = set()
for i, vlaznost in enumerate(openFile['hum']):
    if vlaznost > 1 or vlaznost < 0.1:
        broj_hum += 1
        vlaga.add(i)
#print(broj_hum, vlaga)
#print(openFile.iloc[1564]['hum']) #Ovo su vrijednosti koje su manje od 0.1 i to su uglv 0.0
vjetar = set()
for i, brz_vjetra in enumerate(openFile['windspeed']):
    if brz_vjetra > 0.8 or brz_vjetra < 0:
        vjetar.add(i)

#print(vjetar) #postoje 4 vrijednosti preko 0.8 ali ni jedna preko 1
#provjera za prethodne podatke
anomalije = []
for i, (sezone, mjeseci, sati, praznici, radni_dani, dani) in enumerate(zip(openFile['season'], openFile['mnth'], openFile['hr'], openFile['holiday'], openFile['workingday'], openFile['weekday'])):
    if sezone not in seasons:
        anomalije.append(i)
    if mjeseci not in months:
        anomalije.append(i)
    if sati not in hours:
        anomalije.append(i)
    if praznici not in work_and_holiday:
        anomalije.append(i)
    if radni_dani not in work_and_holiday:
        anomalije.append(i)
    if dani not in weekdays:
        anomalije.append(i)

#Kada se uradi print utvrdjeno je da ne postoje anomalije i da su vrijednosti u svim kolonama odgovarajuce
#print(f"Anomalije su na pozicijama: {anomalije}")

bicikla = []
for i, ukupan_broj in enumerate(openFile['cnt']):
    if ukupan_broj != int(ukupan_broj) or ukupan_broj < 0:
        bicikla.append(i)
#niz je prazan svi su brojevi tip int i veci od nule
#print(bicikla)

if __name__ == '__main__' :
    print(f"Vrijeme tj kategorije mogu biti:{weather_categories}")
    print(f"Anomalije(na kojim pozicijama)za niske: {temperaturesS}, i velike: {temperaturesB}")
    print(f'Koliko je temp preko 0.5:{broj_visokih_temp} i to je procentualno:{broj_visokih_temp / len(openFile)}')




