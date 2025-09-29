from dataAnalysis import openFile, pd, StandardScaler
import numpy as np

'''
U pocetku su se podaci formirali tako da neke kolone budu rasparcane,odnosno
,ako imamo kolonu hr koja moze imati vrijednosti od 0 do 23 da to rasparcam u
24 kolone sa mogucim vrijjednostima True i False

Sada ce ovdje biti dosta zakomentarisanih linija i one ce imati oznaku 
1 za formiranje finalDataSet.csv na kome je postignut max score od 0.61
2 za formiranje novog .csv fajla (sin i cos, bez tih kolona) na kome ce se opet trenirati modeli
3 za formiranje novog .csv fajla (sin i cos, sa tim fajlovima)
4 za formiranje novog .csv fajla ,isti kao prvi samo iskljucen parametar dropFirst
5 za formiranje novog .csv fajla, isti kao 3 samo sa kolonm yr
6 za formiranje novog .csv fajla, sadrzi samo najbitnije atribute
7 za formiranje novog .csv fajla, sadrzi kao 5 samo bez hr mnth i weekday

Ideja je da se kruzni podaci prikazu preko sin i cos funkcije 
To ce se uraditi za hr mnth i weekday jer su najfrekventnije
'''

#Nakon EDA u dataAnalysis.py formiranje konacnog dataSet-a
openFile = openFile.drop('temp', axis=1) #1

"""
#odredjeni nakon feature importance
najvazniji_atributi = [
        'hr', 'mnth', 'weekday', # Originali za sin/cos
        'yr',
        'atemp',
        'workingday',
        'hum',
        'season',
        'weathersit',
        'windspeed',
        'cnt' # Target kolona uvek mora biti tu
    ]
"""
#Kreiranje novih num podataka
openFile['hr_sin'] = np.sin(2 * np.pi * openFile['hr'] / 24) #2
openFile['hr_cos'] = np.cos(2 * np.pi * openFile['hr'] / 24) #2
#openFile = openFile.drop('hr', axis=1) #2, a 3 kada se izbaci

openFile['mnth_sin'] = np.sin(2 * np.pi * openFile['mnth'] / 12) #2
openFile['mnth_cos'] = np.cos(2 * np.pi * openFile['mnth'] / 12) #2
#openFile = openFile.drop('mnth', axis=1) #2, a 3 kada se izbaci

openFile['weekday_sin'] = np.sin(2 * np.pi * openFile['weekday'] / 7) #2
openFile['weekday_cos'] = np.cos(2 * np.pi * openFile['weekday'] / 7) #2
#openFile = openFile.drop('weekday', axis=1) #2, a 3 kada se izbaci

#Ovo ostaje slicno samo sto se neke kolone oduzmu
numeric_columns = ['atemp', 'hum', 'windspeed', 'hr_cos', 'hr_sin', 'mnth_sin', 'mnth_cos', 'weekday_sin', 'weekday_cos'] #za 2 i 3
#numeric_columns = ['atemp', 'hum', 'windspeed'] #za 1 i 4
category_columns = ['season', 'holiday', 'workingday', 'weathersit', 'yr', 'hr', 'mnth', 'weekday']
scaler = StandardScaler()
target = ['cnt'] #1


numeric_data = openFile[numeric_columns] #1,2,3,4
category_data = openFile[category_columns] #1,2,3,4

y = openFile[target] #1,2,3,4

#skaliranje numerickih podataka || Ostaje isto
scaler = StandardScaler()
scale_num_data = pd.DataFrame(scaler.fit_transform(numeric_data), columns = numeric_columns)

#encoding kategorijskih podataka
enc_cate_data = pd.get_dummies(category_data, columns = category_columns, drop_first = True) #false -> drop_first samo za 4
#True da se izbjegne redudantnost

"""
kolone_za_izbacivanje = [
    'mnth_2', 'mnth_3', 'mnth_4', 'mnth_5',
    'mnth_6', 'mnth_7', 'mnth_12',

    'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5',

    'weathersit_4',


    'season_3'
]
"""

#Konacan dataSet
X_final = pd.concat([scale_num_data, enc_cate_data], axis = 1)
#X_final = X_final.drop(columns = kolone_za_izbacivanje) #6 izbacivanje "nepotrebnih" kolona
#Kreiranje .csv fajla
#X_final.to_csv('finalDataSet.csv', index = False)
#Kreiranje novog .csv fajla sa izmjenjenim podacima 2
#X_final.to_csv('finalDataSet2.1.csv', index = False)
#Kreiranje novog .csv fajla sa izmjenjenim podacima 3
#X_final.to_csv('finalDataSet3.csv', index = False)
#Kreiranje novog .csv fajla sa izmjenjenim podacima 4
#X_final.to_csv('finalDataSet4.csv', index = False)
#Kreiranje novog .csv fajla sa izmjenjenim podacima 5
X_final.to_csv('finalDataSet5.csv', index = False)
#Kreiranje novog .csv fajla sa izmjenjenim podacima 6
#Sve je vezano za plan 6
#print("\nFinalni skup podataka (X_final) je kreiran sa sledecim brojem atributa:", X_final.shape[1])
#print("Atributi:", X_final.columns.tolist())
#X_final.to_csv('finalDataSet6.csv', index = False)
#Kreiranje novog .csv fajla sa izmjenjenim podacima 7
#X_final.to_csv('finalDataSet7.csv', index = False)
