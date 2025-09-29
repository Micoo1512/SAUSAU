from dataImport import openFile
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

from sklearn.preprocessing import StandardScaler

#Izdvajanje podataka koji imaju jasne brojne podatke
#Ostale kolone od interesa su vise neka stanja predstavljena int vrijednostima
numeric_columns_m = ['temp', 'atemp', 'hum', 'windspeed', 'cnt'] # i target
numeric_columns = ['temp', 'atemp', 'hum', 'windspeed'] # sluzi za kombinovan rezultat
category_columns = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
target = ['cnt']

numeric = openFile[numeric_columns_m]

numeric_data = openFile[numeric_columns]
category_data = openFile[category_columns]
y = openFile[target]

#skaliranje numerickih podataka
scaler = StandardScaler()
scale_num_data = pd.DataFrame(scaler.fit_transform(numeric_data), columns = numeric_columns)

#encoding kategorijskih podataka
enc_cate_data = pd.get_dummies(category_data, columns = category_columns, drop_first = True)
#True da se izbjegne redudantnost

#Korelaciona matrica i plot sa heat mapom
corr_matrix = numeric.corr()

if __name__ == '__main__':
    plt.figure(figsize=(8, 8))
    sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Korelaciona Matrica za numericke vrijenosti')
    plt.show()

    print("\n--- 2. Vizualizacija odnosa: Temperatura vs. Iznajmljivanja ---")
    plt.figure(figsize=(10, 6))
    sb.scatterplot(x='atemp', y='cnt', data=openFile, alpha=0.3)
    plt.title('Odnos subjektivne temperature i broja iznajmljivanja')
    plt.xlabel('Subjektivna Temperatura (atemp)')
    plt.ylabel('Broj Iznajmljivanja (cnt)')
    plt.show()

    print("\n--- 3. Vizualizacija odnosa: Sat u danu vs. Iznajmljivanja ---")
    plt.figure(figsize=(14, 7))
    sb.barplot(x='hr', y='cnt', data=openFile)
    plt.title('Prosečan broj iznajmljivanja po satu u danu')
    plt.xlabel('Sat (hr)')
    plt.ylabel('Prosečan broj iznajmljivanja (cnt)')
    plt.show()

    print("\n--- 4. Provera uspešnosti skaliranja numeričkih podataka ---")
    plt.figure(figsize=(10, 6))
    sb.boxplot(data=scale_num_data)
    plt.title("Provera distribucije nakon skaliranja")
    plt.xlabel("Kolone")
    plt.ylabel("Skalirane vrednosti")
    plt.show()

    print("\n--- 5. Rezultati obrade podataka ---")
    print("Finalni skup podataka spreman za model (prvih 5 redova):")
    print(X_final.head())
    print(f"\nOriginalni broj kolona za model: {len(numeric_columns) + len(category_columns)}")
    print(f"Novi broj kolona nakon obrade (spreman za model): {X_final.shape[1]}")



