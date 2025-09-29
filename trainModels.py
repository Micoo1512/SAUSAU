from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dataSet import X_final, y, pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sn

'''
Proces treninga ide ispocetka samo sto je sada dataSet promijenjen
'''

#--------------------------------------
#--------------------------------------
#--------------------------------------
#Treniranje Linearnih Modela-----------
#--------------------------------------
#--------------------------------------
#--------------------------------------

#funkcija za crtanje grafika tacnosti
def plotPred(X_test, y_test, model):
    model_name = model.__class__.__name__
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    score = r2_score(y_test, y_pred)
    print(f"\n Predvidjanje modela:{model}")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, Score: {score:.4f}")

    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)

    line_coords = [min(y_test.values.ravel().min(), y_pred.min()), max(y_test.values.ravel().max(), y_pred.max())]
    plt.plot(line_coords, line_coords, '--', color='red', lw=2, label='Savrsena Predikcija')

    plt.title(f'Performanse za {model_name} (R²: {score:.2f})')
    plt.xlabel('Stvarne Vrednosti (cnt)')
    plt.ylabel('Predvidjene Vrednosti (cnt)')
    plt.legend()
    plt.grid(True)
    plt.show()

#podjela skupa
#formiranje odnosa za train, test i val
lenght = len(X_final)
train_size = int(lenght*0.8)
test_size = int(lenght*0.1)
val_size = int(lenght*0.1)

#ovako je formiram radi hronologije, kako bi trenirao na dogadjajima koji su se desili prije
X_train = X_final[:train_size]
y_train = y[:train_size]
X_val = X_final[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X_final[train_size+val_size:]
y_test = y[train_size+val_size:]

linear_models = [Lasso, Ridge]
decision_models = [DecisionTreeRegressor, RandomForestRegressor]

#prvo treniranje na linearnim modelima ----> Stacking

params = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

best_alpha_r = None
best_score_r = -1
best_model_r = None

best_alpha_l = None
best_score_l = -1
best_model_l = None

min_mse = -1

errors_res = {}

for models in linear_models:
    model_name = models.__name__

    alpha_list = []
    score_list = []
    rmse_list = []
    mae_list = []

    for alphas in params:

        if models == Lasso:
            model = models(alpha=alphas, max_iter=25000)
        else:
            model = models(alpha=alphas)
        model.fit(X_train, y_train.values.ravel())

        #nakon treninga ide validacija
        y_pred = model.predict(X_val)
        score = model.score(X_val, y_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)

        #cuvanje podataka svake iteracije za plotovanje
        alpha_list.append(alphas)
        score_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)

        errors_res[model_name] = {
            'alphas': alpha_list,
            'scores': score_list,
            'rmses': rmse_list,
            'maes': mae_list
        }

        if isinstance(model, Ridge):
            if score > best_score_r:
                best_score_r = score
                best_alpha_r = alphas
                best_model_r = model
        else:
            if score > best_score_l:
                best_score_l = score
                best_alpha_l = alphas
                best_model_l = model

#Feature Importance kod linearnih modela
best_linear_models = {
    "Ridge": best_model_r,
    "Lasso": best_model_l
}

feature_names = X_train.columns

for model_name, model in best_linear_models.items():
    if model is not None:
        #uzimanje koeficjenata
        coefficients = model.coef_

        # Kreiramo DataFrame za laksu analizu
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        })

        # Dodajemo apsolutne vrednosti za sortiranje po uticaju
        coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

        print(f"\nKoeficijenti za najbolji {model_name} model (poredjani po uticaju):")
        # Prikazujemo samo Feature i Coefficient za preglednost
        print(coef_df[['Feature', 'Coefficient']])

        # Vizualizacija
        plt.figure(figsize=(12, 8))
        sn.barplot(x='Coefficient', y='Feature', data=coef_df, hue='Feature', palette='viridis', legend=False)
        plt.title(f'Vaznost atributa (Koeficijenti) - {model_name} model')
        plt.xlabel('Vaznost atributa')
        plt.ylabel('Atribut (Feature)')
        plt.axvline(0, color='black', linewidth=0.8) # Linija na nuli
        plt.show()




#--------------------------------------
#--------------------------------------
#--------------------------------------
#Treniranje DecisionTree i RandomForest
#--------------------------------------
#--------------------------------------
#--------------------------------------

dt_param_grid = {
    "max_depth": [None, 3, 5, 7, 10, 15, 20],
    "min_samples_split": [2, 5, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "random_state": [42]
}

rf_param_grid = {
    "n_estimators": [1, 50, 100, 200],
    "max_depth": [None, 3, 5, 7, 10, 15, 20],
    "min_samples_split": [2, 5, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "random_state": [42]
}

gb_param_grid = {
    "n_estimators": [10, 50, 100, 200],
    "max_depth": [None, 3, 5, 7, 10, 15, 20],
    "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.5],
    "random_state": [42]
}



#trening predugo traje pa je zakomentarisan vec je jednom istreniran

best_models = {}
print('\n REZULTATI TRENIRANJA DECISION MODELA')

for model in decision_models:
    model_name = model.__name__

    if model_name == 'DecisionTreeRegressor':
        print(f"\n Treniranje modela: {model_name}...")
        grid_search = GridSearchCV(model(), dt_param_grid, cv = 5, scoring = 'r2', n_jobs = -1)
        grid_search.fit(X_train, y_train.values.ravel())

        print(f"\n Najbolji parametri za {model_name} su: {grid_search.best_params_}")
        print(f"\n Najbolji score za {model_name} je: {grid_search.best_score_:.4f}")

        best_models[model_name] = grid_search.best_estimator_

    elif model_name == 'RandomForestRegressor':
        print(f"\n Treniranje modela: {model_name}...")
        grid_search = GridSearchCV(model(), rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train.values.ravel())

        print(f"\n Najbolji parametri za {model_name} su: {grid_search.best_params_}")
        print(f"\n Najbolji score za {model_name} je: {grid_search.best_score_:.4f}")

        best_models[model_name] = grid_search.best_estimator_

#Feature importance za ostale modele
rf_model = best_models['RandomForestRegressor']
dt_model = best_models['DecisionTreeRegressor']

feature_name = X_train.columns

importances_r = rf_model.feature_importances_
importances_d = dt_model.feature_importances_

importance_rf = pd.DataFrame({
    'Feature': feature_name,
    'Importance': importances_r
})

importance_dt = pd.DataFrame({
    'Feature': feature_name,
    'Importance': importances_d
})

importance_rf = importance_rf.sort_values(by='Importance', ascending=False)
importance_dt = importance_dt.sort_values(by='Importance', ascending=False)

print("Vaznost atributa za RandomForest model:")
print(importance_rf)

print("Vaznost atributa za DecisionTree model:")
print(importance_dt)

plt.figure(figsize=(12, 8))
sn.barplot(x='Importance', y='Feature', data=importance_rf)
plt.title('Vaznost atributa (Feature Importance) - RandomForest')
plt.xlabel('Vaznost')
plt.ylabel('Atribut (Feature)')
plt.show()

plt.figure(figsize=(12, 8))
sn.barplot(x='Importance', y='Feature', data=importance_dt)
plt.title('Vaznost atributa (Feature Importance) - DecisionTree')
plt.xlabel('Vaznost')
plt.ylabel('Atribut (Feature)')
plt.show()

'''
#Trening za GradientBoosting naknadno
#------------------------------------
print("\n Treniranje modela: GradientBoostingRegressor...")
grid_search = GridSearchCV(GradientBoostingRegressor(), gb_param_grid, cv = 5, scoring = 'r2', n_jobs = -1)
grid_search.fit(X_train, y_train.values.ravel())

print(f"\n Najbolji parametri za GradientBoostingRegressor su: {grid_search.best_params_}")
print(f"\n Najbolji score za GradientBoostingRegressor je: {grid_search.best_score_:.4f}")
#------------------------------------
'''


#Vec znam najbolje parametre samo ih uvrstavam
best_decisionTree = DecisionTreeRegressor(max_depth = 20, min_samples_leaf = 8, min_samples_split = 20, random_state = 42)
best_randomForest = RandomForestRegressor(max_depth = None, min_samples_leaf = 4, min_samples_split = 2, n_estimators = 200, random_state = 42)
best_decisionTree.fit(X_train, y_train.values.ravel())
best_randomForest.fit(X_train, y_train.values.ravel())


#Glavni dio za pokretanje plotovanja
if __name__ == "__main__":
    if best_model_r:
        print(f"Najbolji Ridge model: alpha={best_alpha_r}, Score={best_score_r:.4f}")
    if best_model_l:
        print(f"Najbolji Lasso model: alpha={best_alpha_l}, Score={best_score_l:.4f}")

    #Plotovanje metrika linearnih modela
    #plotovanje alpha vs score i greske
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Poređenje Metrika u Zavisnosti od Alpha', fontsize=16)

    # Plot 1:Score
    for model_name, results in errors_res.items():
        axes[0].plot(results['alphas'], results['scores'], marker='o', label=model_name)
    axes[0].set_title('Score vs. Alpha')
    axes[0].set_ylabel('Score')
    axes[0].legend()

    # Plot 2:RMSE
    for model_name, results in errors_res.items():
        axes[1].plot(results['alphas'], results['rmses'], marker='o', label=model_name)
    axes[1].set_title('RMSE and Alpha')
    axes[1].set_ylabel('RMSE')
    axes[1].legend()

    # Plot 3:MAE
    for model_name, results in errors_res.items():
        axes[2].plot(results['alphas'], results['maes'], marker='o', label=model_name)
    axes[2].set_title('MAE and Alpha')
    axes[2].set_ylabel('MAE')
    axes[2].set_xlabel('Alpha')
    axes[2].legend()

    # Postavljamo X osu na logaritamsku skalu
    plt.xscale('log')

    #plt.show()

    #plotovanje predikcija za linearne
    plotPred(X_test, y_test, best_model_l)
    plotPred(X_test, y_test, best_model_r)

    #plotovanje predikcija za ostale
    plotPred(X_test, y_test, best_decisionTree)
    plotPred(X_test, y_test, best_randomForest)









