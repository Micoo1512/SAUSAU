from sklearn.ensemble import StackingRegressor
from trainModels import best_model_r, best_model_l, rf_model, dt_model, X_train, y_train, X_test, y_test, plotPred, Ridge, RandomForestRegressor

estimators2 = [
    ('ridge', best_model_r),
    ('random forest', rf_model),
]

extended_model = StackingRegressor(
    estimators=estimators2,
    final_estimator=Ridge()
)
#kombinacija svih modela
extended_model.fit(X_train, y_train.values.ravel())
print("\n Obrada rezultata finalnog modela")
plotPred(X_test, y_test, extended_model)