import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from model_pipeline import (
    load_data, clean_data, feature_engineering,
    treat_outliers, prepare_data, select_relevant_features,
    train_model, evaluate_model
)

def run_pipeline(data_path):
    # Initialisation MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Player_Injury_Duration_Prediction")
    
    with mlflow.start_run():
        # 1. Chargement des données
        raw_data = load_data(data_path)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("initial_rows", raw_data.shape[0])
        mlflow.log_param("initial_columns", raw_data.shape[1])
        
        # 2. Nettoyage des données (identique au notebook)
        cleaned_data = clean_data(raw_data)
        
        # 3. Feature engineering (identique au notebook)
        engineered_data = feature_engineering(cleaned_data)
        
        # 4. Traitement des outliers (identique au notebook)
        outlier_treated_data = treat_outliers(engineered_data)
        
        # 5. Préparation finale (encodage + normalisation)
        prepared_data, label_encoders, scaler = prepare_data(outlier_treated_data)
        
        # 6. Sélection des features (identique au notebook)
        X, y, top_features, selector = select_relevant_features(prepared_data, k=6)
        mlflow.log_param("selected_features", str(top_features))
        
        # 7. Split train/test (mêmes paramètres que le notebook)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        # 8. Entraînement du modèle (mêmes paramètres que le notebook)
        model_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        }
        model = train_model(X_train, y_train, model_params)
        mlflow.log_params(model_params)
        
        # 10. Création du plot des features sélectionnées
        plt.figure(figsize=(10, 6))
        feature_importance = pd.DataFrame({
            'Feature': top_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Sauvegarde du plot comme image temporaire
        plot_path = "feature_importance_plot.png"
        plt.savefig(plot_path)
        plt.close()
        
        # 11. Enregistrement des artefacts dans MLflow
        mlflow.log_artifact(plot_path) 

        # 9. Évaluation
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics({
            'mse': metrics['mse'],
            'mae': metrics['mae'],
            'r2': metrics['r2']
        })
        
        # 10. Enregistrement des artefacts
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_dict(
            {"feature_importance": dict(zip(top_features, model.feature_importances_))},
            "feature_importance.json"
        )
        
        print(f"Run terminé avec succès.")
        print(f"R2: {metrics['r2']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"Top features: {top_features}")
        print(f"Feature importance: {model.feature_importances_}")

if __name__ == "__main__":
    run_pipeline("player_injuries_impact.csv")