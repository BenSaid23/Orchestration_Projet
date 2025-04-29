import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(filepath):
    """Charge les données depuis un fichier CSV"""
    return pd.read_csv(filepath)

def clean_data(df):
    """Nettoyage des données identique au notebook original"""
    df = df.copy()
    
    # Liste des colonnes numériques
    numeric_cols = [
        "Match1_before_injury_GD", "Match1_before_injury_Player_rating",
        "Match2_before_injury_GD", "Match2_before_injury_Player_rating",
        "Match3_before_injury_GD", "Match3_before_injury_Player_rating",
        "Match1_missed_match_GD", "Match2_missed_match_GD",
        "Match3_missed_match_GD", "Match1_after_injury_GD",
        "Match1_after_injury_Player_rating", "Match2_after_injury_GD",
        "Match2_after_injury_Player_rating", "Match3_after_injury_GD",
        "Match3_after_injury_Player_rating"
    ]
    
    # Conversion des colonnes numériques
    df[numeric_cols] = df[numeric_cols].replace('N.A.', np.nan).apply(pd.to_numeric, errors="coerce")
    
    # Remplissage des valeurs manquantes
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    # Correction du format des dates avant conversion
    for date_col in ['Date of Injury', 'Date of return']:
        if date_col in df.columns:
            # Ajouter un espace après la virgule si nécessaire
            df[date_col] = df[date_col].astype(str).str.replace(r',(\d)', r', \1', regex=True)
            # Conversion en datetime avec gestion des erreurs
            df[date_col] = pd.to_datetime(df[date_col], format='%b %d, %Y', errors='coerce')
    
    # Remplissage des dates manquantes
    most_frequent_date_IN = df['Date of Injury'].mode()[0] if not df['Date of Injury'].mode().empty else pd.Timestamp.now()
    most_frequent_date_RE = df['Date of return'].mode()[0] if not df['Date of return'].mode().empty else pd.Timestamp.now()
    
    df['Date of Injury'] = df['Date of Injury'].fillna(most_frequent_date_IN)
    df['Date of return'] = df['Date of return'].fillna(most_frequent_date_RE)
    
    return df

def feature_engineering(df):
    """Feature engineering identique au notebook original"""
    df = df.copy()
    
    # Durée de blessure
    df['Injury Duration'] = (df['Date of return'] - df['Date of Injury']).dt.days
    
    # Moyennes des performances
    df['Average_Before_Injury_Player_Rating'] = df[
        ['Match1_before_injury_Player_rating', 'Match2_before_injury_Player_rating', 
         'Match3_before_injury_Player_rating']
    ].mean(axis=1)
    
    df['Average_After_Injury_Player_Rating'] = df[
        ['Match1_after_injury_Player_rating', 'Match2_after_injury_Player_rating', 
         'Match3_after_injury_Player_rating']
    ].mean(axis=1)
    
    # Niveau de sévérité
    df['severity_level'] = df['Injury Duration'].apply(
        lambda x: 'Low' if x <= 15 else 'Moderate' if x <= 60 else 'High'
    )
    
    # Différence de performance
    df['Player_Rating_Difference'] = df['Average_After_Injury_Player_Rating'] - df['Average_Before_Injury_Player_Rating']
    
    # Matchs manqués
    df['Missed_Matches'] = (
        df[['Match1_missed_match_GD', 'Match2_missed_match_GD', 'Match3_missed_match_GD']].sum(axis=1) > 0
    ).astype(int)
    
    # Autres features
    average_injury_duration = df['Injury Duration'].mean()
    df['Quick_Recovery'] = (df['Injury Duration'] < average_injury_duration).astype(int)
    df['Performance_Improved'] = (df['Average_After_Injury_Player_Rating'] > df['Average_Before_Injury_Player_Rating']).astype(int)
    df['Career_Duration_Before_Injury'] = (pd.to_datetime(df['Date of Injury']) - pd.to_datetime(df['Season'].str[:4] + '-01-01')).dt.days
    df['Previous_Injuries'] = df['Injury'].apply(lambda x: 1 if x != 'No Injury' else 0)
    df['Severe_Injury'] = df['severity_level'].apply(lambda x: 1 if x == 'High' else 0)
    
    # Goal Difference features
    df['Total_Before_Injury_GD'] = df[['Match1_before_injury_GD', 'Match2_before_injury_GD', 'Match3_before_injury_GD']].sum(axis=1)
    df['Total_After_Injury_GD'] = df[['Match1_after_injury_GD', 'Match2_after_injury_GD', 'Match3_after_injury_GD']].sum(axis=1)
    df['Average_Before_Injury_GD'] = df[['Match1_before_injury_GD', 'Match2_before_injury_GD', 'Match3_before_injury_GD']].mean(axis=1)
    df['Average_After_Injury_GD'] = df[['Match1_after_injury_GD', 'Match2_after_injury_GD', 'Match3_after_injury_GD']].mean(axis=1)
    df['Team_GD_Improved'] = (df['Average_After_Injury_GD'] > df['Average_Before_Injury_GD']).astype(int)
    
    return df

# Ajoutez cette fonction si elle manque
def treat_outliers(df):
    """Traitement des outliers avec la méthode IQR"""
    df = df.copy()
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, 
                          np.where(df[col] > upper_bound, upper_bound, df[col]))
    return df

def prepare_data(df):
    """Préparation finale des données avant modélisation"""
    # Suppression des doublons
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.loc[:, ~df.T.duplicated()]
    df = df.drop_duplicates()
    
    # Encodage
    cat_features = df.select_dtypes(include=['object', 'bool']).columns
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in cat_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    
    # Normalisation
    df_encoded_no_datetime = df_encoded.select_dtypes(exclude=['datetime'])
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(df_encoded_no_datetime)
    normalized_df = pd.DataFrame(normalized, columns=df_encoded_no_datetime.columns)
    
    return normalized_df, label_encoders, scaler

def select_relevant_features(df, k=6):
    """Sélection des features identique au notebook original"""
    relevant_features = [
        'FIFA rating', 'Age', 'Position', 'Injury', 'Season', 'Date of Injury',  
        'Match1_before_injury_GD', 'Match1_before_injury_Player_rating',
        'Match2_before_injury_GD', 'Match2_before_injury_Player_rating',
        'Match3_before_injury_GD', 'Match3_before_injury_Player_rating',
        'Player_Rating_Difference', 'Missed_Matches', 'severity_level',
        'Performance_Improved', 'Career_Duration_Before_Injury', 'Previous_Injuries',
        'Severe_Injury', 'Total_Before_Injury_GD', 'Total_After_Injury_GD', 
        'Team_GD_Improved', 'Average_Before_Injury_GD', 'Average_Before_Injury_Player_Rating'
    ]
    
    # Ne garder que les colonnes existantes
    available_features = [f for f in relevant_features if f in df.columns]
    X = df[available_features]
    y = df['Injury Duration']
    
    # Sélection des features
    selector = SelectKBest(score_func=f_regression, k=min(k, len(available_features)))
    X_new = selector.fit_transform(X, y)
    
    return X_new, y, X.columns[selector.get_support()].tolist(), selector

def train_model(X_train, y_train, params=None):
    """Entraînement du modèle avec les mêmes paramètres que le notebook"""
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    }
    if params:
        default_params.update(params)
    
    model = GradientBoostingRegressor(**default_params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Évaluation identique au notebook"""
    y_pred = model.predict(X_test)
    return {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'predictions': y_pred
    }