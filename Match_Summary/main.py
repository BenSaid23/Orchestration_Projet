import mlflow
from model_pipeline import FootballAnalysisPipeline
from datetime import datetime

def main():
    # Chemin de la vidéo défini directement dans le code
    video_path = "match.mp4"  # Mettez le nom de votre fichier vidéo ici
    
    # Initialiser MLflow
    mlflow.set_experiment("Football Analysis")
    
    # Démarrer une nouvelle run
    with mlflow.start_run():
        print("🚀 Début de l'analyse du match...")
        
        # Enregistrer les paramètres
        mlflow.log_param("video_path", video_path)
        mlflow.log_param("model", "Whisper + spaCy + BART")
        mlflow.log_param("timestamp", datetime.now().isoformat())
        
        # Initialiser le pipeline
        pipeline = FootballAnalysisPipeline()
        
        # Traiter la vidéo
        results = pipeline.process_video(video_path)
        
        # Enregistrer les résultats
        mlflow.log_text(results["transcription"], "transcription.txt")
        mlflow.log_dict({"events": results["events"]}, "events.json")
        mlflow.log_text(results["summary"], "summary_en.txt")
        mlflow.log_text(results["translated_summary"], "summary_fr.txt")
        
        # Afficher les résultats
        print("\n📋 Transcription complète enregistrée")
        print("\n⚽ Événements détectés:")
        for event in results["events"]:
            print(f"  - {event['player']}: {event['action']}")
        
        print("\n📝 Résumé en anglais:")
        print(results["summary"])
        
        print("\n📝 Résumé en français:")
        print(results["translated_summary"])
        
        print("\n✅ Analyse terminée avec succès!")

if __name__ == "__main__":
    main()