import mlflow
from datetime import datetime
from model_pipeline import FootballMatchAnalyzer  # Assurez-vous que ce fichier contient la classe

def main():
    video_path = "match.mp4"  # Chemin vers votre vidéo

    # Initialiser MLflow
    mlflow.set_experiment("Football Analysis")

    with mlflow.start_run():
        print("🚀 Début de l'analyse du match...")

        # Enregistrer des paramètres
        mlflow.log_param("video_path", video_path)
        mlflow.log_param("model", "Whisper + spaCy + BART")
        mlflow.log_param("timestamp", datetime.now().isoformat())

        # Initialisation de l’analyseur
        analyzer = FootballMatchAnalyzer()

        # Exécuter l’analyse
        results = analyzer.analyze_video(video_path)

        # Sauvegarde des résultats
        mlflow.log_text(results["transcription"], "transcription.txt")
        mlflow.log_dict({"events": results["events"]}, "events.json")
        mlflow.log_text(results["english_summary"], "summary_en.txt")
        mlflow.log_text(results["french_summary"], "summary_fr.txt")

        # Affichage
        print("\n📋 Transcription complète enregistrée")

        print("\n⚽ Événements détectés:")
        for event in results["events"]:
            print(f"  - {event['player']}: {event['action']}")

        print("\n📝 Résumé en anglais:")
        print(results["english_summary"])

        print("\n📝 Résumé en français:")
        print(results["french_summary"])

if __name__ == "__main__":
    main()
