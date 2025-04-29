import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from moviepy  import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
import os
import whisper
import time
import textwrap
import spacy
import re
from collections import Counter
from transformers import pipeline, BartTokenizer
from googletrans import Translator

# Configuration
TEMP_AUDIO = "temp_audio.wav"
CHUNK_FOLDER = "audio_chunks"
MIN_CHUNK_LENGTH = 30  # secondes minimum par chunk
WHISPER_MODEL = "tiny"  # tiny / base / small / medium / large

class FootballAnalysisPipeline:
    def __init__(self):
        # Charger les modèles une seule fois
        self.whisper_model = whisper.load_model(WHISPER_MODEL)
        self.nlp = spacy.load("en_core_web_sm")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.translator = Translator()
    
    def extract_audio(self, video_path):
        """Extrait l'audio de la vidéo en format WAV"""
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(TEMP_AUDIO, codec='pcm_s16le')
        return TEMP_AUDIO

    def split_audio(self, audio_path):
        """Découpe l'audio en chunks de durée égale"""
        if not os.path.exists(CHUNK_FOLDER):
            os.makedirs(CHUNK_FOLDER)

        audio = AudioSegment.from_wav(audio_path)
        duration_ms = len(audio)
        chunk_length_ms = max(MIN_CHUNK_LENGTH * 1000, duration_ms // 10)  # 10 chunks maximum

        chunks = []
        for i in range(0, duration_ms, chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            chunk_path = os.path.join(CHUNK_FOLDER, f"chunk_{i//1000}s.wav")
            exported = chunk.export(chunk_path, format="wav")
            exported.close()
            chunks.append(chunk_path)

        return chunks

    def transcribe_whisper(self, chunk_path):
        """Transcription avec Whisper"""
        try:
            result = self.whisper_model.transcribe(chunk_path, language="en")
            return result["text"]
        except Exception as e:
            print(f"Erreur Whisper : {e}")
            return ""

    def cleanup(self):
        """Nettoie les fichiers temporaires"""
        if os.path.exists(TEMP_AUDIO):
            os.remove(TEMP_AUDIO)
        if os.path.exists(CHUNK_FOLDER):
            for f in os.listdir(CHUNK_FOLDER):
                os.remove(os.path.join(CHUNK_FOLDER, f))
            os.rmdir(CHUNK_FOLDER)

    def process_video(self, video_path):
        """Traite la vidéo et retourne les résultats"""
        try:
            # 1. Extraction audio
            print("🔊 Extraction de l'audio...")
            audio_file = self.extract_audio(video_path)

            # 2. Découpage en chunks
            print("✂️ Découpage de l'audio...")
            chunks = self.split_audio(audio_file)

            # 3. Transcription
            print("📝 Transcription en cours...")
            full_text = []

            for chunk in chunks:
                print(f"  Traitement de {os.path.basename(chunk)}...")
                time.sleep(0.1)
                text = self.transcribe_whisper(chunk)
                full_text.append(text)

            # 4. Résultat final
            final_transcription = "\n".join(full_text)
            
            # 5. Extraction des événements
            events = self.extract_football_events(final_transcription)
            
            # 6. Génération du résumé
            summary = self.generate_summary(final_transcription)
            
            # 7. Traduction du résumé
            translated_summary = self.translate_summary(summary)
            
            return {
                "transcription": final_transcription,
                "events": events,
                "summary": summary,
                "translated_summary": translated_summary
            }

        finally:
            self.cleanup()

    def extract_football_events(self, text):
        """Extrait les événements de football du texte"""
        doc = self.nlp(text)

        # Détection des joueurs
        player_counter = Counter()

        def is_likely_player(entity):
            return (entity.label_ == "PERSON" and
                    entity.text.istitle() and
                    len(entity.text.split()) <= 3)

        for ent in doc.ents:
            if is_likely_player(ent):
                player_counter[ent.text] += 1

        confirmed_players = [player for player, count in player_counter.items() if count >= 2]

        # Détection des actions
        action_patterns = {
            'goal': r'\b(goal|score[sd]|net[sd]|finish(?:es|ed)|converted)\b',
            'penalty': r'\b(penalty|penalties|spot kick|PK)\b',
            'yellow card': r'\b(yellow card|booked|cautioned|first warning)\b',
            'red card': r'\b(red card|sent off|ejection|dismissed)\b',
            'substitute': r'\b(sub(?:s|stitution)|replaced|coming on|brought on)\b',
            'assist': r'\b(assist(?:sd)?|setup|provided the cross|laid off)\b',
            'pass': r'\b(pass(?:es|ed)|cross(?:es|ed)|through ball|long ball|short pass)\b',
            'attack': r'\b(attack|breakaway|counterattack|going forward)\b',
            'offside': r'\b(offside|in an offside position)\b',
            'corner': r'\b(corner kick|flag kick)\b',
            'free kick': r'\b(free kick|direct free kick|set piece)\b',
            'shot': r'\b(shot|strike[sd]|hit[sd]|attempt[sd]|volley|half volley)\b',
            'kickoff': r'\b(kick[-\s]?off|start(?:ed|ing) the match)\b',
            'foul': r'\b(foul[sd]?|illegal challenge|reckless tackle)\b',
            'header': r'\b(header|headed|with the head)\b',
            'save': r'\b(save[sd]?|block[sd]|denie[sd]|stop[sd]|parr(?:y|ied))\b',
            'tackle': r'\b(tackle[sd]?|intercept[sd]?|won the ball)\b'
        }

        events = []
        previous_player_action = {}

        for sent in doc.sents:
            current_players = [ent.text for ent in sent.ents if ent.text in confirmed_players]

            for action, pattern in action_patterns.items():
                if re.search(pattern, sent.text, re.IGNORECASE):
                    for player in current_players:
                        if (player, action) not in previous_player_action:
                            events.append({
                                'player': player,
                                'action': action
                            })
                            previous_player_action[(player, action)] = True

        # Élimination des doublons
        unique_events = []
        seen = set()

        for event in events:
            key = (event['player'], event['action'])
            if key not in seen:
                seen.add(key)
                unique_events.append(event)

        return unique_events

    def generate_summary(self, text):
        """Génère un résumé du texte"""
        def chunk_text(text, max_tokens=512):
            tokens = self.tokenizer.encode(text, truncation=False, return_tensors="pt")[0]
            chunks = []
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i:i+max_tokens]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
            return chunks

        chunks = chunk_text(text)
        summaries = []
        
        for chunk in chunks:
            try:
                summary = self.summarizer(chunk, max_length=300, min_length=50, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Erreur lors du résumé : {e}")
                continue

        if not summaries:
            return "Aucun résumé généré."

        final_summary = " ".join(summaries)
        
        try:
            final_resumed_summary = self.summarizer(
                final_summary,
                max_length=1200,
                min_length=300,
                do_sample=False
            )
            return self.format_bullet_points(final_resumed_summary[0]['summary_text'])
        except Exception as e:
            print(f"Erreur lors de la génération du résumé final: {e}")
            return self.format_bullet_points(final_summary)

    def format_bullet_points(self, text):
        """Formate le texte en liste à puces"""
        sentences = []
        for s in text.split('.'):
            s = s.strip()
            if s:
                if not s.endswith('.'):
                    s += '.'
                sentences.append(f"- {s}")
        return "\n".join(sentences)

    def translate_summary(self, summary):
        """Traduit le résumé en français"""
        try:
            translated = self.translator.translate(
                " ".join(s.strip('- ') for s in summary.split('\n')), 
                src="en", 
                dest="fr"
            ).text
            return self.format_bullet_points(translated)
        except Exception as e:
            print(f"Erreur de traduction : {e}")
            return "Traduction non disponible"