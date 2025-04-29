import warnings
import os
import re
from collections import Counter
from googletrans import Translator
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import spacy
import whisper
from moviepy import VideoFileClip
from transformers import pipeline, BartTokenizer

# Configuration
warnings.filterwarnings("ignore")
TEMP_AUDIO = "temp_audio.wav"
CHUNK_FOLDER = "audio_chunks"
WHISPER_MODEL = "small"  # tiny/base/small/medium/large

class FootballMatchAnalyzer:
    def __init__(self):
        """Initialise tous les modèles et outils"""
        print("⚙️ Chargement des modèles...")
        self.whisper_model = whisper.load_model(WHISPER_MODEL)
        self.nlp = spacy.load("en_core_web_sm")
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn"
        )
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.translator = Translator()
        os.makedirs(CHUNK_FOLDER, exist_ok=True)

    def extract_audio(self, video_path):
        """Extrait l'audio de la vidéo"""
        print(f"🔊 Extraction audio depuis {video_path}...")
        try:
            with VideoFileClip(video_path) as video:
                video.audio.write_audiofile(
                    TEMP_AUDIO,
                    codec='pcm_s16le',
                    fps=16000,
                )
            return TEMP_AUDIO
        except Exception as e:
            raise RuntimeError(f"Échec extraction audio: {str(e)}")

    def split_audio(self, audio_path):
        """Découpe l'audio en segments intelligents"""
        print("✂️ Découpage audio...")
        try:
            audio = AudioSegment.from_wav(audio_path)
            non_silent_ranges = detect_nonsilent(
                audio,
                min_silence_len=1000,
                silence_thresh=-40
            )
            
            chunks = []
            for i, (start, end) in enumerate(non_silent_ranges):
                chunk_path = os.path.join(CHUNK_FOLDER, f"chunk_{i}.wav")
                audio[start:end].export(chunk_path, format="wav")
                chunks.append(chunk_path)
                
            return chunks if chunks else [audio_path]
        except Exception as e:
            print(f"⚠️ Erreur découpage: {str(e)}")
            return [audio_path]

    def transcribe_audio(self, chunk_path):
        """Transcrit un segment audio en texte"""
        try:
            result = self.whisper_model.transcribe(
                chunk_path,
                language="en",
                initial_prompt=(
                    "Football commentary with players like Salah, Kane. "
                    "Focus on goals, cards, substitutions. "
                    "Ignore crowd noise."
                ),
                temperature=0.2
            )
            return result["text"].strip()
        except Exception as e:
            print(f"⚠️ Erreur transcription: {str(e)}")
            return ""

    def analyze_text(self, text):
        """Analyse le texte pour détecter les événements"""
        print("⚽ Analyse des événements...")
        doc = self.nlp(text)
        events = []
        
        # Détection des joueurs
        players = set()
        for ent in doc.ents:
            if (ent.label_ == "PERSON" and ent.text.istitle() 
                and len(ent.text.split()) <= 3):
                players.add(ent.text)
        
        # Détection des actions
        action_patterns = {
            'goal': r'\b(goal|scored|netted)\b',
            'card': r'\b(yellow card|red card|booked)\b',
            'substitution': r'\b(sub|substitution|replaced)\b',
            'foul': r'\b(foul|handball|penalty)\b'
        }
        
        for sent in doc.sents:
            for action, pattern in action_patterns.items():
                if re.search(pattern, sent.text, re.IGNORECASE):
                    for player in players:
                        if player in sent.text:
                            events.append({
                                'player': player,
                                'action': action,
                                'details': sent.text.strip()
                            })
        
        # Élimination des doublons
        unique_events = []
        seen = set()
        for ev in events:
            key = (ev['player'], ev['action'], ev['details'][:50])
            if key not in seen:
                seen.add(key)
                unique_events.append(ev)
                
        return unique_events

    def generate_summary(self, text):
        """Génère un résumé structuré"""
        print("📝 Création du résumé...")
        try:
            # Découpage en chunks
            chunks = self._chunk_text(text)
            stage1_summaries = []
            
            for chunk in chunks:
                summary = self.summarizer(
                    chunk,
                    max_length=150,
                    min_length=30,
                    do_sample=False
                )
                stage1_summaries.append(summary[0]['summary_text'])
            
            # Résumé final
            final_summary = self.summarizer(
                " ".join(stage1_summaries),
                max_length=300,
                min_length=100,
                do_sample=False
            )
            
            return self._format_bullet_points(final_summary[0]['summary_text'])
        except Exception as e:
            print(f"⚠️ Erreur résumé: {str(e)}")
            return self._format_bullet_points(text[:500])

    def translate_summary(self, summary):
        """Traduit le résumé en français"""
        print("🌍 Traduction...")
        try:
            # Traduction du texte sans puces
            clean_text = ' '.join([line[2:] for line in summary.split('\n') if line])
            translated = self.translator.translate(
                clean_text,
                src='en',
                dest='fr',
                timeout=10
            ).text
            
            # Remise en forme
            return self._format_bullet_points(translated)
        except Exception as e:
            print(f"⚠️ Erreur traduction: {str(e)}")
            return self._format_fallback_translation(summary)

    def _chunk_text(self, text, max_tokens=1000):
        """Découpe le texte en morceaux"""
        paragraphs = [p for p in text.split('\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_len = len(self.tokenizer.tokenize(para))
            if current_length + para_len > max_tokens:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(para)
            current_length += para_len
            
        if current_chunk:
            chunks.append("\n".join(current_chunk))
            
        return chunks

    def _format_bullet_points(self, text):
        """Formate en liste à puces"""
        sentences = [
            f"- {s.strip().capitalize()}" 
            for s in re.split(r'[.!?]', text) 
            if s.strip()
        ]
        return "\n".join(sentences[:15])

    def _format_fallback_translation(self, text):
        """Formatage de secours"""
        return text.replace("goal", "but").replace("penalty", "pénalty")

    def cleanup(self):
        """Nettoyage des fichiers temporaires"""
        try:
            if os.path.exists(TEMP_AUDIO):
                os.remove(TEMP_AUDIO)
            if os.path.exists(CHUNK_FOLDER):
                for f in os.listdir(CHUNK_FOLDER):
                    os.remove(os.path.join(CHUNK_FOLDER, f))
                os.rmdir(CHUNK_FOLDER)
        except:
            pass

    def analyze_video(self, video_path):
        """Pipeline complet d'analyse"""
        try:
            # 1. Extraction audio
            audio_file = self.extract_audio(video_path)
            
            # 2. Découpage audio
            chunks = self.split_audio(audio_file)
            print(f"🔈 {len(chunks)} segments créés")
            
            # 3. Transcription
            print("📜 Transcription...")
            transcripts = [self.transcribe_audio(chunk) for chunk in chunks]
            full_text = "\n".join([t for t in transcripts if t.strip()])
            
            if not full_text.strip():
                raise ValueError("Aucun contenu transcrit")
            
            # 4. Analyse
            events = self.analyze_text(full_text)
            summary = self.generate_summary(full_text)
            french_summary = self.translate_summary(summary)
            
            return {
                'transcription': full_text,
                'events': events,
                'english_summary': summary,
                'french_summary': french_summary
            }
            
        finally:
            self.cleanup()

# Exemple d'utilisation
if __name__ == "__main__":
    analyzer = FootballMatchAnalyzer()
    results = analyzer.analyze_video("match.mp4")
    
    print("\n" + "="*50)
    print("📋 RESULTATS COMPLETS")
    print("="*50)
    
    print("\n⚽ ÉVÉNEMENTS CLÉS:")
    for event in results['events'][:10]:
        print(f"- {event['player']}: {event['action']}")
    
    print("\n🇬🇧 RÉSUMÉ ANGLAIS:")
    print(results['english_summary'])
    
    print("\n🇫🇷 RÉSUMÉ FRANÇAIS:")
    print(results['french_summary'])
    
    print("\n✅ Analyse terminée!")