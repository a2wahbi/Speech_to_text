import streamlit as st
import whisper
import os
import tempfile
import speech_recognition as sr
from scipy.io.wavfile import write

# Titre de l'application
st.title("Speech-to-Text avec Whisper")
st.write("Cliquez sur le bouton pour enregistrer votre voix avec le micro et obtenir une transcription.")

# Charger le modèle Whisper avec mise en cache
@st.cache_resource
def load_model():
    return whisper.load_model("small")  # Modèles possibles : tiny, base, small, medium, large

model = load_model()

# Fonction pour enregistrer via le micro et détecter le silence
def record_audio_with_silence_detection(filename):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Enregistrement en cours... Parlez maintenant.")
        try:
            # Ajuster le bruit ambiant pour une meilleure détection
            recognizer.adjust_for_ambient_noise(source)
            # Écouter jusqu'au silence
            audio_data = recognizer.listen(source)
            st.success("Enregistrement terminé !")

            # Sauvegarder l'audio
            with open(filename, "wb") as f:
                f.write(audio_data.get_wav_data())
        except Exception as e:
            st.error(f"Erreur pendant l'enregistrement : {e}")
            return None

# Bouton pour utiliser le micro
if st.button("🎤 Enregistrer avec le micro"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        audio_path = tmpfile.name
    try:
        # Enregistrement
        record_audio_with_silence_detection(audio_path)

        # Charger et jouer l'audio enregistré
        st.audio(audio_path, format="audio/wav", start_time=0)

        # Transcrire l'audio
        st.write("Transcription en cours...")
        result = model.transcribe(audio_path)

        # Afficher la transcription
        st.success("Transcription terminée !")
        st.text_area("Transcription", value=result["text"], height=300)

        # Option de téléchargement de la transcription
        st.download_button(
            label="Télécharger la transcription",
            data=result["text"],
            file_name="transcription.txt",
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"Une erreur est survenue pendant l'enregistrement ou la transcription : {e}")
    finally:
        # Supprimer le fichier temporaire
        if os.path.exists(audio_path):
            os.remove(audio_path)
