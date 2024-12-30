import streamlit as st
import whisper
import os
import tempfile

# Titre de l'application
st.title("Speech-to-Text avec Whisper")
st.write("Chargez un fichier audio pour obtenir une transcription.")

# Charger le modèle Whisper avec mise en cache
@st.cache_resource
def load_model():
    return whisper.load_model("small", device="cpu")  # Modèles possibles : tiny, base, small, medium, large

model = load_model()

# Upload d'un fichier audio
audio_file = st.file_uploader("Chargez un fichier audio", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        audio_path = tmpfile.name
        try:
            # Sauvegarder le fichier uploadé temporairement
            tmpfile.write(audio_file.read())

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
            st.error(f"Une erreur est survenue pendant la transcription : {e}")
        finally:
            # Supprimer le fichier temporaire
            if os.path.exists(audio_path):
                os.remove(audio_path)
