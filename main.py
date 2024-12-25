import streamlit as st
import whisper
import os
import tempfile

# Titre de l'application
st.title("Speech-to-Text avec Whisper")
st.write("Utilisez votre micro pour enregistrer votre voix et obtenir une transcription.")

# Charger le modèle Whisper avec mise en cache
@st.cache_resource
def load_model():
    return whisper.load_model("small").cpu  # Modèles possibles : tiny, base, small, medium, large

model = load_model()

# Enregistrement via st.audio_input
audio_data = st.audio_input("🎤 Cliquez pour enregistrer votre voix")

if audio_data is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        audio_path = tmpfile.name
        try:
            # Sauvegarder l'audio temporairement
            tmpfile.write(audio_data.read())

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
