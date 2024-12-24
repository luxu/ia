from pathlib import Path

import assemblyai as aai
from decouple import config

# Configurar API Key
aai.settings.api_key = config('API_ASSEMBLYAI')


desktop = Path.joinpath(Path.home(), Path("Desktop"))
video_output = desktop / 'bloco1.mp4'

# Fazer upload do arquivo MP4
transcriber = aai.Transcriber()
transcript = transcriber.transcribe(str(video_output))

# Exibir transcrição
print(transcript["text"])
