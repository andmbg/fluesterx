# A transcription and embedding companion to the Podology app

This is the external API that helps transcribe audio episodes and compute vector embeddings of text chunks for the Podology app as showcased on
[this site](somethingsomethingdata.eu).

It basically just wraps the WhisperX transcription service and adds an endpoint for SentenceEmbeddings. Things that could functionally also reside on the machine running the app, but that you would commonly use before making the app productive and public in order to do all the transcribing and embedding on a GPU-powered machine.

This container image can easily be loaded to a vast.ai server. Experience has been made with using an RTX 4090 GPU, which typically yields 20-23x transcription speed. Given current prices at vast.ai, for instance, $10 buy you around 600-700 hours worth of transcribing, done in 30 hours.

## Important: Model choice

Note the importance of selecting the same embedder model within this API and within your instance of Podology. There, an embedder is instantiated as well, which serves to embed user prompts. You do want both prompt and transcript embeddings to reside in the same vector space.

## Mandatory ENV settings

To start a container with this image, it is necessary to set two environment variables:

- `API_TOKEN`: Set an arbitrary but secure random token here and in your Podology instance. This is for your API to be accessible only for you.
- `HF_TOKEN`: The huggingface token necessary to access a model for diarization. At the moment, it appears to be policy that this hoop is necessary for users of the model to jump through. If not already done, get a huggingface account, go to [the relevant model](https://huggingface.co/pyannote/speaker-diarization) and leave your contact data there. It is free.



