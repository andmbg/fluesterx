# A transcription and embedding companion to the Glasspod app

This is the external API that helps transcribe audio episodes for the Glasspod app as showcased on
[this site](somethingsomethingdata.eu/glasspod).

It basically just wraps and slightly customizes the WhisperX transcription service.

This container image can easily be loaded to a vast.ai server. Experience has been made with using an RTX 4090 GPU, which typically yields 15-23x transcription speed. Given current prices at vast.ai, for instance, $10 buy you around 600-700 hours worth of transcribing, done in 30 hours.

## Mandatory ENV settings

To start a container with this image, it is necessary to set two environment variables:

- `API_TOKEN`: Set an arbitrary but secure random token here and in your Podology instance. This is for your API to be accessible only for you.
- `HF_TOKEN`: The huggingface token necessary to access a model for diarization. At the moment, it appears to be policy that this hoop is necessary for users of the model to jump through. If not already done, get a huggingface account, go to [the relevant model](https://huggingface.co/pyannote/speaker-diarization) and leave your contact data there. It is free.

# Getting Started

## Locally

To run this API locally, pull the repo and build the image, or pull the docker image associated with this repo.

Then start a container:

```bash
docker run -e API_TOKEN=yourtoken -e HF_TOKEN=tokenFromHuggingface -p 19000:19000 -d andmbg/fluesterx
```

Now set this host (on which this container runs) and port in the docker-compose of podology.

## On vast.ai

vast.ai is one of several places at which cheap GPU computing capacity can be rented. Upon paying into your account, select an instance, e.g., with a RTX 4090, 4080 or comparable GPU. Since transcription can take some minutes, an On-Demand instance is better (although somewhat more expensive) than an Interruptible instance. Select your image (in this case the one in this repo), set the mandatory environment variables and the open port. Wait till the instance is running, copy its IP and port and paste those in the Glasspod docker-compose configuration .env:

- TRANSCRIBER_URL_PORT
- API_TOKEN
