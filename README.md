Simple python video inference streamer, also some other random inference things

## Getting started

```
pip install -r requirements.txt
```

have a webcam available and connected

## Running

To run the server, run the following command
```
python3 server.py
```
Start the client by running the following command
```
python3 client.py
```
# Exiting

To exit the client, press 'q' on the keyboard where the video is playing

To exit the server, in the terminal running the server, press 'ctrl+c'
There are signal handlers that will close the server gracefully

# OWL-ViT

There are a few implementations of the [Owl Vit ](https://huggingface.co/google/owlvit-base-patch32) in the owl_clip folder. Including one with a basic SORT tracker.