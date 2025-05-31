#!/bin/bash

# Lancer le serveur Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
