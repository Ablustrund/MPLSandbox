#!/bin/bash

URL="http://localhost:6000/run-app"

curl -X POST $URL \
-H "Content-Type: application/json" \
-d @data.json