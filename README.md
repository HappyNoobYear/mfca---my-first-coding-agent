# mfca---my-first-coding-agent
Repo for the coding agent project for advanced media technologies

Todos:
- get access to an AI model (free tier OpenAi or local gemma 4)
-   openAi no longer has a free API tier
-   local gemma 4
- create agent loop
-   user input --> model inference 
-   a) model inference --> tool calls --> model inference
-     agent executes tool and appends returns to original input 
-     agent führt tool Aufruf aus und hängt Ergebnis an die ursprüngliche Anfrage an
-   b) agent response (end  of loop)
- add tools:
-   read code
-   modify code
-     mvp: copy whole code
-     nice to have: modify functions or lines of code
-   read directory
-   execute shell commandos
-     mvp: yolo
-     nice to have: sandbox or permission check from user
