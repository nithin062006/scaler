---
title: GraphForge OpenEnv
emoji: 🧱
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# GraphForge — OpenEnv server

Live deployment of the GraphForge environment for the Meta PyTorch OpenEnv
Hackathon. The server hosts the OpenEnv-compliant `/reset`, `/step`, `/state`
endpoints over HTTP. Anything that speaks the OpenEnv client protocol (or
plain JSON) can drive episodes.

See the main project repo for the architecture overview, training notebook,
plots, and writeup.

## Endpoints

```
POST /reset             → GraphForgeObservation
POST /step  { ... }     → { observation, reward, done }
GET  /state             → GraphForgeState
GET  /healthz
```

## Quick smoke test

```bash
EID=$(curl -s -X POST $SPACE_URL/reset | python3 -c "import sys,json; print(json.load(sys.stdin)['episode_id'])")
curl -s -X POST $SPACE_URL/step -H 'content-type: application/json' \
  -d '{"kind": "add_module", "payload": {"name": "validators", "responsibility": "validation"}}' \
  | python3 -m json.tool
```
