"""FastAPI OpenEnv server.

Endpoints (PROPOSAL.md §6.1):

  POST /reset    -> create a fresh episode, return initial observation
  POST /step     -> apply an Action, return (observation, reward, done, info)
  GET  /state    -> snapshot the current episode state for debugging
  POST /close    -> tear down the episode

The server is a thin shell: it owns episode state (graph, task spec,
action history, token counter, turn counter, materialization cache) and
delegates the work to the dispatcher, reward engine, and validators.

The training-side OpenEnv client calls this over HTTP at localhost:8000.
"""

from graphforge.server.app import app

__all__ = ["app"]
