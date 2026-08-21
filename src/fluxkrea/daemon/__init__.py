"""The per-node daemon: HTTP API, task runner, job queue.

One of these runs on each GPU node and owns everything - the dataset
folders, the queue, the training subprocesses. Clients hold no state worth
losing: close the laptop and training continues (doc 06).
"""

from .app import API, create_app, serve
from .state import State

__all__ = ["API", "State", "create_app", "serve"]
