"""Action surface for GraphForge.

Public API:

    from graphforge.actions import dispatch, ActionResult
    from graphforge.actions.schema import Action, AddNode, ...
    from graphforge.actions.errors import ActionError

See PROPOSAL.md §4 for the full action vocabulary.
"""

from graphforge.actions.dispatcher import ActionResult, dispatch
from graphforge.actions.errors import ActionError

__all__ = ["ActionError", "ActionResult", "dispatch"]
