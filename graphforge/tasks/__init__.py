"""Task bank and variant generator.

Tier-0 ships one hand-written task. Tier-1+ tasks and parametric variant
generation are TODO. See PROPOSAL.md §2.1, §2.3 for the full design.
"""

from graphforge.tasks.bank import default_task, get_task, list_tasks
from graphforge.tasks.schema import Task

__all__ = ["Task", "default_task", "get_task", "list_tasks"]
