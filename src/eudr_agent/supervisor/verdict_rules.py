"""Fixed verdict table (design doc, section 4). Planned interface; not implemented yet (owner: Alvin).

Rules, in order:
  1. plot inside a protected area                                        -> high_risk
  2. no forest in 2020 (maps agree) or no loss after 2020                -> clear
  3. small scattered loss inside land farmed in 2020, stable under +-30 m -> clear
  4. datasets disagree / loss at edge / boundary shift changes result    -> needs_evidence
  5. forest in 2020 by several maps + large block cleared after 2020     -> high_risk
  default                                                                -> needs_evidence
"""

from __future__ import annotations

from eudr_agent.state import CaseFile, Verdict


def decide(case: CaseFile) -> Verdict | None:
    raise NotImplementedError
