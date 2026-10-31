"""Prompts for the rules agent (used only when an LLM is configured)."""

GRADE_SYSTEM = (
    "You judge whether a passage from EU law helps answer a question about the EU Deforestation "
    "Regulation. Reply with exactly one word: yes or no."
)
GRADE_USER = "Question: {question}\n\nPassage ({ref}):\n{text}\n\nDoes this passage help answer the question?"

REWRITE_SYSTEM = (
    "You rewrite search queries for a keyword search over the EU Deforestation Regulation (EUDR). "
    "Use the regulation's own vocabulary (e.g. 'agricultural use', 'forest degradation', "
    "'geolocation', 'plot of land', 'deforestation-free', 'competent authorities'). "
    "Reply with the new query only, at most 20 words."
)
REWRITE_USER = "Original question: {question}\nPrevious query: {query}\nThe passages found did not answer it. New query:"

ANSWER_SYSTEM = (
    "You answer questions about the EU Deforestation Regulation using ONLY the numbered passages given. "
    "Cite every claim with the passage label in square brackets, e.g. [P1]. "
    "Quote short phrases where wording matters. Do not give legal advice. "
    "If a passage gives a definition or rule that bears on the question, state it, apply it only as far "
    "as its wording allows, and say what it leaves open. "
    "Only if no passage bears on the question, reply exactly: NO_RULE_FOUND"
)
ANSWER_USER = "Question: {question}\n\nPassages:\n{passages}\n\nAnswer in at most 120 words."
