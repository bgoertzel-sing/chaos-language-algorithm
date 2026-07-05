"""JSON persistence and edit-log replay for CLA grammar states.

Provides:
- ``state_to_json`` / ``state_from_json`` — round-trip GrammarState to JSON-dict.
- ``replay_edits`` — reconstruct a GrammarState by replaying its edit log.
- ``grammar_to_text`` — stable, human-readable text rendering of a grammar.
- ``CLAModel.to_json()`` convenience on the fitted model.

Design constraints:
- Pure stdlib, no external deps.
- Deterministic ordering: sorted keys everywhere.
- Round-trip preserves exact reconstruction (symbols == expand(parse, grammar)).
- Edit-log replay is deterministic given the same initial symbols and seed.
"""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Dict, List, Tuple

from .core.types import (
    Category,
    CategoryOccurrence,
    Corpus,
    Edit,
    Grammar,
    GrammarState,
    ParseEntry,
    Production,
    Score,
    Token,
    expand_parse,
)


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def _token_to_dict(tok: Token) -> Dict[str, str]:
    return {"value": tok.value, "kind": tok.kind}


def _token_from_dict(d: Dict[str, str]) -> Token:
    return Token(value=d["value"], kind=d.get("kind", "base"))


def _entry_to_dict(entry: ParseEntry) -> Dict[str, Any]:
    if isinstance(entry, CategoryOccurrence):
        return {
            "type": "category_occurrence",
            "category": entry.category,
            "member": _token_to_dict(entry.member),
        }
    return {"type": "token", "token": _token_to_dict(entry)}


def _entry_from_dict(d: Dict[str, Any]) -> ParseEntry:
    if d["type"] == "category_occurrence":
        return CategoryOccurrence(
            category=d["category"],
            member=_token_from_dict(d["member"]),
        )
    return _token_from_dict(d["token"])


def _production_to_dict(prod: Production) -> Dict[str, Any]:
    return {
        "lhs": _token_to_dict(prod.lhs),
        "rhs": [_entry_to_dict(e) for e in prod.rhs],
    }


def _production_from_dict(d: Dict[str, Any]) -> Production:
    return Production(
        lhs=_token_from_dict(d["lhs"]),
        rhs=tuple(_entry_from_dict(e) for e in d["rhs"]),
    )


def _category_to_dict(cat: Category) -> Dict[str, Any]:
    return {
        "name": cat.name,
        "members": [_token_to_dict(m) for m in sorted(cat.members, key=lambda t: (t.kind, t.value))],
    }


def _category_from_dict(d: Dict[str, Any]) -> Category:
    return Category(
        name=d["name"],
        members=frozenset(_token_from_dict(m) for m in d["members"]),
    )


def _score_to_dict(score: Score) -> Dict[str, Any]:
    return {
        "model_bits": score.model_bits,
        "data_bits": score.data_bits,
        "diagnostics": dict(score.diagnostics),
    }


def _score_from_dict(d: Dict[str, Any]) -> Score:
    return Score(
        model_bits=d["model_bits"],
        data_bits=d["data_bits"],
        diagnostics=dict(d.get("diagnostics", {})),
    )


def _edit_to_dict(edit: Edit) -> Dict[str, Any]:
    # JSON-safe payload: convert tuples to lists, frozensets to sorted lists
    payload = {}
    for k, v in edit.payload.items():
        if isinstance(v, tuple):
            payload[k] = list(v)
        elif isinstance(v, frozenset):
            payload[k] = sorted(v)
        else:
            payload[k] = v
    return {"kind": edit.kind, "payload": payload}


def _edit_from_dict(d: Dict[str, Any]) -> Edit:
    """Deserialize an Edit, normalizing lists back to tuples for stable equality."""
    payload = {}
    for k, v in d.get("payload", {}).items():
        if isinstance(v, list):
            payload[k] = tuple(v)
        else:
            payload[k] = v
    return Edit(kind=d["kind"], payload=payload)


def state_to_json(state: GrammarState, *, indent: int | None = 2) -> str:
    """Serialize a GrammarState to a JSON string."""
    return json.dumps(state_to_dict(state), indent=indent, sort_keys=True)


def state_to_dict(state: GrammarState) -> Dict[str, Any]:
    """Serialize a GrammarState to a JSON-compatible dict."""
    return {
        "corpus": {
            "symbols": [_token_to_dict(s) for s in state.corpus.symbols],
            "metadata": dict(state.corpus.metadata),
        },
        "parse": [_entry_to_dict(e) for e in state.parse],
        "grammar": {
            "productions": [
                _production_to_dict(p)
                for _, p in sorted(state.grammar.productions.items(), key=lambda kv: (kv[0].kind, kv[0].value))
            ],
            "categories": [
                _category_to_dict(c)
                for _, c in sorted(state.grammar.categories.items(), key=lambda kv: kv[0])
            ],
        },
        "score": _score_to_dict(state.score),
        "history": list(state.history),
        "edit_log": [_edit_to_dict(e) for e in state.edit_log],
    }


def state_from_dict(d: Dict[str, Any]) -> GrammarState:
    """Deserialize a GrammarState from a JSON-compatible dict."""
    corpus = Corpus(
        symbols=tuple(_token_from_dict(s) for s in d["corpus"]["symbols"]),
        metadata=dict(d["corpus"].get("metadata", {})),
    )
    parse = tuple(_entry_from_dict(e) for e in d["parse"])
    productions = {}
    for pd in d["grammar"]["productions"]:
        prod = _production_from_dict(pd)
        productions[prod.lhs] = prod
    categories = {}
    for cd in d["grammar"]["categories"]:
        cat = _category_from_dict(cd)
        categories[cat.name] = cat
    grammar = Grammar(productions, categories)
    score = _score_from_dict(d["score"])
    history = tuple(d.get("history", []))
    edit_log = tuple(_edit_from_dict(e) for e in d.get("edit_log", []))
    return GrammarState(
        corpus=corpus,
        parse=parse,
        grammar=grammar,
        score=score,
        history=history,
        edit_log=edit_log,
    )


def state_from_json(s: str) -> GrammarState:
    """Deserialize a GrammarState from a JSON string."""
    return state_from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Edit-log replay
# ---------------------------------------------------------------------------

def _proposal_from_edit(edit: Edit) -> Tuple[str, Any]:
    """Convert an Edit back into a proposal-like description.

    Returns (kind, payload_dict) where kind is "chunk" or "category".
    Raises ValueError for unknown edit kinds.
    """
    kind = edit.kind
    payload = edit.payload
    if kind == "AddChunkRule":
        # Reconstruct ChunkProposal fields
        name = payload["name"]
        # block entries are stored as strings; we need to reconstruct parse entries
        # But for replay we apply edits to a fresh state, so we need the block
        # as it existed at that point. The edit log stores block as string repr.
        # For replay, we return the raw fields and let the caller reconstruct.
        block_strs = payload.get("block", [])
        occurrences = tuple(payload.get("occurrences", []))
        return ("chunk", {"name": name, "block_strs": block_strs, "occurrences": occurrences})
    elif kind == "AddCategory":
        name = payload["name"]
        members = tuple(payload.get("members", []))
        positions = tuple(payload.get("positions", []))
        return ("category", {"name": name, "members": members, "positions": positions})
    elif kind == "DeleteUnusedRule":
        return ("prune", {"name": payload["name"]})
    else:
        raise ValueError(f"Unknown edit kind for replay: {kind!r}")


def replay_edits(
    symbols: List[str] | Tuple[str, ...],
    edit_log: Tuple[Edit, ...],
    *,
    seed: int = 0,
) -> GrammarState:
    """Reconstruct a GrammarState by replaying its edit log from scratch.

    This applies each edit in order to a fresh initial state.  The edit log
    records enough information to deterministically reconstruct the grammar
    without re-running the mining/scoring loop.

    Parameters
    ----------
    symbols : the original input symbol stream
    edit_log : the sequence of edits to replay
    seed : stored for reference (replay itself is deterministic from edits)
    """
    state = GrammarState.initial(symbols)

    for edit in edit_log:
        kind, info = _proposal_from_edit(edit)

        if kind == "chunk":
            # Reconstruct the block from the current parse at the recorded positions
            occurrences = info["occurrences"]
            block_len = len(info["block_strs"])
            # Find the block at the first occurrence position
            if not occurrences:
                continue
            first_pos = occurrences[0]
            block = state.parse[first_pos : first_pos + block_len]
            # Verify the block matches the recorded strings
            block_strs = info["block_strs"]
            actual_strs = tuple(str(e) for e in block)
            if actual_strs != tuple(block_strs):
                # The parse has changed since the edit was recorded (e.g. due to
                # a prior prune or chunk). Try to find the block by scanning.
                found = False
                for i in range(len(state.parse) - block_len + 1):
                    if tuple(str(e) for e in state.parse[i : i + block_len]) == tuple(block_strs):
                        block = state.parse[i : i + block_len]
                        # Recalculate non-overlapping occurrences
                        occ_list = []
                        next_free = 0
                        for j in range(len(state.parse) - block_len + 1):
                            if j >= next_free and tuple(str(e) for e in state.parse[j : j + block_len]) == tuple(block_strs):
                                occ_list.append(j)
                                next_free = j + block_len
                        occurrences = tuple(occ_list)
                        found = True
                        break
                if not found:
                    # Block no longer exists in the parse; skip this edit
                    continue

            # Build a ChunkProposal and apply it
            from .core.types import ChunkProposal
            proposal = ChunkProposal(
                block=tuple(block),
                occurrences=occurrences,
                name=info["name"],
            )
            from .core.edits import EditApplier
            applier = EditApplier()
            state = applier.apply_chunk(state, proposal)

        elif kind == "category":
            from .core.types import CategoryProposal, Token as CT
            # Reconstruct member tokens from their string values
            # We need to find actual Token objects in the current parse
            member_strs = set(info["members"])
            member_tokens = set()
            for entry in state.parse:
                if isinstance(entry, CT) and entry.value in member_strs:
                    member_tokens.add(entry)
            # Also check existing grammar productions for matching tokens
            for prod in state.grammar.productions:
                if prod.value in member_strs:
                    member_tokens.add(prod)

            positions = info["positions"]
            # Filter positions to those where the entry is actually a member token
            valid_positions = tuple(
                p for p in positions
                if p < len(state.parse)
                and isinstance(state.parse[p], CT)
                and state.parse[p] in member_tokens
            )

            if len(member_tokens) >= 2 and valid_positions:
                from .core.types import CategoryProposal
                proposal = CategoryProposal(
                    members=frozenset(member_tokens),
                    name=info["name"],
                    positions=valid_positions,
                )
                from .core.edits import EditApplier
                applier = EditApplier()
                state = applier.apply_category(state, proposal)

        elif kind == "prune":
            # Prune is handled automatically by apply_chunk when uses < min,
            # but explicit prune edits need to be applied too
            from .core.edits import EditApplier
            applier = EditApplier()
            # Find the rule and prune it
            name = info["name"]
            target = None
            for tok in state.grammar.productions:
                if tok.value == name:
                    target = tok
                    break
            if target is not None:
                # Force prune by calling prune_dead_rules which handles expansion
                # We need to remove the specific rule. Since prune_dead_rules
                # removes ALL unused rules, we first check if it's unused.
                uses = applier.rule_use_counts(state)
                if uses.get(target, 0) < applier.min_rule_uses:
                    state = applier.prune_dead_rules(state)

    return state


# ---------------------------------------------------------------------------
# Stable text rendering
# ---------------------------------------------------------------------------

def grammar_to_text(state: GrammarState) -> str:
    """Produce a stable, human-readable text rendering of the grammar.

    Format:
        == Corpus ==
        <symbol0> <symbol1> ...

        == Parse ==
        <entry0> <entry1> ...

        == Productions ==
        N0 -> a b c
        N1 -> N0 d

        == Categories ==
        M0 = {a, b}
        M1 = {c, d}

        == Score ==
        model=<float> data=<float> total=<float>

        == History ==
        chunk N0 -> a b c
        category M0 -> {a,b}
        prune N0

        == Edit Log ==
        AddChunkRule N0
        AddCategory M0
        DeleteUnusedRule N0
    """
    lines: List[str] = []

    lines.append("== Corpus ==")
    lines.append(" ".join(str(s) for s in state.corpus.symbols))
    lines.append("")

    lines.append("== Parse ==")
    lines.append(" ".join(str(e) for e in state.parse))
    lines.append("")

    lines.append("== Productions ==")
    for lhs in sorted(state.grammar.productions, key=lambda t: (t.kind, t.value)):
        prod = state.grammar.productions[lhs]
        rhs_str = " ".join(str(e) for e in prod.rhs)
        lines.append(f"{lhs} -> {rhs_str}")
    if not state.grammar.productions:
        lines.append("(none)")
    lines.append("")

    lines.append("== Categories ==")
    for name in sorted(state.grammar.categories):
        cat = state.grammar.categories[name]
        members_str = ", ".join(
            str(t) for t in sorted(cat.members, key=lambda t: (t.kind, t.value))
        )
        lines.append(f"{name} = {{{members_str}}}")
    if not state.grammar.categories:
        lines.append("(none)")
    lines.append("")

    lines.append("== Score ==")
    s = state.score
    lines.append(f"model={s.model_bits:.6f} data={s.data_bits:.6f} total={s.total:.6f}")
    lines.append("")

    lines.append("== History ==")
    for h in state.history:
        lines.append(h)
    if not state.history:
        lines.append("(none)")
    lines.append("")

    lines.append("== Edit Log ==")
    for e in state.edit_log:
        lines.append(f"{e.kind} {e.payload.get('name', '')}".rstrip())
    if not state.edit_log:
        lines.append("(none)")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Convenience: CLAModel persistence
# ---------------------------------------------------------------------------

def model_to_json(model, *, seed: int | None = None, indent: int | None = 2) -> str:
    """Serialize a CLAModel to JSON, including seed and config metadata.

    The seed is optional since CLAModel doesn't store it; pass it explicitly
    for full reproducibility records.
    """
    d = state_to_dict(model.state)
    d["__model_meta__"] = {
        "seed": seed,
        "scorer": type(model.scorer).__name__,
    }
    return json.dumps(d, indent=indent, sort_keys=True)
