from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from pagestorm.artifacts_14b import (
    split_book_plan_components,
    split_first_chapter_plan_components,
)
from pagestorm.models import ChapterTextCompletionPrefix, StoryBundle
from pagestorm.protocol import (
    PagestormStageProtocol,
    build_chapter_text_completion_prefix,
    build_full_book_chapter_plan_completion_prefix,
    build_stage_completion_prefix,
    chapter_text_prefill,
)


def test_builds_conditional_stage_prefix_from_accepted_text() -> None:
    protocol = PagestormStageProtocol(
        PROFILE_NAME="14b",
        DEFAULT_STAGE_SEQUENCE=("book_plan",),
        STAGE_ROLES=("prompt", "book_preview", "book_plan"),
        _structured_outputs={},
    )
    bundle = cast(StoryBundle, SimpleNamespace(protocol=protocol))

    prefix = build_stage_completion_prefix(
        bundle,
        prompt="Write a mystery.",
        completed_stages=[("book_preview", "## Book Title\nThe Ledger")],
        stage_role="book_plan",
        accepted_stage_prefix="## World Rules\n- Records cannot be destroyed.\n\n",
    )

    assert prefix.endswith(
        "<|start_header_id|>book_plan<|stop_header_id|>\n\n"
        "## World Rules\n- Records cannot be destroyed.\n\n"
    )
    assert prefix.count("<|eot_id|>") == 2


def test_builds_first_chapter_completion_contract() -> None:
    protocol = PagestormStageProtocol(
        PROFILE_NAME="14b",
        DEFAULT_STAGE_SEQUENCE=("first_chapter_text",),
        STAGE_ROLES=("prompt", "first_chapter_text"),
        _structured_outputs={},
    )
    bundle = cast(
        StoryBundle,
        SimpleNamespace(
            protocol=protocol,
            profile=SimpleNamespace(name="14b"),
            tokenizer=SimpleNamespace(eos_token_id=42),
        ),
    )

    prefix = build_chapter_text_completion_prefix(
        bundle,
        prompt="Write a mystery.",
        completed_stages=[],
        stage_role="first_chapter_text",
        chapter_name="The Locked Room",
    )

    assert prefix == ChapterTextCompletionPrefix(
        serialized_stage_prefix=(
            "<|start_header_id|>prompt<|stop_header_id|>\n\n"
            "Write a mystery.<|eot_id|>"
            "<|start_header_id|>first_chapter_text<|stop_header_id|>\n\n"
            "## The Locked Room\n```\n"
        ),
        stage_text_prefix="## The Locked Room\n```\n",
        eot_token="<|eot_id|>",
        chapter_end_stop_reasons=("<|eot_id|>", 42, None),
    )


def test_builds_later_chapter_plan_completion_contract() -> None:
    protocol = PagestormStageProtocol(
        PROFILE_NAME="14b",
        DEFAULT_STAGE_SEQUENCE=("full_book_chapters_plan",),
        STAGE_ROLES=("prompt", "book_preview", "full_book_chapters_plan"),
        _structured_outputs={},
    )
    bundle = cast(
        StoryBundle,
        SimpleNamespace(
            protocol=protocol,
            profile=SimpleNamespace(name="14b"),
        ),
    )

    prefix = build_full_book_chapter_plan_completion_prefix(
        bundle,
        prompt="Write a mystery.",
        completed_stages=[("book_preview", "## Book Title\nThe Ledger")],
        accepted_stage_text=(
            "### Chapter Two\n"
            "**Word Count:** 2000\n"
            "**Embedding Space:** action: 10"
        ),
        planned_chapter={
            "chapter_name": "Chapter Three",
            "chapter_word_count": 2400,
            "chapter_embedding_space": (
                "action: 20, dialog: 30, world_building: 10, exposition: 20, "
                "romantic: 0, erotic: 0, pacing: 40"
            ),
        },
        next_chapter_name="Chapter Four",
    )

    assert prefix.stage_text_prefix.startswith("\n\n### Chapter Three\n")
    assert prefix.serialized_stage_prefix.endswith(prefix.stage_text_prefix)
    assert prefix.stop_strings == ("### Chapter Four", "<|eot_id|>")
    assert prefix.structured_outputs["regex"].startswith(r"^\n")


@pytest.mark.parametrize("chapter_name", ["", "One\nTwo"])
def test_rejects_invalid_chapter_names(chapter_name: str) -> None:
    with pytest.raises(ValueError, match="one non-empty line"):
        chapter_text_prefill(chapter_name)


def test_validates_structured_stage_text_with_python_regex() -> None:
    protocol = PagestormStageProtocol(
        PROFILE_NAME="test",
        DEFAULT_STAGE_SEQUENCE=("stage",),
        STAGE_ROLES=("stage",),
        _structured_outputs={"stage": {"regex": r"^accepted\z"}},
    )

    assert protocol.stage_text_matches("stage", "accepted")
    assert not protocol.stage_text_matches("stage", "rejected")


def test_splits_book_plan_at_trained_component_boundaries() -> None:
    components = split_book_plan_components(
        "## World Rules\n- Records bind.\n\n"
        "## Short Story Arcs\n### Arc 1: Discovery\n- Mara opens the ledger."
    )

    assert components == [
        ("world_rules", "## World Rules\n- Records bind."),
        (
            "short_story_arcs",
            "## Short Story Arcs\n### Arc 1: Discovery\n- Mara opens the ledger.",
        ),
    ]


def test_splits_first_chapter_plan_at_trained_component_boundary() -> None:
    components = split_first_chapter_plan_components(
        "## One\n### Chapters Plan\n**Word Count:** 1000\n\n"
        "### Scene Breakdown\n#### Scene 1: The Ledger"
    )

    assert components == [
        (
            "first_chapter_plan_header",
            "## One\n### Chapters Plan\n**Word Count:** 1000",
        ),
        (
            "first_chapter_scene_breakdown",
            "### Scene Breakdown\n#### Scene 1: The Ledger",
        ),
    ]
