from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import artifacts_14b, artifacts_24b
from .models import (
    ChapterTextCompletionPrefix,
    FullBookChapterPlanCompletionPrefix,
)

if TYPE_CHECKING:
    from .artifacts_14b import PlannedChapterInfo
    from .models import StoryBundle

EOT_TOKEN = "<|eot_id|>"
START_HEADER_TOKEN = "<|start_header_id|>"
STOP_HEADER_TOKEN = "<|stop_header_id|>"

STAGE_ROLES_BY_PROFILE: dict[str, tuple[str, ...]] = {
    "24b": (
        "prompt",
        "book_plan",
        "first_chapter_plan",
        "first_chapter_scene_breakdown",
        "first_chapter_text",
    ),
    "14b": (
        "prompt",
        "book_preview",
        "book_plan",
        "first_chapter_plan",
        "first_chapter_text",
        "full_book_chapters_plan",
        "book_characters_list",
        "scene_breakdown",
        "chapter_text",
    ),
}

DEFAULT_STAGE_SEQUENCE_BY_PROFILE: dict[str, tuple[str, ...]] = {
    "24b": (
        "book_plan",
        "first_chapter_plan",
        "first_chapter_scene_breakdown",
        "first_chapter_text",
    ),
    "14b": (
        "book_preview",
        "book_plan",
        "first_chapter_plan",
        "first_chapter_text",
        "full_book_chapters_plan",
        "book_characters_list",
        "scene_breakdown",
        "chapter_text",
    ),
}

STRUCTURED_OUTPUTS_BY_PROFILE: dict[str, dict[str, dict[str, str]]] = {
    "24b": {
        "book_plan": {"regex": artifacts_24b.BOOK_PLAN_REGEX_24B},
        "first_chapter_plan": {"regex": artifacts_24b.FIRST_CHAPTER_PLAN_REGEX_24B},
        "first_chapter_scene_breakdown": {"regex": artifacts_24b.SCENE_BREAKDOWN_REGEX},
    },
    "14b": {
        "book_preview": {"regex": artifacts_14b.BOOK_PREVIEW_REGEX},
        "book_plan": {"regex": artifacts_14b.BOOK_PLAN_REGEX_14B},
        "first_chapter_plan": {"regex": artifacts_14b.FIRST_CHAPTER_PLAN_REGEX_14B},
        "scene_breakdown": {"regex": artifacts_14b.SCENE_BREAKDOWN_REGEX},
        "book_characters_list": {"regex": artifacts_14b.BOOK_CHARACTERS_LIST_REGEX},
    },
}


def _stage_header(role: str) -> str:
    return f"{START_HEADER_TOKEN}{role}{STOP_HEADER_TOKEN}"


def chapter_text_prefill(chapter_name: str) -> str:
    normalized_chapter_name = chapter_name.strip()
    if not normalized_chapter_name or "\n" in normalized_chapter_name:
        raise ValueError("Chapter name must be one non-empty line")
    return f"## {normalized_chapter_name}\n```\n"


def build_stage_completion_prefix(
    bundle: StoryBundle,
    *,
    prompt: str,
    completed_stages: list[tuple[str, str]],
    stage_role: str,
    accepted_stage_prefix: str = "",
) -> str:
    """Serialize a stage prefix for external conditional generation."""

    serialized_prefix = bundle.protocol.render_stage_prompt(
        prompt,
        completed_stages,
        stage_role,
    )
    return serialized_prefix + accepted_stage_prefix


def build_chapter_text_completion_prefix(
    bundle: StoryBundle,
    *,
    prompt: str,
    completed_stages: list[tuple[str, str]],
    stage_role: str,
    chapter_name: str,
) -> ChapterTextCompletionPrefix:
    if bundle.profile.name != "14b":
        raise ValueError(
            "Chapter Text completion prefixes are currently supported only "
            "for the 14b profile"
        )
    if stage_role not in {"first_chapter_text", "chapter_text"}:
        raise ValueError(f"Unsupported Chapter Text stage role: {stage_role}")

    stage_text_prefix = chapter_text_prefill(chapter_name)
    serialized_stage_prefix = build_stage_completion_prefix(
        bundle,
        prompt=prompt,
        completed_stages=completed_stages,
        stage_role=stage_role,
        accepted_stage_prefix=stage_text_prefix,
    )
    end_reasons: list[int | str | None] = [bundle.protocol.EOT_TOKEN]
    tokenizer_eos_token_id = getattr(bundle.tokenizer, "eos_token_id", None)
    if isinstance(tokenizer_eos_token_id, int):
        end_reasons.append(tokenizer_eos_token_id)
    end_reasons.append(None)
    return ChapterTextCompletionPrefix(
        serialized_stage_prefix=serialized_stage_prefix,
        stage_text_prefix=stage_text_prefix,
        eot_token=bundle.protocol.EOT_TOKEN,
        chapter_end_stop_reasons=tuple(end_reasons),
    )


def build_full_book_chapter_plan_completion_prefix(
    bundle: StoryBundle,
    *,
    prompt: str,
    completed_stages: list[tuple[str, str]],
    accepted_stage_text: str,
    planned_chapter: PlannedChapterInfo,
    next_chapter_name: str | None,
) -> FullBookChapterPlanCompletionPrefix:
    if bundle.profile.name != "14b":
        raise ValueError(
            "Full-book chapter-plan completion prefixes are supported only "
            "for the 14b profile"
        )
    stage_text_prefix = (
        ("\n\n" if accepted_stage_text else "")
        + f"### {planned_chapter['chapter_name']}\n"
        + f"**Word Count:** {planned_chapter['chapter_word_count']}\n"
        + f"**Embedding Space:** {planned_chapter['chapter_embedding_space']}"
    )
    serialized_stage_prefix = build_stage_completion_prefix(
        bundle,
        prompt=prompt,
        completed_stages=completed_stages,
        stage_role="full_book_chapters_plan",
        accepted_stage_prefix=accepted_stage_text + stage_text_prefix,
    )
    stop_strings = list(
        bundle.protocol.stage_stop_strings("full_book_chapters_plan")
    )
    if next_chapter_name is not None:
        stop_strings.insert(0, f"### {next_chapter_name}")
    return FullBookChapterPlanCompletionPrefix(
        serialized_stage_prefix=serialized_stage_prefix,
        stage_text_prefix=stage_text_prefix,
        stop_strings=tuple(stop_strings),
        structured_outputs={
            "regex": artifacts_14b.build_full_book_chapter_plan_continuation_regex(),
        },
    )


@dataclass
class PagestormStageProtocol:
    PROFILE_NAME: str
    DEFAULT_STAGE_SEQUENCE: tuple[str, ...]
    STAGE_ROLES: tuple[str, ...]
    _structured_outputs: dict[str, dict[str, str]]

    EOT_TOKEN: str = EOT_TOKEN

    def stage_header(self, role: str) -> str:
        if role not in self.STAGE_ROLES:
            raise ValueError(f"Unsupported story stage role: {role}")
        return _stage_header(role)

    def stage_stop_strings(self, role: str) -> list[str]:
        if role not in self.STAGE_ROLES:
            raise ValueError(f"Unsupported story stage role: {role}")
        return [self.EOT_TOKEN]

    def stage_structured_outputs(self, role: str) -> dict[str, Any] | None:
        structured_outputs = self._structured_outputs.get(role)
        if structured_outputs is None:
            return None
        return dict(structured_outputs)

    def stage_text_matches(self, role: str, text: str) -> bool:
        structured_outputs = self.stage_structured_outputs(role)
        if structured_outputs is None:
            return True
        regex = structured_outputs.get("regex")
        if not isinstance(regex, str):
            raise TypeError(
                f"Structured output for {role!r} must include a string regex"
            )
        return re.fullmatch(regex.replace(r"\z", r"\Z"), text) is not None

    def render_stage_prompt(
        self, prompt: str, completed_stages: list[tuple[str, str]] | None, next_stage_role: str
    ) -> str:
        if next_stage_role == "prompt":
            raise ValueError("prompt cannot be used as a generation stage")
        if next_stage_role not in self.STAGE_ROLES:
            raise ValueError(f"Unsupported story stage role: {next_stage_role}")
        rendered_parts = [f"{self.stage_header('prompt')}\n\n{prompt}{self.EOT_TOKEN}"]
        for role, content in completed_stages or []:
            if role == "prompt":
                raise ValueError("completed_stages must not include prompt")
            rendered_parts.append(f"{self.stage_header(role)}\n\n{content}{self.EOT_TOKEN}")
        rendered_parts.append(f"{self.stage_header(next_stage_role)}\n\n")
        return "".join(rendered_parts)

    def trim_generated_stage_text(
        self,
        text: str,
        generated_stage_role: str,
        *,
        strip_leading_blank_lines: bool = True,
    ) -> str:
        if generated_stage_role not in self.STAGE_ROLES:
            raise ValueError(f"Unsupported story stage role: {generated_stage_role}")
        trimmed_stage_text = text.split(self.EOT_TOKEN, 1)[0]
        normalized_lines = [line.rstrip() for line in trimmed_stage_text.split("\n")]
        if strip_leading_blank_lines:
            while normalized_lines and normalized_lines[0] == "":
                normalized_lines.pop(0)
        while normalized_lines and normalized_lines[-1] == "":
            normalized_lines.pop()
        return "\n".join(normalized_lines)


def build_story_stage_protocol(profile_name: str) -> PagestormStageProtocol:
    if profile_name not in STAGE_ROLES_BY_PROFILE:
        raise ValueError(f"Unsupported profile: {profile_name}")
    return PagestormStageProtocol(
        PROFILE_NAME=profile_name,
        DEFAULT_STAGE_SEQUENCE=DEFAULT_STAGE_SEQUENCE_BY_PROFILE[profile_name],
        STAGE_ROLES=STAGE_ROLES_BY_PROFILE[profile_name],
        _structured_outputs=STRUCTURED_OUTPUTS_BY_PROFILE[profile_name],
    )
