"""Render detailed, auditable MADRS-BERT word-attribution reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from symptom_scoring.config import PHQ_SYMPTOM_LABELS


CANVAS_WIDTH = 2400
MARGIN = 70
LEFT_WIDTH = 1650
GUTTER = 55
RIGHT_X = MARGIN + LEFT_WIDTH + GUTTER
RIGHT_WIDTH = CANVAS_WIDTH - RIGHT_X - MARGIN


def write_explanation_json(report: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output)
    return output


def read_explanation_json(path: str | Path) -> dict[str, Any] | None:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _font(size: int, *, mono: bool = False, bold: bool = False):
    family = "DejaVu Sans Mono" if mono else "DejaVu Sans"
    if bold and not mono:
        family = "DejaVu Sans"
    path = font_manager.findfont(
        font_manager.FontProperties(family=family, weight="bold" if bold else "normal")
    )
    return ImageFont.truetype(path, size=size)


def _blend(base: tuple[int, int, int], target: tuple[int, int, int], amount: float):
    amount = max(0.0, min(1.0, amount))
    return tuple(round(left + (right - left) * amount) for left, right in zip(base, target))


def _attribution_fill(normalized: float) -> tuple[int, int, int] | None:
    strength = min(0.82, abs(float(normalized)) * 0.82)
    if strength < 0.025:
        return None
    target = (220, 70, 70) if normalized > 0 else (55, 125, 220)
    return _blend((255, 255, 255), target, strength)


def _word_lookup(words: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {(int(word["start"]), int(word["end"])): word for word in words}


def _wrapped_input_lines(
    text: str,
    words: list[dict[str, Any]],
    max_chars: int,
) -> list[list[tuple[str, dict[str, Any] | None]]]:
    lookup = _word_lookup(words)
    rendered: list[list[tuple[str, dict[str, Any] | None]]] = []
    cursor = 0
    for source_line in text.splitlines(keepends=True):
        content = source_line.rstrip("\r\n")
        current: list[tuple[str, dict[str, Any] | None]] = []
        current_length = 0
        for match in re.finditer(r"\S+|\s+", content):
            token = match.group(0)
            absolute = (cursor + match.start(), cursor + match.end())
            if not token.isspace() and current and current_length + len(token) > max_chars:
                rendered.append(current)
                current = []
                current_length = 0
            if token.isspace() and not current:
                continue
            current.append((token, lookup.get(absolute)))
            current_length += len(token)
        rendered.append(current)
        cursor += len(source_line)
    if not rendered:
        rendered.append([])
    return rendered


def _top_words(item: dict[str, Any], direction: str) -> list[dict[str, Any]]:
    words = [
        word
        for word in item.get("words", [])
        if not word.get("structural")
        and (
            float(word.get("attribution", 0.0)) > 0
            if direction == "raises"
            else float(word.get("attribution", 0.0)) < 0
        )
    ]
    return sorted(
        words,
        key=lambda word: abs(float(word.get("attribution", 0.0))),
        reverse=True,
    )[:5]


def _panel_metrics(item: dict[str, Any], body_font) -> tuple[list, int]:
    text = str(item.get("effective_model_input") or item.get("model_input") or "")
    char_width = max(1, int(body_font.getlength("M")))
    lines = _wrapped_input_lines(text, item.get("words", []), LEFT_WIDTH // char_width)
    line_height = body_font.size + 9
    left_height = 100 + len(lines) * line_height
    right_height = 850
    return lines, max(left_height, right_height)


def _draw_input(
    draw: ImageDraw.ImageDraw,
    lines: list[list[tuple[str, dict[str, Any] | None]]],
    *,
    x: int,
    y: int,
    font,
) -> None:
    line_height = font.size + 9
    char_width = max(1, int(font.getlength("M")))
    for line in lines:
        cursor_x = x
        for token, word in line:
            token_width = max(char_width, int(font.getlength(token)))
            if word is not None:
                fill = _attribution_fill(float(word.get("normalized_attribution", 0.0)))
                if fill:
                    draw.rounded_rectangle(
                        (cursor_x - 2, y - 1, cursor_x + token_width + 2, y + font.size + 5),
                        radius=3,
                        fill=fill,
                    )
            draw.text((cursor_x, y), token, font=font, fill=(25, 32, 45))
            cursor_x += token_width
        y += line_height


def _draw_right_column(
    draw: ImageDraw.ImageDraw,
    item: dict[str, Any],
    *,
    x: int,
    y: int,
    normal_font,
    bold_font,
) -> None:
    dark = (25, 32, 45)
    muted = (85, 95, 110)
    line = normal_font.size + 10

    def put(text: str, *, bold: bool = False, color=dark, gap: int = 0):
        nonlocal y
        draw.text((x, y), text, font=bold_font if bold else normal_font, fill=color)
        y += line + gap

    status = str(item.get("status"))
    put(f"MADRS: {item.get('madrs_topic')}", bold=True)
    symptom = item.get("symptom")
    if symptom:
        put(f"PHQ: {PHQ_SYMPTOM_LABELS.get(symptom, symptom)}", color=muted, gap=5)
    else:
        put("PHQ mapping: none", color=muted, gap=5)
    put(f"Raw output: {float(item.get('raw_model_output', 0.0)):.3f}")
    put(f"Clipped score: {float(item.get('raw_score', 0.0)):.3f}")
    put(f"Rounded score: {item.get('rounded_score')}")
    put(f"Reproduced: {float(item.get('reproduced_raw_output', 0.0)):.3f}")
    put(f"Baseline: {float(item.get('baseline_raw_output', 0.0)):.3f}")
    put(f"Input compacted: {'yes' if item.get('input_compacted') else 'no'}")
    put(f"Tokenizer truncated: {'yes' if item.get('tokenizer_truncated') else 'no'}", gap=8)

    if status != "ok":
        put("Attribution unavailable", bold=True, color=(170, 45, 45))
        put(f"Score mismatch: {float(item.get('score_difference', 0.0)):.6f}", color=(170, 45, 45))
        return

    put(f"Completeness error: {float(item.get('completeness_error', 0.0)):+.4f}", color=muted, gap=6)
    put("Strongest raising words", bold=True, color=(175, 45, 45))
    raising = _top_words(item, "raises")
    if not raising:
        put("  none", color=muted)
    for word in raising:
        text = str(word.get("text", ""))
        if len(text) > 38:
            text = text[:35] + "..."
        put(f"  {text}  {float(word['attribution']):+.4f}", color=(145, 45, 45))

    y += 5
    put("Strongest lowering words", bold=True, color=(45, 85, 170))
    lowering = _top_words(item, "lowers")
    if not lowering:
        put("  none", color=muted)
    for word in lowering:
        text = str(word.get("text", ""))
        if len(text) > 38:
            text = text[:35] + "..."
        put(f"  {text}  {float(word['attribution']):+.4f}", color=(45, 75, 150))


def render_explanation_report(
    report: dict[str, Any],
    path: str | Path,
    *,
    phq_only: bool,
) -> Path:
    """Render one tall session PNG with exact highlighted inputs and scores."""

    items = [
        item for item in report.get("items", []) if not phq_only or item.get("symptom")
    ]
    body_font = _font(23, mono=True)
    normal_font = _font(24)
    small_font = _font(21)
    bold_font = _font(25, bold=True)
    title_font = _font(38, bold=True)
    section_font = _font(29, bold=True)

    panel_data = [_panel_metrics(item, body_font) for item in items]
    top_height = 190
    gap = 28
    total_height = top_height + sum(height + gap for _, height in panel_data) + 80
    total_height = max(total_height, 650)
    image = Image.new("RGB", (CANVAS_WIDTH, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    view = "PHQ-mapped topics" if phq_only else "all MADRS topics"
    draw.text(
        (MARGIN, 45),
        f"MADRS-BERT word attributions — {report.get('session_id')}",
        font=title_font,
        fill=(20, 28, 42),
    )
    draw.text((MARGIN, 100), view, font=normal_font, fill=(75, 85, 100))
    draw.rounded_rectangle((MARGIN, 143, MARGIN + 32, 169), radius=4, fill=(244, 154, 154))
    draw.text((MARGIN + 42, 140), "raises raw output", font=small_font, fill=(70, 70, 80))
    draw.rounded_rectangle((MARGIN + 280, 143, MARGIN + 312, 169), radius=4, fill=(145, 180, 235))
    draw.text((MARGIN + 322, 140), "lowers raw output", font=small_font, fill=(70, 70, 80))
    draw.text(
        (RIGHT_X, 140),
        "Local sensitivity, not proof of linguistic causality",
        font=small_font,
        fill=(100, 85, 65),
    )

    if not items:
        draw.text(
            (MARGIN, 300),
            "No accepted scores are available for this view.",
            font=section_font,
            fill=(85, 95, 110),
        )
    else:
        y = top_height
        previous_turn = None
        for item, (lines, panel_height) in zip(items, panel_data):
            turn = int(item.get("turn_index", 0))
            if turn != previous_turn:
                heading = f"Conversation {turn}"
                previous_turn = turn
            else:
                heading = f"Conversation {turn} (continued)"
            draw.rounded_rectangle(
                (MARGIN, y, CANVAS_WIDTH - MARGIN, y + panel_height),
                radius=12,
                fill=(250, 251, 253),
                outline=(210, 216, 225),
                width=2,
            )
            draw.text((MARGIN + 22, y + 18), heading, font=section_font, fill=(30, 42, 62))
            draw.text(
                (MARGIN + 360, y + 22),
                str(item.get("madrs_topic")),
                font=bold_font,
                fill=(70, 80, 100),
            )
            draw.line(
                (RIGHT_X - GUTTER // 2, y + 72, RIGHT_X - GUTTER // 2, y + panel_height - 25),
                fill=(215, 220, 228),
                width=2,
            )
            _draw_input(
                draw,
                lines,
                x=MARGIN + 22,
                y=y + 82,
                font=body_font,
            )
            _draw_right_column(
                draw,
                item,
                x=RIGHT_X,
                y=y + 82,
                normal_font=normal_font,
                bold_font=bold_font,
            )
            if item.get("tokenizer_truncated"):
                draw.text(
                    (MARGIN + 22, y + panel_height - 38),
                    "[Tokenizer-truncated suffix omitted from the effective model input]",
                    font=small_font,
                    fill=(170, 65, 45),
                )
            y += panel_height + gap

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    image.save(temporary, format="PNG", optimize=True)
    temporary.replace(output)
    return output
