from pathlib import Path

import pytest

from plugin_eval.layers.static import StaticAnalyzer
from plugin_eval.models import LayerResult


class TestStaticAnalyzer:
    def test_analyze_valid_skill(self, sample_skill_dir: Path):
        analyzer = StaticAnalyzer()
        result = analyzer.analyze_skill(sample_skill_dir)
        assert isinstance(result, LayerResult)
        assert result.layer == "static"
        assert result.score > 0.5
        assert len(result.anti_patterns) == 0

    def test_analyze_poor_skill(self, poor_skill_dir: Path):
        analyzer = StaticAnalyzer()
        result = analyzer.analyze_skill(poor_skill_dir)
        assert result.score < 0.7
        flags = [ap.flag for ap in result.anti_patterns]
        assert "OVER_CONSTRAINED" in flags
        assert "MISSING_TRIGGER" in flags

    def test_analyze_plugin(self, sample_plugin_dir: Path):
        analyzer = StaticAnalyzer()
        result = analyzer.analyze_plugin(sample_plugin_dir)
        assert result.layer == "static"
        assert result.score > 0.5
        assert "skill_scores" in result.sub_scores
        assert "agent_scores" in result.sub_scores

    def test_anti_pattern_penalty(self):
        analyzer = StaticAnalyzer()
        assert analyzer._anti_pattern_penalty(0) == 1.0
        assert analyzer._anti_pattern_penalty(2) == pytest.approx(0.9)
        assert analyzer._anti_pattern_penalty(10) == 0.5
        assert analyzer._anti_pattern_penalty(20) == 0.5

    def test_description_pushiness_score(self):
        analyzer = StaticAnalyzer()
        good = "Test skill for evaluation. Use when testing plugin-eval. Use PROACTIVELY for quality checks."
        weak = "A skill."
        assert analyzer._description_pushiness(good) > analyzer._description_pushiness(weak)


def _make_skill_with_frontmatter(
    tmp_path: Path, frontmatter_lines: list[str], name: str = "test-skill"
) -> Path:
    skill_dir = tmp_path / name
    skill_dir.mkdir()
    frontmatter = "\n".join(frontmatter_lines)
    (skill_dir / "SKILL.md").write_text(
        f"---\n{frontmatter}\n---\n\n# Skill\n\n## Overview\n\nBody.\n"
    )
    return skill_dir


class TestTriggerExemptions:
    """`disable-model-invocation: true` and `paths:` frontmatter should exempt
    a skill from the MISSING_TRIGGER check, because those skills are not
    auto-invoked from the description.
    """

    def test_disable_model_invocation_exempts_skill(self, tmp_path: Path) -> None:
        # No "Use when…" phrase, but the skill is explicitly slash-only.
        skill_dir = _make_skill_with_frontmatter(
            tmp_path,
            [
                "name: setup",
                "description: One-time setup that adds .claude/state/ to the project's .gitignore.",
                "disable-model-invocation: true",
            ],
        )
        analyzer = StaticAnalyzer()
        result = analyzer.analyze_skill(skill_dir)
        flags = [ap.flag for ap in result.anti_patterns]
        assert "MISSING_TRIGGER" not in flags, (
            "Slash-only skills should not be flagged for missing description trigger"
        )

    def test_paths_auto_load_exempts_skill(self, tmp_path: Path) -> None:
        # Path-triggered skill: invocation is driven by `paths:` glob, not the description.
        skill_dir = _make_skill_with_frontmatter(
            tmp_path,
            [
                "name: self-evaluate",
                "description: Self-critical evaluation guard for test/spec files.",
                'paths: "**/*test*,**/*spec*"',
            ],
        )
        analyzer = StaticAnalyzer()
        result = analyzer.analyze_skill(skill_dir)
        flags = [ap.flag for ap in result.anti_patterns]
        assert "MISSING_TRIGGER" not in flags, (
            "Path-triggered skills should not be flagged for missing description trigger"
        )

    def test_disable_model_invocation_false_still_checks_trigger(
        self, tmp_path: Path
    ) -> None:
        # Explicitly model-invocable, no trigger phrase → should still flag.
        skill_dir = _make_skill_with_frontmatter(
            tmp_path,
            [
                "name: model-invocable",
                "description: A description without a trigger phrase whatsoever.",
                "disable-model-invocation: false",
            ],
        )
        analyzer = StaticAnalyzer()
        result = analyzer.analyze_skill(skill_dir)
        flags = [ap.flag for ap in result.anti_patterns]
        assert "MISSING_TRIGGER" in flags, (
            "Model-invocable skills without a trigger phrase must still be flagged"
        )

    def test_empty_paths_value_still_checks_trigger(self, tmp_path: Path) -> None:
        # `paths:` with no value is not a valid auto-load configuration.
        skill_dir = _make_skill_with_frontmatter(
            tmp_path,
            [
                "name: bad-paths",
                "description: Some skill description that lacks the trigger phrase.",
                'paths: ""',
            ],
        )
        analyzer = StaticAnalyzer()
        result = analyzer.analyze_skill(skill_dir)
        flags = [ap.flag for ap in result.anti_patterns]
        assert "MISSING_TRIGGER" in flags
