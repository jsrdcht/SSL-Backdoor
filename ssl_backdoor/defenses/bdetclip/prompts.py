"""Safely load the ImageNet classes, templates, and BDetCLIP prompts."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path


def _dictionary_expression(path: Path) -> ast.Dict:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise ValueError(f"{path} must contain exactly one dictionary expression")
    value = tree.body[0].value
    if not isinstance(value, ast.Dict):
        raise ValueError(f"{path} must contain a top-level dictionary")
    return value


def _dict_value(node: ast.Dict, key: str) -> ast.AST:
    for candidate, value in zip(node.keys, node.values):
        if isinstance(candidate, ast.Constant) and candidate.value == key:
            return value
    raise ValueError(f"class resource is missing {key!r}")


def _template_from_lambda(node: ast.AST) -> str:
    if not isinstance(node, ast.Lambda) or len(node.args.args) != 1:
        raise ValueError("templates must be single-argument lambdas")
    parameter = node.args.args[0].arg
    if not isinstance(node.body, ast.JoinedStr):
        raise ValueError("template lambdas must return f-strings")
    parts = []
    for item in node.body.values:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            parts.append(item.value.replace("{", "{{").replace("}", "}}"))
        elif (
            isinstance(item, ast.FormattedValue)
            and isinstance(item.value, ast.Name)
            and item.value.id == parameter
            and item.format_spec is None
        ):
            parts.append("{}")
        else:
            raise ValueError("class template contains an unsupported Python expression")
    return "".join(parts)


@dataclass(frozen=True)
class PromptBank:
    classes: list[str]
    templates: list[str]
    benign: list[list[str]]
    malignant: list[str]

    @classmethod
    def load(cls, classes_path: str, benign_path: str, malignant_path: str) -> "PromptBank":
        dictionary = _dictionary_expression(Path(classes_path).expanduser())
        classes = ast.literal_eval(_dict_value(dictionary, "classes"))
        template_nodes = _dict_value(dictionary, "templates")
        if not isinstance(template_nodes, ast.List):
            raise ValueError("templates must be a list")
        templates = [_template_from_lambda(item) for item in template_nodes.elts]

        with Path(benign_path).expanduser().open(encoding="utf-8") as file:
            benign_by_name = json.load(file)
        malignant = Path(malignant_path).expanduser().read_text(encoding="utf-8").splitlines()
        missing = sorted(set(classes) - set(benign_by_name))
        if missing:
            raise ValueError(f"benign prompts are missing classes: {missing[:5]}")
        benign = [benign_by_name[name] for name in classes]

        if len(classes) != 1000 or len(templates) != 80 or len(malignant) != 1000:
            raise ValueError(
                "ImageNet BDetCLIP resources must contain 1000 classes, 80 templates, "
                "and 1000 malignant prompts"
            )
        if any(
            len(items) != 7
            or not all(isinstance(item, str) for item in items)
            or not any(item.strip() for item in items)
            for items in benign
        ):
            raise ValueError(
                "each ImageNet class must contain seven strings with at least one non-empty string"
            )
        if any(not item.strip() for item in malignant):
            raise ValueError("malignant prompts must not contain empty lines")
        return cls(classes, templates, benign, malignant)

    def benign_texts(self) -> list[str]:
        return [text for descriptions in self.benign for text in descriptions]

    def malignant_texts(self) -> list[str]:
        return [
            template.format(name).replace(".", ",")
            + " "
            + sentence.lower().replace('"', "")
            for name, sentence in zip(self.classes, self.malignant)
            for template in self.templates
        ]
