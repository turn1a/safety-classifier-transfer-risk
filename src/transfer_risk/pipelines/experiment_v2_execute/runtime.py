"""Reuse existing ML helpers while retaining v2 source query and identity accounting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from transfer_risk.lib.cka import cka_matrix
from transfer_risk.lib.experiment_v2.protocol import partition_digest
from transfer_risk.lib.experiment_v2.schemas import SimilarityRecord
from transfer_risk.pipelines.experiment_v2_execute.offline import (
    configure_offline_assets,
    require_offline_assets,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from transfer_risk.lib.experiment_v2.schemas import CorpusEvidence, ModelIdentity, Protocol


class SourceQueryBudgetError(RuntimeError):
    """Signal an attempted source-query request exceeding its registered per-cell cap."""


class QueryLedger:
    """Count actual source calls, cache repeated texts, and enforce the complete cap."""

    def __init__(self, wrapper: Callable[[list[str]], Any], query_budget: int) -> None:
        """Start one isolated original/recipe query ledger.

        Args:
            wrapper: Existing transformer or masked-BiLSTM source wrapper.
            query_budget: Complete per-cell cap, including its original source query.
        """
        self.wrapper = wrapper
        self.query_budget = query_budget
        self.source_queries = 0
        self._cache: dict[str, Any] = {}

    def __call__(self, texts: list[str]) -> Any:
        """Query only uncached texts and charge even source forwards that raise.

        Args:
            texts: Candidate texts requested by the existing recipe.

        Returns:
            Scores in the original request order.

        Raises:
            SourceQueryBudgetError: Before a request would exceed the fixed cap.
        """
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if self.source_queries + len(missing) > self.query_budget:
            msg = "source recipe attempted to exceed its complete query cap"
            raise SourceQueryBudgetError(msg)
        if missing:
            self.source_queries += len(missing)
            scores = self.wrapper(missing)
            if hasattr(scores, "detach"):
                scores = scores.detach().cpu().numpy()
            self._cache.update(zip(missing, scores, strict=True))
        return np.asarray([self._cache[text] for text in texts])


def train_model(
    identity: ModelIdentity,
    backbone: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    recipe: dict[str, Any],
    device_policy: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train through existing loops on exact registered membership and output indices.

    Args:
        identity: Reviewed raw output ordering and canonical label map.
        backbone: Catalog-loaded immutable backbone.
        train: Exact registered source training partition.
        validation: Exact separate registered validation partition.
        recipe: Fixed architecture-specific training settings.
        device_policy: Existing project device policy.
        seed: Root-derived model/replica seed.

    Returns:
        Existing checkpoint bundle and training metadata.
    """
    from transfer_risk.devices import resolve_device  # noqa: PLC0415
    from transfer_risk.pipelines.models.training import (  # noqa: PLC0415
        finetune_transformer,
        train_bilstm,
    )

    mapping = {
        int(identity.label_map[raw] == "prompt_injection"): index
        for index, raw in enumerate(identity.output_labels)
    }
    train, validation = train.copy(), validation.copy()
    train["label"] = train["label"].map(mapping)
    validation["label"] = validation["label"].map(mapping)
    device = resolve_device(device_policy)
    if identity.kind == "bilstm":
        return train_bilstm(train, validation, params=recipe, device=device, seed=seed)
    return finetune_transformer(backbone, train, validation, recipe, device, seed)


def make_source_wrapper(
    identity: ModelIdentity, descriptor: dict[str, str], settings: dict[str, Any]
) -> Any:
    """Load one saved source once per shard through the existing victim wrapper.

    Args:
        identity: Reviewed source kind and token window.
        descriptor: Exact isolated saved checkpoint descriptor.
        settings: Registered device policy.

    Returns:
        Existing source classifier wrapper, without any target model access.
    """
    from transfer_risk.devices import resolve_device  # noqa: PLC0415

    device = resolve_device(settings["device_policy"])
    configure_offline_assets(settings.get("_attack_assets"), str(device))
    from transfer_risk.pipelines.attacks.runner import build_wrapper  # noqa: PLC0415

    return build_wrapper(
        {"kind": identity.kind, "source": descriptor["path"]},
        device,
        max_seq_len=identity.max_tokens,
    )


def tokenization_diagnostics(
    texts: list[str], tokenizer: Any, max_tokens: int
) -> list[dict[str, Any]]:
    """Measure each full input before the registered token-window truncation.

    Args:
        texts: Exact lower-trust classifier inputs.
        tokenizer: Saved transformer tokenizer, or None for the fixed BiLSTM tokenizer.
        max_tokens: Registered inference window.

    Returns:
        Token counts, window, and truncation status without filtering any input.
    """
    if tokenizer is None:
        from transfer_risk.pipelines.models.bilstm import tokenize  # noqa: PLC0415

        lengths = [len(tokenize(text)) for text in texts]
    else:
        encoded = tokenizer(texts, truncation=False, padding=False, add_special_tokens=True)
        lengths = [len(ids) for ids in encoded["input_ids"]]
    return [
        {"input_tokens": length, "max_tokens": max_tokens, "truncated": length > max_tokens}
        for length in lengths
    ]


def predict_model(
    identity: ModelIdentity, descriptor: dict[str, str], texts: list[str], settings: dict[str, Any]
) -> list[str]:
    """Reuse existing prediction helpers and retain the reviewed raw output ordering.

    Args:
        identity: Detector's actual output ordering and registered window.
        descriptor: Exact private checkpoint descriptor.
        texts: Exact originals or already-generated candidates.
        settings: Registered device and inference batch size.

    Returns:
        Unmodified raw output labels in the requested text order.
    """
    from transfer_risk.devices import resolve_device  # noqa: PLC0415
    from transfer_risk.modeling import load_transformer, predict  # noqa: PLC0415

    device = resolve_device(settings["device_policy"])
    batch_size = settings["inference_batch_size"]
    if identity.kind == "bilstm":
        wrapper = make_source_wrapper(identity, descriptor, settings)
        tokenizer = None
        indices = []
        for start in range(0, len(texts), batch_size):
            indices.extend(np.argmax(wrapper(texts[start : start + batch_size]), axis=-1).tolist())
    else:
        model, tokenizer = load_transformer(descriptor["path"], device)
        indices = predict(
            model,
            tokenizer,
            texts,
            max_seq_len=identity.max_tokens,
            batch_size=batch_size,
            device=device,
        )
    if "_tokenization_diagnostics" in settings:
        settings["_tokenization_diagnostics"].extend(
            tokenization_diagnostics(texts, tokenizer, identity.max_tokens)
        )
    return [identity.output_labels[index] for index in indices]


def _ledger_wrapper(ledger: QueryLedger, source: Any) -> Any:
    """Adapt accounting to TextAttack's existing ModelWrapper interface.

    Args:
        ledger: Per-original query cap and cache.
        source: Existing classifier wrapper.

    Returns:
        A TextAttack wrapper delegating every classifier call through the ledger.
    """
    from textattack.models.wrappers import ModelWrapper  # noqa: PLC0415

    class CountedWrapper(ModelWrapper):  # type: ignore[misc]
        """Expose the existing served model while counting its source queries."""

        def __init__(self) -> None:
            """Expose model/tokenizer metadata required by recipe compatibility checks."""
            self.model = source.model
            self.ledger = ledger
            if hasattr(source, "tokenizer"):
                self.tokenizer = source.tokenizer

        def __call__(self, texts: list[str]) -> Any:
            """Return source scores through the complete bounded query ledger.

            Args:
                texts: Recipe-requested candidate strings.

            Returns:
                Existing source score arrays.
            """
            return self.ledger(texts)

    return CountedWrapper()


def load_recipe_backend() -> tuple[dict[str, Any], Callable[[int], None]]:
    """Load existing TextAttack recipes only when a future attack cell executes.

    Returns:
        Existing recipe builders and TextAttack's seed setter.
    """
    require_offline_assets()
    from textattack.attack_recipes import TextBuggerLi2018  # noqa: PLC0415
    from textattack.shared.utils import set_seed  # noqa: PLC0415

    from transfer_risk.pipelines.attacks.runner import RECIPES  # noqa: PLC0415

    return {**RECIPES, "textbugger": TextBuggerLi2018}, set_seed


def attack_original(
    model_input: dict[str, Any],
    original: dict[str, Any],
    settings: dict[str, Any],
    seed: int,
    query_budget: int,
) -> dict[str, Any]:
    """Run one existing recipe and preserve its candidate, status, and actual query count.

    Args:
        model_input: Only the source identity, artifact, and reusable source wrapper.
        original: One exact registered positive original.
        settings: Registered recipe and query batch size.
        seed: Root-derived source/recipe/replica seed.
        query_budget: Complete per-original cap including its original source query.

    Returns:
        Private candidate text and actual bounded source accounting, including errors.
    """
    from transfer_risk.lib.experiment_v2.schemas import ModelIdentity  # noqa: PLC0415

    identity = ModelIdentity.model_validate(model_input["identity"])
    returned: dict[str, Any] = {
        "status": "error",
        "candidate_text": None,
        "source_raw_perturbed": None,
        "source_queries": 0,
        "error_type": None,
    }
    if model_input.get("setup_error"):
        return {**returned, "error_type": model_input["setup_error"]}
    ledger = None
    try:
        source = model_input.get("runtime_wrapper")
        if source is None:
            source = make_source_wrapper(identity, model_input["artifact"], settings)
        tokenizer = getattr(source, "tokenizer", None)
        if tokenizer is not None or identity.kind == "bilstm":
            returned["original_tokenization"] = tokenization_diagnostics(
                [original["text"]], tokenizer, identity.max_tokens
            )[0]
        recipes, set_seed = load_recipe_backend()
        ledger = QueryLedger(source, query_budget)
        set_seed(seed)
        session_key = (settings["recipe"], query_budget, settings["query_batch_size"])
        session = model_input.get("_recipe_session")
        if session is None:
            counted = _ledger_wrapper(ledger, source)
            recipe = recipes[settings["recipe"]].build(counted)
            recipe.goal_function.query_budget = query_budget
            recipe.goal_function.batch_size = settings["query_batch_size"]
            model_input["_recipe_session"] = {
                "key": session_key,
                "wrapper": counted,
                "recipe": recipe,
            }
        else:
            if session["key"] != session_key:
                message = "one source shard cannot change its recipe or fixed query settings"
                raise ValueError(message)
            recipe = session["recipe"]
            session["wrapper"].ledger = ledger
            recipe.clear_cache()
        set_seed(seed)
        original_index = int(np.argmax(ledger([original["text"]])[0]))
        if identity.label_map[identity.output_labels[original_index]] == "benign":
            returned["status"] = "skipped_source_benign"
        else:
            result = recipe.attack(original["text"], original_index)
            final = result.perturbed_result
            raw = identity.output_labels[int(final.output)]
            returned.update(
                {
                    "status": "success" if identity.label_map[raw] == "benign" else "failed",
                    "candidate_text": final.attacked_text.text,
                    "source_raw_perturbed": raw,
                }
            )
            if tokenizer is not None or identity.kind == "bilstm":
                returned["candidate_tokenization"] = tokenization_diagnostics(
                    [final.attacked_text.text], tokenizer, identity.max_tokens
                )[0]
    except Exception as error:
        returned["error_type"] = type(error).__name__
    returned["source_queries"] = 0 if ledger is None else ledger.source_queries
    return returned


def compute_similarities(
    protocol: Protocol,
    corpus: CorpusEvidence,
    manifest: dict[str, Any],
    probe: list[str],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reuse full-probe representations and pure globally centered CKA.

    Args:
        protocol: Frozen target roles, source pool, windows, and replicas.
        corpus: Independent probe identity.
        manifest: Exact realized artifact descriptors.
        probe: Complete independently grouped representation-probe texts.
        settings: Registered pooling, device, and inference batch size.

    Returns:
        Mean layer-pair CKA bound to exact source and target identities.
    """
    from transfer_risk.devices import resolve_device  # noqa: PLC0415
    from transfer_risk.modeling import layer_representations  # noqa: PLC0415

    models = {model.id: model for model in protocol.models}
    result = []
    for replicate in protocol.replicates:
        representations = {}
        for model in protocol.models:
            descriptor = manifest["artifacts"][f"{model.id}:r{replicate}"]
            representations[model.id] = layer_representations(
                {"kind": model.kind, "source": descriptor["path"]},
                probe,
                pooling=settings["pooling"],
                max_seq_len=model.max_tokens,
                batch_size=settings["inference_batch_size"],
                device=resolve_device(settings["device_policy"]),
            )
        for role in protocol.target_roles:
            target = manifest["artifacts"][f"{role.model_id}:r{replicate}"]
            for source_id in role.candidate_ids:
                source = manifest["artifacts"][f"{source_id}:r{replicate}"]
                matrix = cka_matrix(representations[role.model_id], representations[source_id])
                result.append(
                    SimilarityRecord(
                        target_role=role.id,
                        source_id=models[source_id].id,
                        replicate=replicate,
                        mean_cka=float(matrix.mean()),
                        probe_partition_sha256=partition_digest(corpus, "probe"),
                        source_checkpoint_sha256=source["checkpoint_sha256"],
                        source_tokenizer_sha256=source["tokenizer_sha256"],
                        target_checkpoint_sha256=target["checkpoint_sha256"],
                        target_tokenizer_sha256=target["tokenizer_sha256"],
                    ).model_dump()
                )
    return result
