"""Token -> dollar pricing.

The defaults here are *illustrative* and deliberately conservative placeholders.
Real prices change frequently and vary by provider, region, and contract; set
your own with :meth:`PriceBook.with_rate` or by passing a mapping to the
constructor. Caliper never hardcodes a vendor's price list as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelRate:
    """USD per 1M tokens, input and output priced separately."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.input_per_mtok
            + output_tokens / 1_000_000 * self.output_per_mtok
        )


@dataclass
class PriceBook:
    """Maps a model name to a :class:`ModelRate`.

    Lookup is exact-match first, then longest-prefix match, so a rate registered
    for ``"gpt-4o"`` also covers ``"gpt-4o-2024-08-06"``. Unknown models fall back
    to :attr:`fallback` (zero by default, so unpriced models are visible as $0
    rather than silently inflating spend).
    """

    rates: dict[str, ModelRate] = field(default_factory=dict)
    fallback: ModelRate = ModelRate(0.0, 0.0)

    def with_rate(self, model: str, input_per_mtok: float, output_per_mtok: float) -> PriceBook:
        self.rates[model] = ModelRate(input_per_mtok, output_per_mtok)
        return self

    def rate_for(self, model: str | None) -> ModelRate:
        if not model:
            return self.fallback
        if model in self.rates:
            return self.rates[model]
        best: ModelRate | None = None
        best_len = -1
        for name, rate in self.rates.items():
            if model.startswith(name) and len(name) > best_len:
                best, best_len = rate, len(name)
        return best if best is not None else self.fallback

    def cost(self, model: str | None, input_tokens: int, output_tokens: int) -> float:
        return self.rate_for(model).cost(input_tokens, output_tokens)

    @classmethod
    def default(cls) -> PriceBook:
        """A starter price book. **Verify and override these for production.**"""
        return cls(
            rates={
                # Illustrative placeholders — confirm against current provider pricing.
                "gpt-4o-mini": ModelRate(0.15, 0.60),
                "gpt-4o": ModelRate(2.50, 10.00),
                "claude-haiku": ModelRate(0.80, 4.00),
                "claude-sonnet": ModelRate(3.00, 15.00),
                "claude-opus": ModelRate(15.00, 75.00),
            }
        )
