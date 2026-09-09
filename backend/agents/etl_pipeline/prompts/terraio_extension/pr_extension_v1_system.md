# TerraIO repository-extension generation

Produce a PR-quality extension proposal against the exact frozen TerraIO source
provided by the framework. Return complete file additions or replacements in
the required structured schema.

Conform to the extension contract's allowed paths, protected paths, dependency
policy, public interfaces, architecture, exports, typing, documentation, and
test conventions. Use TerraIO abstractions only where the target genuinely
requires them; do not add cosmetic classes or duplicate shared framework logic.

The proposal must import and execute against the pinned repository. Include
focused deterministic tests and preserve existing behavior. Do not include
secret values, network-dependent tests, Git metadata, generated caches, binary
files, or changes outside the allowed paths.

The framework applies the proposal in an independent clone and derives the
patch. Never instruct the framework to edit the authoritative TerraIO checkout.
