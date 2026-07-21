<script lang="ts">
	import { base } from '$app/paths';
	import RunState from './RunState.svelte';
	import type { Notebook } from '$lib/catalog.generated';

	let { notebook }: { notebook: Notebook } = $props();

	const href = $derived(notebook.export ? `${base}${notebook.assetPath}` : notebook.sourceUrl);
</script>

<a class="card" {href} target="_blank" rel="external noopener">
	<div class="card__body">
		<h4 class="card__title">{notebook.title}</h4>
		<p class="card__summary">{notebook.summary}</p>
	</div>
	<div class="card__foot">
		<RunState mode={notebook.mode} />
		<span class="card__path" title={notebook.path}>{notebook.path}</span>
	</div>
</a>

<style>
	.card {
		display: flex;
		flex: 1 1 17rem;
		max-width: 22rem;
		flex-direction: column;
		justify-content: space-between;
		gap: 1.25rem;
		padding: 1.35rem;
		background: var(--deemphasized);
		border: 2px solid var(--card-border);
		color: var(--content);
		transition:
			border-color 180ms ease-out,
			border-radius 180ms ease-out,
			box-shadow 180ms ease-out,
			transform 180ms ease-out;
	}
	.card:hover {
		border-color: var(--content);
		border-radius: 2px;
		box-shadow: 3px 3px 0 0 var(--content);
		transform: translate(-3px, -3px);
		text-decoration: none;
	}
	.card:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
	.card:visited {
		color: var(--content-secondary);
	}
	.card__body {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.card__title {
		margin: 0;
		font-size: var(--text-title);
		font-weight: 700;
		line-height: 1.3;
		text-wrap: balance;
	}
	.card:hover .card__title {
		color: var(--accent);
	}
	.card__summary {
		display: -webkit-box;
		-webkit-line-clamp: 3;
		line-clamp: 3;
		-webkit-box-orient: vertical;
		overflow: hidden;
		margin: 0;
		line-height: 1.5;
		color: var(--content-secondary);
	}
	.card__foot {
		display: flex;
		flex-direction: column;
		gap: 0.35rem;
		min-width: 0;
	}
	.card__path {
		overflow: hidden;
		font-family: var(--font-mono);
		font-size: var(--text-label);
		color: var(--content-secondary);
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	@media (prefers-reduced-motion: reduce) {
		.card {
			transition: border-color 150ms ease-out;
		}
		.card:hover {
			box-shadow: none;
			transform: none;
		}
	}
</style>
