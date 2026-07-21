<script lang="ts">
	import type { RunMode } from '$lib/catalog.generated';
	import { runModeLabels, runModeTitles } from '$lib/runModes';

	let { mode }: { mode: RunMode } = $props();
</script>

<span class="runstate" data-mode={mode} title={runModeTitles[mode]}>
	{#if mode === 'interactive'}
		<svg viewBox="0 0 16 16" aria-hidden="true">
			<rect x="1.5" y="2.5" width="13" height="11" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.3" />
			<line x1="1.5" y1="5.4" x2="14.5" y2="5.4" stroke="currentColor" stroke-width="1.3" />
			<path d="M6.9 7.4 11 9.6 6.9 11.8Z" fill="currentColor" />
		</svg>
	{:else if mode === 'results'}
		<svg viewBox="0 0 16 16" aria-hidden="true">
			<rect x="1.5" y="2.5" width="13" height="11" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.3" />
			<path d="M4 11 6.7 8.1 8.6 9.7 11.9 6" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" />
			<circle cx="11.9" cy="6" r="0.5" fill="currentColor" />
		</svg>
	{:else}
		<svg viewBox="0 0 16 16" aria-hidden="true">
			<rect x="1.5" y="2.5" width="13" height="11" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.3" />
			<path d="M4.7 6.6 7 8.6 4.7 10.6" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" />
			<line x1="8.4" y1="10.7" x2="11.5" y2="10.7" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" />
		</svg>
	{/if}
	<span>{runModeLabels[mode]}</span>
</span>

<style>
	.runstate {
		display: inline-flex;
		align-items: center;
		gap: 0.4rem;
		font-family: var(--font-mono);
		font-weight: 500;
		font-size: var(--text-label);
		color: var(--content-secondary);
	}
	.runstate[data-mode="interactive"],
	.runstate[data-mode="results"] {
		color: var(--content);
	}
	.runstate svg {
		flex: none;
		width: 0.95rem;
		height: 0.95rem;
	}
</style>
