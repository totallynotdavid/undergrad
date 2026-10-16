<script lang="ts">
	let { exported }: { exported: boolean } = $props();
</script>

<span
	class="runstate"
	data-exported={exported}
	title={exported ? 'Resultados exportados, solo lectura' : 'Requiere ejecucion local con uv'}
>
	{#if exported}
		<svg viewBox="0 0 16 16" aria-hidden="true">
			<rect x="1.5" y="2.5" width="13" height="11" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.3" />
			<path d="M4 11 6.7 8.1 8.6 9.7 11.9 6" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" />
			<circle cx="11.9" cy="6" r="0.5" fill="currentColor" />
		</svg>
		<span>Resultados</span>
	{:else}
		<svg viewBox="0 0 16 16" aria-hidden="true">
			<rect x="1.5" y="2.5" width="13" height="11" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.3" />
			<path d="M4.7 6.6 7 8.6 4.7 10.6" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round" />
			<line x1="8.4" y1="10.7" x2="11.5" y2="10.7" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" />
		</svg>
		<span>Local</span>
	{/if}
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
	.runstate[data-exported="true"] {
		color: var(--content);
	}
	.runstate svg {
		flex: none;
		width: 0.95rem;
		height: 0.95rem;
	}
</style>
