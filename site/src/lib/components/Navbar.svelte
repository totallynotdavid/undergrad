<script lang="ts">
	import { base } from '$app/paths';
	import type { Snippet } from 'svelte';

	let {
		title,
		children
	}: { title: string; children?: Snippet } = $props();
</script>

<header class="topbar">
	<div class="topbar__inner">
		<a class="topbar__brand" href={base || '/'}>{title}</a>
		{#if children}
			<div class="topbar__controls">{@render children()}</div>
		{/if}
	</div>
</header>

<style>
	.topbar {
		position: sticky;
		top: 0;
		z-index: 200;
		background: var(--surface);
		border-bottom: 1px solid var(--line);
	}
	.topbar__inner {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.75rem 1.25rem;
		max-width: var(--layout-width);
		margin: 0 auto;
		padding: 0.7rem var(--gutter);
	}
	.topbar__brand {
		flex-shrink: 0;
		font-weight: 700;
		font-size: var(--text-title);
		color: var(--content);
	}
	.topbar__brand:hover {
		color: var(--content);
		text-decoration: none;
	}
	.topbar__controls {
		flex: 0 1 auto;
		margin-left: auto;
		display: flex;
		min-width: 0;
	}

	@media (max-width: 640px) {
		.topbar__controls {
			flex-basis: 100%;
			margin-left: 0;
		}
	}
</style>
