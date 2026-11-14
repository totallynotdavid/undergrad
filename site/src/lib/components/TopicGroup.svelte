<script lang="ts">
	import NotebookCard from './NotebookCard.svelte';
	import type { Topic } from '#lib/catalog.generated.js';

	let { topic }: { topic: Topic } = $props();
</script>

<section class="topic" id={topic.id}>
	<div class="topic__head">
		<h2 class="topic__title">{topic.title}</h2>
		<a class="topic__folder" href={topic.folderUrl} target="_blank" rel="external noopener">Source on GitHub ↗</a>
	</div>

	<div class="cards">
		{#each topic.notebooks as notebook (notebook.slug)}
			<NotebookCard {notebook} />
		{/each}
	</div>
</section>

<style>
	.topic {
		display: flex;
		flex-direction: column;
	}
	.topic__head {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		justify-content: space-between;
		gap: 0.5rem 1rem;
		padding-bottom: 0.6rem;
		border-bottom: 2px solid var(--content);
		margin-bottom: 1.75rem;
	}
	.topic__title {
		margin: 0;
	}
	.topic__folder {
		font-family: var(--font-mono);
		font-weight: 500;
		font-size: var(--text-label);
		color: var(--content-secondary);
		white-space: nowrap;
	}
	.topic__folder:hover {
		color: var(--accent);
	}
	.cards {
		display: flex;
		flex-wrap: wrap;
		gap: 1.25rem;
	}
</style>
