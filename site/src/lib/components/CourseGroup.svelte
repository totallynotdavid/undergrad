<script lang="ts">
	import NotebookCard from './NotebookCard.svelte';
	import type { Course } from '$lib/catalog.generated';

	let { course }: { course: Course } = $props();
</script>

<section class="course" id={course.id}>
	<div class="course__head">
		<h2 class="course__title">{course.title}</h2>
		<a class="course__folder" href={course.folderUrl} target="_blank" rel="external noopener">Carpeta en GitHub ↗</a>
	</div>

	{#each course.sections as section (section.id)}
		<div class="topic">
			<h3 class="topic__title">{section.title}</h3>
			<div class="cards">
				{#each section.notebooks as notebook (notebook.slug)}
					<NotebookCard {notebook} />
				{/each}
			</div>
		</div>
	{/each}
</section>

<style>
	.course {
		display: flex;
		flex-direction: column;
	}
	.course__head {
		display: flex;
		flex-wrap: wrap;
		align-items: baseline;
		justify-content: space-between;
		gap: 0.5rem 1rem;
		padding-bottom: 0.6rem;
		border-bottom: 2px solid var(--content);
		margin-bottom: 1.75rem;
	}
	.course__title {
		margin: 0;
	}
	.course__folder {
		font-family: var(--font-mono);
		font-weight: 500;
		font-size: var(--text-label);
		color: var(--content-secondary);
		white-space: nowrap;
	}
	.course__folder:hover {
		color: var(--accent);
	}
	.topic {
		margin-bottom: 2rem;
	}
	.topic:last-child {
		margin-bottom: 0;
	}
	.topic__title {
		margin: 0 0 0.9rem 0;
		font-family: var(--font-mono);
		font-size: var(--text-label);
		font-weight: 500;
		letter-spacing: 0.01em;
		color: var(--content-secondary);
	}
	.cards {
		display: flex;
		flex-wrap: wrap;
		gap: 1.25rem;
	}
</style>
