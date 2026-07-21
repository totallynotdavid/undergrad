<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { onMount } from 'svelte';

	import CourseGroup from '#lib/components/CourseGroup.svelte';
	import Controls from '#lib/components/Controls.svelte';
	import Footer from '#lib/components/Footer.svelte';
	import Navbar from '#lib/components/Navbar.svelte';
	import { site, courses } from '#lib/catalog.generated';
	import {
		applyFilters,
		EMPTY_FILTERS,
		filtersToSearchParams,
		parseFilters,
		type Filters
	} from '#lib/filters';

	// Read URL filters only after hydration so the prerendered HTML still matches.
	let mounted = $state(false);

	onMount(() => {
		mounted = true;
	});

	const filters = $derived(
		mounted ? parseFilters(page.url.searchParams) : EMPTY_FILTERS
	);
	const filtered = $derived(applyFilters(courses, filters));

	function update(next: Filters) {
		const query = filtersToSearchParams(next).toString();

		goto(query ? `?${query}` : location.pathname, {
			replaceState: true,
			keepFocus: true,
			noScroll: true
		});
	}

	function focusSearch(event: KeyboardEvent) {
		if (event.key !== '/') return;

		const target = event.target as HTMLElement | null;
		const tag = target?.tagName;

		if (
			tag === 'INPUT' ||
			tag === 'SELECT' ||
			tag === 'TEXTAREA' ||
			target?.isContentEditable
		) {
			return;
		}

		event.preventDefault();
		document.getElementById('nb-search')?.focus();
	}
</script>

<svelte:window onkeydown={focusSearch} />

<svelte:head>
	<title>{site.title}</title>
	<meta name="description" content={site.description} />
</svelte:head>

<div class="page">
	<Navbar title={site.title}>
		<Controls {filters} {courses} onchange={update} />
	</Navbar>

	<h1 class="sr-only">Cuadernos de fisica computacional</h1>

	<main class="main">
		{#if filtered.length === 0}
			<div class="empty">
				<p class="empty__message">
					Ningun cuaderno coincide con los filtros.
				</p>
				<button
					class="empty__reset"
					type="button"
					onclick={() => update(EMPTY_FILTERS)}
				>
					Limpiar filtros
				</button>
			</div>
		{:else}
			<div class="courses">
				{#each filtered as course (course.id)}
					<CourseGroup {course} />
				{/each}
			</div>
		{/if}
	</main>

	<Footer title={site.title} repository={site.repository} />
</div>

<style>
	.main {
		max-width: var(--layout-width);
		margin: 0 auto;
		padding: 2.5rem var(--gutter) var(--gutter);
	}

	.courses {
		display: flex;
		flex-direction: column;
		gap: 3.5rem;
	}

	.empty {
		padding: 4rem 1rem;
		text-align: center;
	}

	.empty__message {
		margin: 0 0 1rem 0;
		font-family: var(--font-mono);
		color: var(--content-secondary);
	}

	.empty__reset {
		padding: 0.5rem 1rem;
		background: transparent;
		border: 1px solid var(--content);
		border-radius: var(--radius);
		font: 500 var(--text-label) var(--font-mono);
		color: var(--content);
		cursor: pointer;
		transition:
			background 150ms ease-out,
			color 150ms ease-out;
	}

	.empty__reset:hover {
		background: var(--content);
		color: var(--page);
	}

	.empty__reset:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}
</style>
