<script lang="ts">
	import type { Course } from '$lib/catalog.generated';
	import type { Filters } from '$lib/filters';

	let {
		filters,
		courses,
		onchange
	}: {
		filters: Filters;
		courses: Course[];
		onchange: (next: Filters) => void;
	} = $props();
</script>

<form class="controls" role="search" onsubmit={(event) => event.preventDefault()}>
	<div class="controls__field controls__field--search">
		<label class="controls__label" for="nb-search">Buscar</label>
		<input
			id="nb-search"
			name="q"
			class="controls__input"
			type="search"
			value={filters.query}
			oninput={(event) => onchange({ ...filters, query: event.currentTarget.value })}
			placeholder="Lorenz, Ising, Monte Carlo..."
			autocomplete="off"
		/>
	</div>
	<div class="controls__field">
		<label class="controls__label" for="nb-course">Curso</label>
		<select
			id="nb-course"
			name="curso"
			class="controls__input controls__select"
			value={filters.courseId}
			onchange={(event) => onchange({ ...filters, courseId: event.currentTarget.value })}
		>
			<option value="">Todos los cursos</option>
			{#each courses as course (course.id)}
				<option value={course.id}>{course.title}</option>
			{/each}
		</select>
	</div>
</form>

<style>
	.controls {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		justify-content: flex-end;
		gap: 0.6rem 0.75rem;
	}
	.controls__field {
		display: flex;
	}
	.controls__field--search {
		flex: 0 1 18rem;
		min-width: 0;
	}

	@media (max-width: 640px) {
		.controls {
			width: 100%;
		}
		.controls__field--search {
			flex: 1 1 auto;
		}
	}
	.controls__label {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}
	.controls__input {
		width: 100%;
		padding: 0.5rem 0.7rem;
		background: var(--page);
		border: 1px solid var(--line);
		border-radius: var(--radius);
		font: 400 var(--text-body)/1.4 var(--font-base);
		color: var(--content);
		accent-color: var(--accent);
	}
	.controls__input::placeholder {
		color: var(--content-secondary);
	}
	.controls__input:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 1px;
		border-color: var(--accent);
	}
	.controls__select {
		font-family: var(--font-mono);
		font-size: var(--text-label);
		cursor: pointer;
	}
</style>
