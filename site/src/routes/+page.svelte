<script lang="ts">
	import { base } from '$app/paths';
	import manifest from '$lib/notebooks.generated.json';

	type Notebook = {
		course_id: string;
		course: string;
		section_id: string;
		section: string;
		title: string;
		summary: string;
		path: string;
		package: string;
		slug: string;
		export: boolean;
		sourceUrl: string;
		assetPath: string;
	};

	type CourseGroup = {
		id: string;
		course: string;
		sections: {
			id: string;
			section: string;
			notebooks: Notebook[];
		}[];
	};

	const notebooks = manifest.notebooks as Notebook[];
	const onlineCount = notebooks.filter((notebook) => notebook.export).length;
	const localCount = notebooks.length - onlineCount;

	const groups: CourseGroup[] = [];
	for (const notebook of notebooks) {
		let courseGroup = groups.find((group) => group.id === notebook.course_id);
		if (!courseGroup) {
			courseGroup = { id: notebook.course_id, course: notebook.course, sections: [] };
			groups.push(courseGroup);
		}

		let sectionGroup = courseGroup.sections.find((group) => group.id === notebook.section_id);
		if (!sectionGroup) {
			sectionGroup = { id: notebook.section_id, section: notebook.section, notebooks: [] };
			courseGroup.sections.push(sectionGroup);
		}

		sectionGroup.notebooks.push(notebook);
	}

	function notebookHref(notebook: Notebook): string {
		return `${base}${notebook.assetPath}`;
	}
</script>

<svelte:head>
	<title>{manifest.site.title}</title>
	<meta name="description" content={manifest.site.description} />
</svelte:head>

<main>
	<header class="site-header">
		<div>
			<p class="eyebrow">Universidad Nacional de Ingenieria</p>
			<h1>{manifest.site.title}</h1>
			<p class="intro">{manifest.site.description}</p>
		</div>
		<div class="counts" aria-label="Resumen del catalogo">
			<span><strong>{onlineCount}</strong> en linea</span>
			<span><strong>{localCount}</strong> locales</span>
		</div>
	</header>

	{#each groups as group}
		<section class="course" aria-labelledby={group.id}>
			<h2 id={group.id}>{group.course}</h2>

			{#each group.sections as section}
				<div class="section-group">
					<h3>{section.section}</h3>
					<div class="grid">
						{#each section.notebooks as notebook}
							<article class="card">
								<div class="card-title">
									<h4>{notebook.title}</h4>
									{#if notebook.export}
										<span class="badge online">En linea</span>
									{:else}
										<span class="badge local">Local</span>
									{/if}
								</div>

								<p>{notebook.summary}</p>

								<div class="actions">
									{#if notebook.export}
										<a class="button" href={notebookHref(notebook)} rel="external">Abrir notebook</a>
									{:else}
										<a class="button secondary" href={notebook.sourceUrl}>Ver fuente</a>
									{/if}
									<a class="source" href={notebook.sourceUrl}>Codigo</a>
								</div>
							</article>
						{/each}
					</div>
				</div>
			{/each}
		</section>
	{/each}
</main>

<style>
	:global(body) {
		margin: 0;
		background: #f7faf8;
		color: #17211c;
		font-family:
			Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
		line-height: 1.5;
	}

	:global(*) {
		box-sizing: border-box;
	}

	main {
		width: min(1120px, calc(100% - 32px));
		margin: 0 auto;
		padding: 40px 0 56px;
	}

	.site-header {
		display: grid;
		grid-template-columns: minmax(0, 1fr) auto;
		gap: 24px;
		align-items: start;
		margin-bottom: 32px;
	}

	.eyebrow {
		margin: 0 0 8px;
		color: #4f6459;
		font-size: 0.88rem;
		font-weight: 650;
		letter-spacing: 0.02em;
		text-transform: uppercase;
	}

	h1,
	h2,
	h3,
	h4,
	p {
		margin-top: 0;
	}

	h1 {
		margin-bottom: 10px;
		font-size: clamp(2rem, 4vw, 3rem);
		line-height: 1.1;
	}

	.intro {
		max-width: 720px;
		margin-bottom: 0;
		color: #5b6961;
		font-size: 1.05rem;
	}

	.counts {
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
		justify-content: flex-end;
	}

	.counts span {
		border: 1px solid #d8e0db;
		border-radius: 999px;
		background: #ffffff;
		padding: 8px 12px;
		color: #5b6961;
		font-size: 0.92rem;
		white-space: nowrap;
	}

	.counts strong {
		color: #17211c;
	}

	.course {
		padding-top: 24px;
		border-top: 1px solid #d8e0db;
		margin-top: 28px;
	}

	h2 {
		margin-bottom: 18px;
		font-size: 1.55rem;
	}

	.section-group + .section-group {
		margin-top: 24px;
	}

	h3 {
		margin-bottom: 12px;
		color: #5b6961;
		font-size: 1rem;
		font-weight: 650;
	}

	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
		gap: 14px;
	}

	.card {
		display: flex;
		min-height: 170px;
		flex-direction: column;
		border: 1px solid #d8e0db;
		border-radius: 8px;
		background: #ffffff;
		padding: 18px;
	}

	.card-title {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 12px;
		margin-bottom: 10px;
	}

	h4 {
		margin-bottom: 0;
		font-size: 1.1rem;
		line-height: 1.25;
	}

	.card p {
		color: #5b6961;
	}

	.badge {
		flex: 0 0 auto;
		border-radius: 999px;
		padding: 3px 8px;
		font-size: 0.78rem;
		font-weight: 650;
		white-space: nowrap;
	}

	.online {
		background: #dcece5;
		color: #145c43;
	}

	.local {
		background: #f4edcf;
		color: #5a4a18;
	}

	.actions {
		display: flex;
		align-items: center;
		gap: 12px;
		margin-top: auto;
		padding-top: 18px;
	}

	a {
		color: #145c43;
	}

	.button {
		border-radius: 6px;
		background: #145c43;
		color: white;
		padding: 8px 12px;
		text-decoration: none;
		font-weight: 650;
	}

	.button.secondary {
		background: #eef5f1;
		color: #145c43;
	}

	.source {
		font-size: 0.92rem;
	}

	@media (max-width: 720px) {
		main {
			width: min(100% - 24px, 1120px);
			padding-top: 28px;
		}

		.site-header {
			display: block;
		}

		.counts {
			justify-content: flex-start;
			margin-top: 20px;
		}

		.card-title {
			display: block;
		}

		.badge {
			display: inline-block;
			margin-top: 8px;
		}
	}
</style>
