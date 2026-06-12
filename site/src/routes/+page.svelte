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
		return notebook.export ? `${base}${notebook.assetPath}` : notebook.sourceUrl;
	}
</script>

<svelte:head>
	<title>{manifest.site.title}</title>
	<meta name="description" content={manifest.site.description} />
</svelte:head>

<div class="min-h-screen bg-paper font-sans text-ink antialiased">
	<header class="mx-auto w-full max-w-5xl px-6 pt-20 pb-14 text-center sm:pt-28">
		<h1 class="font-display text-5xl leading-tight font-medium sm:text-6xl">
			{manifest.site.title}
		</h1>
		<p class="mx-auto mt-4 max-w-xl text-lg leading-relaxed text-soft">
			{manifest.site.description}
		</p>
		<nav class="mt-10 flex flex-wrap justify-center gap-2" aria-label="Cursos">
			{#each groups as group (group.id)}
				<a
					href="#{group.id}"
					class="rounded-full border border-line bg-surface px-4 py-1.5 text-sm transition-colors hover:border-ink"
				>
					{group.course}
				</a>
			{/each}
		</nav>
	</header>

	<main class="mx-auto w-full max-w-5xl px-6 pb-24">
		{#each groups as group (group.id)}
			<section
				id={group.id}
				class="scroll-mt-8 border-t border-line py-12"
				aria-labelledby="{group.id}-title"
			>
				<h2 id="{group.id}-title" class="font-display text-3xl font-medium">{group.course}</h2>

				{#each group.sections as section (section.id)}
					<div class="mt-8">
						<p
							class="inline-block rounded-full border border-line bg-surface px-3 py-1 text-xs font-semibold tracking-wide text-soft uppercase"
						>
							{section.section}
						</p>
						<div class="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
							{#each section.notebooks as notebook (notebook.slug)}
								<article
									class="group relative flex flex-col rounded-2xl border border-line bg-surface p-6 transition duration-150 hover:-translate-y-0.5 hover:border-soft/60 hover:shadow-[0_10px_28px_-14px_rgba(33,32,28,0.35)]"
								>
									<h3 class="font-display text-xl leading-snug font-medium">
										<a
											href={notebookHref(notebook)}
											rel="external"
											class="rounded-2xl after:absolute after:inset-0 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink"
										>
											{notebook.title}
										</a>
									</h3>
									<p class="mt-2 text-sm leading-relaxed text-soft">{notebook.summary}</p>
									<div class="mt-auto flex items-center justify-between pt-6 text-sm">
										{#if notebook.export}
											<span class="inline-flex items-center gap-2 text-pine">
												<span class="size-1.5 rounded-full bg-pine"></span>
												En linea
											</span>
											<span class="font-medium group-hover:underline">Abrir cuaderno</span>
										{:else}
											<span class="inline-flex items-center gap-2 text-soft">
												<span class="size-1.5 rounded-full bg-soft/70"></span>
												Local
											</span>
											<span class="font-medium group-hover:underline">Ver codigo</span>
										{/if}
									</div>
								</article>
							{/each}
						</div>
					</div>
				{/each}
			</section>
		{/each}

		<section class="rounded-3xl bg-panel px-6 py-10 sm:px-10" aria-labelledby="local-title">
			<h2 id="local-title" class="font-display text-3xl font-medium">Ejecucion local</h2>
			<p class="mt-3 max-w-2xl leading-relaxed text-soft">
				Los cuadernos marcados como locales dependen de herramientas que no funcionan en el
				navegador, como gfortran, GDAL o credenciales de Earth Engine. Cada tarjeta local enlaza
				al codigo fuente con la ruta exacta del cuaderno.
			</p>
			<pre
				class="mt-6 overflow-x-auto rounded-xl bg-shell p-5 text-sm leading-relaxed text-shell-ink"><code
					>git clone {manifest.site.repository}.git
cd undergrad
./install.sh
uv run --package &lt;curso&gt; marimo edit &lt;cuaderno.py&gt;</code></pre>
		</section>
	</main>

	<footer class="border-t border-line">
		<div
			class="mx-auto flex w-full max-w-5xl flex-wrap items-center justify-between gap-3 px-6 py-8 text-sm text-soft"
		>
			<p>{manifest.site.title}</p>
			<a href={manifest.site.repository} class="transition-colors hover:text-ink">
				Codigo fuente en GitHub
			</a>
		</div>
	</footer>
</div>
