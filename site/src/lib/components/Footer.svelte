<script lang="ts">
	import type { Site, Topic } from '#lib/catalog.generated.js';

	let { site, topics }: { site: Site; topics: Topic[] } = $props();
</script>

<footer class="footer">
	<div class="footer__inner">
		<div class="footer__about">
			<h3 class="footer__title">{site.title}</h3>
			<p>
				Each notebook is written in Spanish and published as a page of its results. Your browser
				runs no Python. To change the parameters and rerun a notebook, clone the repository:
			</p>
			<pre class="footer__code"><code
					>git clone {site.repository}
cd notebooks
mise install -y && mise run install
uv run --package dynamics marimo edit \
  packages/dynamics/pendulo_simple.py</code
				></pre>
		</div>

		<nav class="footer__column" aria-label="Topics">
			<h4 class="footer__heading">Topics</h4>
			<ul>
				{#each topics as topic (topic.id)}
					<li><a href="?topic={topic.id}">{topic.title}</a></li>
				{/each}
			</ul>
		</nav>

		<nav class="footer__column" aria-label="Project">
			<h4 class="footer__heading">Project</h4>
			<ul>
				<li><a href={site.repository} rel="external noopener">Source on GitHub ↗</a></li>
				<li><a href="{site.repository}/issues" rel="external noopener">Report a problem ↗</a></li>
				<li><a href="{site.repository}/blob/HEAD/LICENSE" rel="external noopener">MIT license ↗</a></li>
			</ul>
		</nav>
	</div>
</footer>

<style>
	.footer {
		background: var(--deemphasized);
		font-family: var(--font-base);
		padding: 3rem 0 calc(2rem + env(safe-area-inset-bottom));
		color: var(--content-secondary);
	}
	.footer__inner {
		display: grid;
		grid-template-columns: 1fr;
		gap: 2.5rem;
		max-width: var(--layout-width);
		margin: 0 auto;
		padding: 0 var(--gutter);
	}
	.footer__title,
	.footer__heading {
		margin: 0 0 0.75rem;
		color: var(--content);
	}
	.footer__heading {
		font: 500 var(--text-label) var(--font-mono);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.footer__about p {
		max-width: 34rem;
		margin: 0 0 1rem;
		line-height: 1.5;
	}
	.footer__code {
		margin: 0;
		padding: 0.75rem 1rem;
		overflow-x: auto;
		background: var(--page);
		border: 1px solid var(--field-border);
		border-radius: var(--radius);
		font: 400 var(--text-label) / 1.5 var(--font-mono);
		color: var(--content);
	}
	.footer__code code {
		padding: 0;
		background: none;
		border: 0;
		font: inherit;
	}
	.footer__column ul {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.footer__column a {
		color: var(--content);
	}
	.footer__column a:hover {
		color: var(--accent);
	}

	@media (min-width: 768px) {
		.footer__inner {
			grid-template-columns: 2fr 1fr 1fr;
		}
	}
</style>
