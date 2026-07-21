<script lang="ts">
	let { title, repository }: { title: string; repository: string } = $props();
</script>

<footer class="footer">
	<div class="footer__inner">
		<div class="footer__column">
			<h3>{title}</h3>
			<a href={repository} target="_blank" rel="external noopener">Codigo en GitHub ↗</a>
		</div>
		<div class="footer__credits">
			Cuadernos de cursos de pregrado en fisica computacional.
		</div>
	</div>
</footer>

<style>
	.footer {
		background: var(--deemphasized);
		font-family: var(--font-base);
		padding: 3rem var(--gutter) 1.5rem;
	}
	.footer__inner {
		display: flex;
		flex-direction: column;
		justify-content: space-between;
		max-width: var(--layout-width);
		margin: 0 auto;
	}
	.footer__column {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.footer__column a {
		color: var(--content);
	}
	.footer__credits {
		margin-top: 2rem;
		text-align: center;
		color: var(--content-secondary);
	}

	@media (min-width: 768px) {
		.footer__inner {
			flex-direction: row;
		}
		.footer__credits {
			margin-top: 3rem;
		}
	}
</style>
