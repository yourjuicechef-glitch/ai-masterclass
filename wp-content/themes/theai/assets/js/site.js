/* Small UI: the mobile menu, the stuck bar, reveal-on-scroll. Runs at every tier. */
(() => {
	const bar = document.getElementById('bar');
	const menu = bar && bar.querySelector('.bar__menu');
	if (menu) menu.addEventListener('click', () => {
		const open = bar.classList.toggle('is-open');
		menu.setAttribute('aria-expanded', open);
	});
	const stick = () => bar && bar.classList.toggle('is-stuck', scrollY > 20);
	addEventListener('scroll', stick, { passive: true });
	stick();

	// Observe the PARENT of each .rv: the copy panels carry a clip-path, and an element's own
	// clip-path can keep an IntersectionObserver from ever firing on it.
	const rv = [...document.querySelectorAll('.rv')];
	if (!('IntersectionObserver' in window) || document.documentElement.classList.contains('is-static')) {
		rv.forEach(el => el.classList.add('in'));
		return;
	}
	const io = new IntersectionObserver(entries => {
		for (const e of entries) if (e.isIntersecting) {
			e.target.querySelectorAll(':scope > .rv').forEach(el => el.classList.add('in'));
			io.unobserve(e.target);
		}
	}, { rootMargin: '0px 0px -15% 0px' });
	new Set(rv.map(el => el.parentElement)).forEach(p => io.observe(p));
})();
