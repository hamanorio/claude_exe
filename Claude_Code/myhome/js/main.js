// スマホ表示のメニュー（≡ボタン）の開閉
document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.querySelector('.menu-toggle');
    const links = document.querySelector('.nav-links');
    if (!toggle || !links) return;
    toggle.setAttribute('aria-expanded', 'false');
    toggle.addEventListener('click', () => {
        const open = links.classList.toggle('active');
        toggle.setAttribute('aria-expanded', String(open));
    });
    links.addEventListener('click', e => {
        if (e.target.closest('a')) {
            links.classList.remove('active');
            toggle.setAttribute('aria-expanded', 'false');
        }
    });
});
