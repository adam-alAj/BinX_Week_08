/**
 * BinX Week 8 Showcase — Main Application Logic
 * Navigation, sidebar, scroll handling, animations
 */

document.addEventListener('DOMContentLoaded', () => {
    // ============================================================
    // Sidebar Navigation
    // ============================================================
    const sidebar = document.getElementById('sidebar');
    const hamburger = document.getElementById('hamburger');
    const sidebarClose = document.getElementById('sidebarClose');
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.section');

    // Mobile toggle
    if (hamburger) {
        hamburger.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });
    }
    if (sidebarClose) {
        sidebarClose.addEventListener('click', () => {
            sidebar.classList.remove('open');
        });
    }

    // Close sidebar on nav click (mobile)
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            if (window.innerWidth <= 768) {
                sidebar.classList.remove('open');
            }
        });
    });

    // Active section tracking
    const observerOptions = {
        root: null,
        rootMargin: '-20% 0px -60% 0px',
        threshold: 0
    };

    const sectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.id;
                navItems.forEach(nav => {
                    nav.classList.toggle('active', nav.getAttribute('data-section') === id);
                });
            }
        });
    }, observerOptions);

    sections.forEach(section => {
        if (section.id) sectionObserver.observe(section);
    });

    // ============================================================
    // Scroll Animations
    // ============================================================
    const animateElements = document.querySelectorAll('.card, .timeline-item, .metric-card');
    animateElements.forEach(el => el.classList.add('animate-in'));

    const animateObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                animateObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    animateElements.forEach(el => animateObserver.observe(el));

    // ============================================================
    // Animated Counters
    // ============================================================
    const counters = document.querySelectorAll('[data-count]');
    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                counterObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    counters.forEach(c => counterObserver.observe(c));

    function animateCounter(el) {
        const target = parseInt(el.getAttribute('data-count'));
        const duration = 1500;
        const start = performance.now();

        function update(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = Math.floor(eased * target);
            el.textContent = current.toLocaleString();
            if (progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    }

    // ============================================================
    // SVG Circle Animations
    // ============================================================
    const circles = document.querySelectorAll('.circle-fg');
    circles.forEach(circle => {
        const dasharray = circle.getAttribute('stroke-dasharray');
        circle.style.strokeDasharray = '0, 100';
        setTimeout(() => {
            circle.style.strokeDasharray = dasharray;
        }, 500);
    });

    // ============================================================
    // Timeline item click navigation
    // ============================================================
    document.querySelectorAll('.timeline-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const href = item.getAttribute('href');
            if (href) {
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
        });
    });
});
