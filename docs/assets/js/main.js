/**
 * g2-forge - Main JavaScript
 * Interactive enhancements for the documentation site
 */

document.addEventListener('DOMContentLoaded', function() {
    // Smooth scroll for anchor links
    initSmoothScroll();

    // Add scroll animations
    initScrollAnimations();

    // Syntax highlighting for code blocks
    initCodeHighlighting();

    // Add copy buttons to code blocks
    initCodeCopyButtons();
});

/**
 * Initialize smooth scrolling for anchor links
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

/**
 * Initialize scroll-triggered animations
 */
function initScrollAnimations() {
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe cards and sections
    const animatedElements = document.querySelectorAll(
        '.feature-card, .arch-card, .science-item, .doc-card, .reference-card'
    );

    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

/**
 * Basic syntax highlighting for code blocks
 */
function initCodeHighlighting() {
    document.querySelectorAll('pre code').forEach(block => {
        // Add language class for styling
        const languageMatch = block.className.match(/language-(\w+)/);
        if (languageMatch) {
            block.parentElement.classList.add(`language-${languageMatch[1]}`);
        }

        // Simple syntax highlighting for Python
        if (block.className.includes('language-python')) {
            highlightPython(block);
        }

        // Simple syntax highlighting for Bash
        if (block.className.includes('language-bash')) {
            highlightBash(block);
        }
    });
}

/**
 * Simple Python syntax highlighting
 */
function highlightPython(block) {
    let code = block.innerHTML;

    // Keywords
    const keywords = ['import', 'from', 'def', 'class', 'return', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'with', 'as', 'in', 'is', 'not', 'and', 'or'];
    keywords.forEach(keyword => {
        const regex = new RegExp(`\\b${keyword}\\b`, 'g');
        code = code.replace(regex, `<span style="color: #c678dd;">${keyword}</span>`);
    });

    // Strings
    code = code.replace(/(['"])(?:(?=(\\?))\2.)*?\1/g, '<span style="color: #98c379;">$&</span>');

    // Comments
    code = code.replace(/(#.*$)/gm, '<span style="color: #5c6370; font-style: italic;">$1</span>');

    // Numbers
    code = code.replace(/\b(\d+)\b/g, '<span style="color: #d19a66;">$1</span>');

    block.innerHTML = code;
}

/**
 * Simple Bash syntax highlighting
 */
function highlightBash(block) {
    let code = block.innerHTML;

    // Commands
    const commands = ['git', 'pip', 'python', 'cd', 'mkdir', 'ls', 'echo', 'export'];
    commands.forEach(cmd => {
        const regex = new RegExp(`\\b${cmd}\\b`, 'g');
        code = code.replace(regex, `<span style="color: #61afef;">${cmd}</span>`);
    });

    // Comments
    code = code.replace(/(#.*$)/gm, '<span style="color: #5c6370; font-style: italic;">$1</span>');

    // Flags
    code = code.replace(/(\s-[\w-]+)/g, '<span style="color: #c678dd;">$1</span>');

    block.innerHTML = code;
}

/**
 * Add copy buttons to code blocks
 */
function initCodeCopyButtons() {
    document.querySelectorAll('pre').forEach(pre => {
        // Create copy button
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.setAttribute('aria-label', 'Copy code to clipboard');

        // Style the button
        button.style.cssText = `
            position: absolute;
            top: 8px;
            right: 8px;
            padding: 4px 12px;
            background-color: rgba(255, 255, 255, 0.1);
            color: #e5e7eb;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: inherit;
        `;

        // Position the pre element
        pre.style.position = 'relative';

        // Add hover effect
        button.addEventListener('mouseenter', () => {
            button.style.backgroundColor = 'rgba(255, 255, 255, 0.2)';
        });

        button.addEventListener('mouseleave', () => {
            button.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
        });

        // Copy functionality
        button.addEventListener('click', async () => {
            const code = pre.querySelector('code');
            const text = code.textContent;

            try {
                await navigator.clipboard.writeText(text);
                button.textContent = 'Copied!';
                button.style.backgroundColor = 'rgba(34, 197, 94, 0.2)';
                button.style.borderColor = 'rgba(34, 197, 94, 0.4)';

                setTimeout(() => {
                    button.textContent = 'Copy';
                    button.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
                    button.style.borderColor = 'rgba(255, 255, 255, 0.2)';
                }, 2000);
            } catch (err) {
                button.textContent = 'Failed';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            }
        });

        pre.appendChild(button);
    });
}

/**
 * Add CSS class for animated elements
 */
const style = document.createElement('style');
style.textContent = `
    .animate-in {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
`;
document.head.appendChild(style);
