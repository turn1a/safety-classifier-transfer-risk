/* Explain marked terms while preserving their links to the study guide. */
(() => {
  document.querySelectorAll('main table').forEach((table) => {
    const panel = document.createElement('div');
    panel.className = 'table-scroll';
    table.before(panel);
    panel.append(table);
    const caption = table.closest('figure')?.querySelector('figcaption');
    const update = () => {
      const overflows = panel.scrollWidth > panel.clientWidth + 1;
      panel.tabIndex = overflows ? 0 : -1;
      if (overflows) {
        panel.setAttribute('role', 'region');
        panel.setAttribute('aria-label', caption?.textContent.trim() || 'Data table');
      } else {
        panel.removeAttribute('role');
        panel.removeAttribute('aria-label');
      }
    };
    new ResizeObserver(update).observe(panel);
    update();
  });

  const definitions = new Map();
  document.querySelectorAll('.metric-definitions dt[id]').forEach((term) => {
    const description = term.nextElementSibling;
    if (description?.tagName === 'DD') {
      definitions.set(term.id, {
        label: term.textContent.trim(),
        text: description.textContent.trim(),
      });
    }
  });

  if (!definitions.size) return;

  const tooltip = document.createElement('div');
  tooltip.id = 'metric-tooltip';
  tooltip.className = 'metric-tooltip';
  tooltip.setAttribute('role', 'tooltip');
  tooltip.hidden = true;
  document.body.append(tooltip);
  let active = null;
  let closeTimer;

  const close = () => {
    clearTimeout(closeTimer);
    active?.removeAttribute('aria-describedby');
    active = null;
    tooltip.hidden = true;
  };

  const position = () => {
    if (!active) return;
    const availableWidth = document.documentElement.clientWidth;
    tooltip.style.maxWidth = `${availableWidth - 24}px`;
    const anchor = active.getBoundingClientRect();
    const box = tooltip.getBoundingClientRect();
    const margin = 12;
    const left = Math.max(margin, Math.min(anchor.left, availableWidth - box.width - margin));
    const below = anchor.bottom + 8;
    const top = below + box.height <= window.innerHeight - margin
      ? below
      : Math.max(margin, anchor.top - box.height - 8);
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  };

  const open = (link, definition) => {
    close();
    active = link;
    const label = document.createElement('strong');
    label.textContent = definition.label;
    const description = document.createElement('p');
    description.textContent = definition.text;
    tooltip.replaceChildren(label, description);
    link.setAttribute('aria-describedby', tooltip.id);
    tooltip.hidden = false;
    position();
  };

  const deferClose = () => {
    clearTimeout(closeTimer);
    closeTimer = setTimeout(() => {
      if (!active?.matches(':focus') && !tooltip.matches(':hover')) close();
    }, 150);
  };

  document.querySelectorAll('a.term-help').forEach((link) => {
    const definition = definitions.get(new URL(link.href).hash.slice(1));
    if (!definition) return;
    link.addEventListener('mouseenter', () => open(link, definition));
    link.addEventListener('focus', () => open(link, definition));
    link.addEventListener('mouseleave', deferClose);
    link.addEventListener('blur', deferClose);
  });

  tooltip.addEventListener('mouseenter', () => clearTimeout(closeTimer));
  tooltip.addEventListener('mouseleave', deferClose);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') close();
  });
  document.addEventListener('scroll', () => {
    if (active?.matches(':focus')) position();
    else close();
  }, {passive: true, capture: true});
  window.addEventListener('resize', position);
})();
