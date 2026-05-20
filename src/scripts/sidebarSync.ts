/**
 * Shared helper that wires the left "Parameters" sidebar accordion to the
 * form fields in the centre column.
 *
 * Behaviour, identical on both /fh-peds and the ML calculator at /:
 *   - Focusing a form field highlights it AND opens (and scrolls into view)
 *     the corresponding <details> entry in the sidebar.
 *   - Manually opening a sidebar entry highlights the matching form field.
 *
 * Only one sidebar entry is "active" at a time; opening another collapses
 * the previously-opened one. This is the same UX as a controlled
 * accordion, but implemented on top of native <details> elements.
 *
 * The function is page-agnostic — it discovers fields and accordion items
 * by DOM convention (`.field[data-field]` inside the passed form,
 * `.desc-accordion__item[data-desc-id]` inside the sidebar) so each page
 * just needs to call it once with its own form.
 */

export function setupSidebarSync(form: HTMLFormElement): void {
  const sidebar = document.getElementById('desc-sidebar');

  // Map: field id → <details> element in the accordion.
  const descItems = new Map<string, HTMLDetailsElement>();
  if (sidebar) {
    sidebar.querySelectorAll<HTMLElement>('.desc-accordion__item').forEach((li) => {
      const id = li.getAttribute('data-desc-id');
      const details = li.querySelector<HTMLDetailsElement>('details');
      if (id && details) descItems.set(id, details);
    });
  }

  function setActive(fieldName: string): void {
    form.querySelectorAll<HTMLElement>('.field').forEach((f) => {
      f.classList.toggle('field--active', f.dataset.field === fieldName);
    });
    descItems.forEach((details, id) => {
      const li = details.parentElement;
      const active = id === fieldName;
      li?.classList.toggle('desc-accordion__item--active', active);
      if (active) {
        details.open = true;
      } else if (details.open) {
        details.open = false;
      }
    });
  }

  // Wire field focus → expand matching accordion entry.
  form.querySelectorAll<HTMLElement>('.field').forEach((fieldEl) => {
    const name = fieldEl.dataset.field;
    if (!name) return;
    fieldEl
      .querySelectorAll<HTMLInputElement | HTMLSelectElement>('input, select')
      .forEach((inp) => {
        inp.addEventListener('focus', () => setActive(name));
      });
  });

  // Clicking an accordion summary should also highlight the matching form
  // field, so the visual link between the left and centre columns is
  // bi-directional.
  descItems.forEach((details, id) => {
    details.addEventListener('toggle', () => {
      if (!details.open) return;
      form.querySelectorAll<HTMLElement>('.field').forEach((f) => {
        f.classList.toggle('field--active', f.dataset.field === id);
      });
      details.parentElement?.classList.add('desc-accordion__item--active');
    });
  });
}
