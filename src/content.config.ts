import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

/**
 * Field-description markdown for the calculator sidebar.
 *
 * Each .md file is named after the corresponding form field
 * (e.g. `age.md`, `ldl_cholesterol.md`). `intro.md` is the placeholder
 * shown before any field is focused.
 *
 * Astro v6 requires every collection to have an explicit loader; the
 * `glob` loader picks up every .md file in the directory.
 */
const descriptions = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/descriptions' }),
  schema: z.object({}).passthrough().optional(),
});

export const collections = { descriptions };
