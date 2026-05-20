import { defineConfig } from 'astro/config';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

import cloudflare from '@astrojs/cloudflare';

// https://astro.build/config
export default defineConfig({
  site: 'https://fh-peds.pages.dev',
  trailingSlash: 'ignore',

  redirects: {
    '/ml-fh-peds': '/',
  },

  build: {
    inlineStylesheets: 'auto',
  },

  markdown: {
    // GFM (tables, strikethrough, task lists, autolinks) + math. Astro
    // replaces its built-in defaults when remarkPlugins is set, so we list
    // remark-gfm explicitly to keep GitHub-flavored Markdown enabled.
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [rehypeKatex],
  },

  adapter: cloudflare(),
});
