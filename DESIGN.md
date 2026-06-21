---
name: Content Index
description: A dense local archive workbench for retrieval, reading, clustering, and synthesis.
colors:
  primary: "#1d4ed8"
  primary-foreground: "#ffffff"
  background: "#ffffff"
  foreground: "#172033"
  paper: "#f8f9fb"
  card: "#ffffff"
  muted: "#eef0f4"
  muted-foreground: "#667085"
  border: "#e2e5eb"
  input: "#eef0f4"
  rail-background: "#19191f"
  rail-surface: "#22232a"
  rail-foreground: "#e6e7eb"
  amber-state: "#b45309"
  success-state: "#047857"
  danger-state: "#b91c1c"
typography:
  display:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "1.25rem"
    fontWeight: 700
    lineHeight: 1.25
    letterSpacing: "-0.025em"
  headline:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "1rem"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.015em"
  title:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "0.875rem"
    fontWeight: 500
    lineHeight: 1.4
  body:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.625
  label:
    fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: 1.4
  telemetry:
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: "0.045em"
rounded:
  sm: "4px"
  md: "6px"
  lg: "8px"
  pill: "9999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
  page: "32px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.primary-foreground}"
    rounded: "{rounded.lg}"
    padding: "10px 16px"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.muted-foreground}"
    rounded: "{rounded.md}"
    padding: "8px 12px"
  input-search:
    backgroundColor: "{colors.card}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.lg}"
    padding: "10px 40px"
  card-default:
    backgroundColor: "{colors.card}"
    textColor: "{colors.foreground}"
    rounded: "{rounded.lg}"
    padding: "16px"
  chip-tag:
    backgroundColor: "{colors.muted}"
    textColor: "{colors.muted-foreground}"
    rounded: "{rounded.pill}"
    padding: "2px 10px"
---

# Design System: Content Index

## 1. Overview

**Creative North Star: “Archive Workbench”**

Content Index is a compact, light workbench with a dark tool rail for a private knowledge archive. It should feel operational and analytical: a place to triage unread material, scan metadata, follow topic clusters, run semantic searches, and ask synthesis questions without slipping into marketing theater.

The current direction is intentionally **light workspace, dark navigation rail**. The workspace uses white and cool paper-gray surfaces for long reading, quick scanning, and daytime research sessions. The rail stays graphite to provide a stable app frame and strong location memory. This is not a promotional light SaaS surface; it is a utilitarian desk: crisp, quiet, dense, and fast.

The interface explicitly rejects SaaS marketing clichés: glossy gradients, hero metrics, decorative card grids, glassmorphism, oversized empty-state illustrations, promotional page tropes, and AI-magic embellishment. Every visual choice must earn its place by improving retrieval, reading, clustering, ingestion, or synthesis.

**Key Characteristics:**
- Light, cool-neutral work surface with a dark graphite navigation rail.
- Dense information hierarchy with high contrast and compact row scanning.
- Cobalt accent reserved for primary action, focus, current selection, and links.
- Tonal layering and 1px borders instead of ambient shadows or glass.
- Structural components: rail navigation, rows, compact panels, tags, inputs, telemetry, and progress states.

## 2. Colors

The palette is a restrained cobalt-on-cool-neutral product palette. Neutral surfaces carry nearly all screen area. Blue appears where it communicates action, focus, selection, or navigation state.

### Primary
- **Workbench Cobalt**: The single accent for primary buttons, active navigation, links, focus rings, and selected states. Its rarity keeps the dashboard calm.

### Secondary
- **Amber Queue State**: A sparing semantic color for unread, waiting, or queue attention states. It should not decorate category labels.
- **Success Green**: Used for read/completed states and confirmation icons only.
- **Danger Red**: Used for errors and destructive feedback only.

### Neutral
- **Pure Desk**: The white content canvas used for rows and primary panels.
- **Cool Paper**: A very light blue-gray workspace background and secondary panel fill.
- **Blue-Black Ink**: Body and heading text.
- **Muted Slate**: Metadata, timestamps, helper text, keyboard hints, and inactive controls.
- **Fine Divider Gray**: Borders for panels, row boundaries, chips, and field strokes.
- **Graphite Rail**: The persistent navigation frame; it may be dark even when the workspace is light.

### Named Rules
**The Split-Frame Rule.** The dark rail is the app frame; the light workspace is the task surface. Do not scatter dark cards through the workspace unless the entire surface is intentionally changing mode.

**The Accent Ration Rule.** Cobalt belongs to action, focus, current selection, and links. If blue becomes decorative, the system loses operational clarity.

**The Semantic Color Rule.** Amber, green, and red are state colors only. Source types and tags should remain neutral unless they are actively filtered or in an error/warning/success state.

## 3. Typography

**Display Font:** system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif  
**Body Font:** system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif  
**Telemetry Font:** ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace

**Character:** One system sans carries the product. The type should feel native, fast, and familiar rather than expressive; distinction comes from weight, size, color, and spacing, not font switching.

### Hierarchy
- **Display** (700, 20px, 1.25 line-height): Page titles such as Archive, Browse, Search, and Add Content.
- **Headline** (600, 16px, 1.3 line-height): Reserved for major detail views or future larger section leads.
- **Title** (500/600, 13–14px, 1.4 line-height): Document titles, section labels, buttons, and navigation items.
- **Body** (400, 14px, 1.625 line-height): Summaries, descriptions, form text, and readable explanatory copy. Long prose should stay within 65–75ch.
- **Label** (500, 12px, 1.4 line-height): Metadata, filters, helper text, result counts, keyboard hints, and compact controls.
- **Telemetry** (500/700, 11–12px, 1.4 line-height, uppercase tracking): Telemetry rows, compact instrument labels, and keyboard hints. Avoid using 10px for meaningful body text; 10px may only appear in non-critical counters or dense chips with adequate contrast.

### Named Rules
**The Product Sans Rule.** Do not introduce display fonts into labels, buttons, table-like rows, or metadata.

**The Small-Type Contrast Rule.** Metadata can be small, but it must remain legible against white and paper-gray surfaces. Muted text should stay closer to ink than to the border color.

## 4. Elevation

The system uses flat tonal layers. Depth is conveyed through background changes, 1px borders, active fills, and focus rings—not shadows. At rest, cards and rows sit on Pure Desk or Cool Paper with Fine Divider Gray edges; hover states shift border or background tone just enough to confirm interactivity.

### Named Rules
**The Flat Workbench Rule.** Surfaces are flat by default. Do not add ambient card shadows, glass blur, or wide soft glows to make the UI feel premium.

**The State Creates Depth Rule.** If an element needs to feel closer, use state: hover border, focus ring, active fill, selected background. Never add decorative elevation to passive content.

## 5. Components

Components are compact and structural. They should look like durable controls in a research workbench: restrained radius, clear hit targets, consistent icon scale, and predictable hover/focus states.

### Buttons
- **Shape:** Gently curved rectangles: 8px radius for primary, 6px for compact secondary controls.
- **Primary:** Workbench Cobalt fill with white text, medium/semi-bold weight, 8–10px vertical / 12–16px horizontal padding.
- **Hover / Focus:** Hover uses a small background or border shift. Focus uses the cobalt ring and must be visible on light and dark surfaces.
- **Secondary / Ghost:** White or transparent controls with Fine Divider Gray borders and Muted Slate text. Hover should shift to Cool Paper and Blue-Black Ink.

### Chips
- **Style:** Neutral, pill-shaped tags with Cool Paper or Pure Desk fill, Fine Divider Gray border, small text, and compact horizontal padding.
- **State:** Tags act as dense filters and metadata markers. Use selected state only when a filter is actively applied; otherwise keep them quiet.
- **Source Type:** Source labels are metadata, not status. Prefer neutral mono chips such as `ARTICLE`, `YT`, `PDF`, `MD`, `TEXT`; do not assign each source a separate color family.

### Cards / Containers
- **Corner Style:** 6–8px radius, never oversized.
- **Background:** Pure Desk for primary content, Cool Paper for secondary groupings.
- **Shadow Strategy:** No ambient shadows; see Elevation.
- **Border:** 1px Fine Divider Gray. Use border tone changes on hover instead of lift.
- **Internal Padding:** 12px for compact rows/stats, 16px for document cards and cluster panels.

### Inputs / Fields
- **Style:** Pure Desk or Cool Paper fill, 1px input border, 8px radius, 14px text.
- **Focus:** Border shifts to Workbench Cobalt with a 1px focus ring.
- **Error / Disabled:** Error text and border should use Danger Red; disabled states should reduce opacity without losing label readability.

### Navigation
- **Desktop Rail:** A dark graphite rail is approved. It may be compact (64px) when the workspace provides clear page context and the nav icons have accessible labels/titles. For first-run clarity, prefer a visible brand mark that is not emoji and tooltips that are readable on focus and hover. A future expanded 224px sidebar is allowed but not required.
- **Mobile Nav:** Bottom navigation with visible labels is approved. Keep targets at least 44px high and preserve safe-area padding.
- **Active State:** Active rail item uses cobalt fill or cobalt-tinted treatment with high-contrast foreground.

### Quick Actions / Command Surfaces
If the top action panel is a static list, call it **Quick Actions** and use accurate shortcut hints only. If it is called a command console or command palette, it must support real command behavior: typing/filtering, arrow navigation, Enter to run, and complete shortcut documentation.

### Document Rows and Cards
Document rows and cards are the signature content primitives. Rows are for fast scan and queue management; cards are for search results and richer summaries. Both must preserve the same title → source/date → tags order so users can move between Browse, Library, and Search without relearning the hierarchy.

## 6. Do's and Don'ts

### Do:
- **Do** preserve the light workbench/dark rail system: white and cool paper surfaces, graphite frame, fine dividers, and cobalt only for action/state/focus.
- **Do** keep action placement obvious: Search, Ask, Add, Open, read/unread, and source-link affordances should be visible without decorative framing.
- **Do** use skeletons or inline progress for loading content-heavy regions; spinners alone are only acceptable for brief isolated transitions.
- **Do** make keyboard focus visible with the existing cobalt ring vocabulary.
- **Do** keep metadata concise and aligned with document scanning: type, score, date, tags, read state.
- **Do** provide recovery for high-frequency triage actions such as marking read/unread.

### Don't:
- **Don't** use SaaS marketing clichés: glossy gradients, hero metrics, decorative card grids, glassmorphism, oversized empty-state illustrations, or promotional page tropes.
- **Don't** add gradient text, glass cards, ambient shadows, diagonal stripe backgrounds, sketchy SVG illustrations, or side-stripe card accents.
- **Don't** over-round product surfaces. Cards and inputs top out at 8px in the current system; pills are only for tags and tiny keyboard hints.
- **Don't** spend semantic colors on decoration. Amber, green, and red communicate state only.
- **Don't** use emoji as the product mark in production UI.
- **Don't** call a static link list a command console.
- **Don't** introduce display typography into app-shell labels, buttons, filters, rows, or metadata.
