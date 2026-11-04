# Design — RecoEngine UI/UX & Visual Design System

**Version:** 1.0 · **Created:** 2026-11-04 · Scope: the Streamlit demo and the visuals used in the README/presentation. There is no consumer product UI beyond this.

---

## 1. Design Direction

- **Style:** clean, data-forward, "ML dashboard meets streaming app." Dark theme by default, light theme supported.
- **Personality:** precise, transparent, credible. The UI explains *why* a result appears (source, score, latency).
- **Philosophy:** show the system's mechanics — evaluators care about how it works, not just that it does.
- **Emotional response:** "this is real and well-engineered."
- **No external images:** no posters or metadata APIs. Cards use genre-based color accents instead.

## 2. Color System

| Role | Dark | Light | Notes |
|---|---|---|---|
| Background | `#0F1216` | `#F7F8FA` | page |
| Surface | `#171C22` | `#FFFFFF` | cards, panels |
| Border | `#2A313A` | `#DDE1E7` | |
| Text | `#E8EBEF` | `#16191D` | ≥ 4.5:1 on surface |
| Muted text | `#9AA4B0` | `#5A6572` | ≥ 4.5:1 on surface |
| Primary | `#4C8DFF` | `#1F5FD1` | buttons, links, variant B |
| Secondary | `#8B7CFF` | `#5B4BD6` | variant A, secondary accents |
| Accent | `#2DD4BF` | `#0F9E8D` | highlights, cache-hit badge |
| Success | `#3DDC84` | `#1B8A4B` | |
| Warning | `#FFB84D` | `#B26A00` | fallback badge |
| Error | `#FF6B6B` | `#C62D2D` | |

Verify contrast with a checker before finalizing. Never rely on color alone: every status also has a text label.

**Genre accent colors** (card left-border only, decorative): Action `#FF6B6B`, Comedy `#FFB84D`, Drama `#8B7CFF`, Sci-Fi `#4C8DFF`, Horror `#C2415D`, Romance `#F472B6`, Documentary `#2DD4BF`, Other `#9AA4B0`.

**Source badges:** `retrieved+ranked` (primary), `retrieval-only` (secondary), `fallback: popularity` / `fallback: content` (warning), `cache hit` (accent), `served locally` (muted, shown when Triton was unavailable).

## 3. Typography

- **Family:** Inter (UI), JetBrains Mono (numbers, IDs, latency). Fallback: system sans / monospace.
- Hierarchy: H1 28/36 semibold · H2 22/30 semibold · H3 18/26 medium · Body 15/24 regular · Label 13/18 medium · Caption 12/16 regular · Button 14/20 medium.
- Numbers (scores, latency, metrics) always monospace, right-aligned in tables.

## 4. Spacing

4-pt base scale: 4 · 8 · 12 · 16 · 24 · 32 · 48. Card padding 16; gap between cards 12; section gap 32. Radius: 8 (inputs/buttons), 12 (cards).

## 5. Layout

**Page structure (Streamlit wide mode)**
- **Sidebar (controls):** user picker (ID input + "random warm user" + "new user" toggle), seed genres (shown for new user), k slider (5–20), variant selector (A / B / Compare), "bypass cache" toggle, service status indicators (API, Redis, Triton).
- **Main area, tabs:**
  1. **Recommend** — result cards; in Compare mode two columns (A vs B) with overlap highlighted.
  2. **User history** — the user's recent liked items (explains the recommendations).
  3. **Results** — metrics tables and charts from `reports/` (baselines vs retrieval vs ranker; cold vs warm; A/B with confidence intervals; latency).
  4. **System** — architecture diagram, model version, index info, DAG last-run status.

**Responsive:** Desktop (≥1024px): sidebar + 2-column compare. Tablet: sidebar collapses; compare stacks to tabs. Mobile (≤640px): single column, cards full-width (best effort — demo is desktop-first).

## 6. Components

- **Recommendation card:** rank number, title, year, genre chips, score (mono), source badge, optional "why" line (e.g., "similar to your liked: …"). Genre accent on the left border.
- **Metric card:** label, large number, delta vs baseline (▲/▼ with text, not color only).
- **Badges/chips:** pill shape, 12px text, icon + label.
- **Tables:** sticky header, zebra rows optional, numeric columns right-aligned; best value per column in bold.
- **Charts:** simple bar/line, labeled axes, colorblind-safe palette (primary vs secondary), confidence-interval error bars on A/B.
- **Buttons:** primary (filled), secondary (outline). **Inputs:** labeled, with help text and validation message.
- **Alerts:** info / warning / error banners with icon + text (e.g., "Triton unavailable — served locally").
- **Loading state:** skeleton cards (5) while the API responds; spinner for long jobs only.
- **Empty state:** "No recommendations yet — pick a user or choose New user."
- **Error state:** message in plain language + what the user can try; technical detail in an expandable "Details."

## 7. Interaction Design

- **Hover:** card lifts 2px with subtle border highlight (150 ms ease-out).
- **Focus:** 2px visible outline in primary color with 2px offset on every interactive element.
- **Active:** slight darkening / 1px press.
- **Loading:** show skeletons; display latency badge once complete.
- **Transitions:** only for hover/focus and tab change (≤ 200 ms). No decorative animation.
- **Validation:** inline, on change; invalid user ID → helpful message and fallback suggestion ("Unknown ID: showing new-user recommendations").
- **Feedback:** every request shows source, cache-hit/miss, latency, model version. Click on a card logs a simulated click event only in a clearly labeled "simulation" mode.

## 8. Accessibility

- Full keyboard navigation (tab order follows visual order).
- Text contrast ≥ 4.5:1; large text ≥ 3:1; UI components ≥ 3:1.
- Status never by color alone (icon + text).
- Alt text/captions for diagrams and charts; chart data also available as a table.
- Touch targets ≥ 44×44 px where applicable.
- Respect `prefers-reduced-motion` (disable hover lift and tab transitions).
- Streamlit limits: note any known accessibility gaps in README rather than hiding them.

## 9. Documentation Visuals (README & slides)

- **Architecture diagram:** same palette; offline flow top, online flow bottom; services as rounded boxes; Redis/Triton/FAISS/Airflow/Spark labeled with their logos only if licensing allows, otherwise text labels.
- **Results charts:** exported PNG at 2×; title states the takeaway ("Ranker improves NDCG@10 by X over retrieval-only"), numbers taken only from `reports/`.
- **Slides (10 min):** 1 idea per slide, ≤ 30 words, diagram and result slides carry the story.

## 10. Responsive Design Summary

Desktop-first demo; tablet/mobile are best-effort single-column fallbacks. The README and slides are the primary artifacts for non-desktop viewers, so they must stand on their own.
