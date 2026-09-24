# React Bits attribution

VoxSentinel uses a small local adaptation of the React Bits `ShinyText` pattern
for the one-pass VoxSentinel wordmark sheen.

- Project: React Bits
- Repository: https://github.com/DavidHDev/react-bits
- Source revision checked: `c5df8610c0b47d7cd805cda480baba402f7267c1`
- Revision checked: 2026-09-23
- License: MIT + Commons Clause License Condition v1.0
- Source reference: `src/ts-default/TextAnimations/ShinyText/ShinyText.tsx`
  and `ShinyText.css`
- Local modifications: CSS-only, one-pass motion; no `motion/react` or
  requestAnimationFrame loop; added `prefers-reduced-motion` behavior and
  VoxSentinel color tokens.

The copied/adapted pattern remains part of the VoxSentinel application and is
not distributed as a standalone component library.

React Bits `DotGrid`, `GridMotion`, and `SpotlightCard` were reviewed and not
used because their continuous canvas, pointer, or GSAP activity was not a good
fit beside CPU-bound live AASIST/ECAPA inference.
