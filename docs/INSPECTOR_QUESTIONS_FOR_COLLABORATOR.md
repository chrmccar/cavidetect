# Inspector — questions for the person using the tool

The new **🔍 Inspect** view (v0.6.0) is a first pass. These are the things we guessed at and
would like your answer on, so the next revision matches how you actually work. Answer as many
as you like — even rough answers help. (Phase 2 will add tracing/correcting conduits and
re-running; several questions below feed that.)

## The picture itself
1. **The middle "difference" panel** currently shows the change *between the frames you step
   through*. Is that the signal you want there, or would you rather see change **from the first
   (wet) frame**, or **from the final (dry) frame**? (They emphasise different things.)
2. **Colour-by-time ramp.** Onset is coloured blue→red (early→late). Do you prefer a specific
   LUT you already use (e.g. the ImageJ/OSOV "fire" or a rainbow)? Any colour-blindness to design
   around?
3. **Background of the conduits panel.** Pure black like the OSOV video, or should
   non-embolised rhizoids show faintly for context (so you can see what *didn't* fill)?
4. **Vertical stack vs side-by-side.** The three panels are stacked (like the video). On your
   monitor, would side-by-side be easier? What's your screen size/shape?

## "Which tube is which"
5. When you say you want to tell conduits apart, is a "tube" **one whole rhizoid strand
   end-to-end**, or **each segment between branch points**? Where bundles cross, should they be
   **one colour** or **split** at the crossing?
6. In **by-tube** mode, is a distinct colour per tube enough, or do you want **hover-to-highlight
   a tube and read its ID + onset time + area**?
7. Should the by-tube view colour **only embolised conduits**, or **all rhizoids** (filled or
   not) so you can see which never embolised?

## Time
8. **Real calibration** — what is the **µm per pixel** for your sequences (from ImageJ)? We're
   still using a 2.5 fallback; nothing physical is trustworthy until this is right.
9. **Frame interval** — seconds between frames, and is it constant? We can then label time in
   minutes/seconds instead of frame numbers (and later line it up with water potential).
10. Should onset reveal be **cumulative** (a conduit lights up at its onset and *stays* lit, as
    now) or **flash at onset** then fade? Does playback speed matter?

## The correction loop (Phase 2 — coming next)
11. Which corrections do you do **most**: adding missed faint tubes, removing false positives
    (scales / tape / debris), **splitting** a merged bundle, **merging** fragments, or fixing
    **onset timing**? (This sets what we build first.)
12. When you "re-run until it's satisfactory," what's your **acceptance test** — it matches your
    eye, it matches a hand-drawn mask, or the embolised-area curve looks physically right?
13. Should a correction be **per-sequence only**, or do you want to **save and reuse** one — e.g.
    mask the same scale once and apply it across all crops of a plant?
14. Do you need to **export** the inspector (a PNG still or an MP4/GIF of the animation) for
    figures or talks?

## Anything else
15. What's the **first thing you'd try to do** when you open Inspect on one of your own
    sequences — and what would make you trust (or distrust) what it shows?
