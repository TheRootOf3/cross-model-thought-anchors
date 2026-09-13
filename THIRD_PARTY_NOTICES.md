# Third-party code, data and fonts

## Code copied from Thought Anchors and Thought Branches

Parts of this repository are copied, verbatim or with small documented changes, from

- `interp-reasoning/thought-anchors` — https://github.com/interp-reasoning/thought-anchors
- `interp-reasoning/thought-branches` — https://github.com/interp-reasoning/thought-branches

Each such function carries a header comment naming the upstream file and line range, and the
commit it was taken from. Copying rather than reimplementing was deliberate: the point of this
work is comparability with those papers, so the measurement code has to be theirs.

Affected files: `src/common/{answers,filters,split,data,prompts,mcq,stats}.py`,
`src/phase2/{chunk_ranges,prompts_p2}.py`, and the routines cited in
`scripts/{01,02,03,11,12,13,90}_*.py`.

Both upstream repositories are MIT licensed:

```
MIT License

Copyright (c) 2025 interp-reasoning

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Data

- **MATH rollouts** — `uzaymacar/math-rollouts` on Hugging Face, released with Thought Anchors.
  Not redistributed here; the README gives the download command.
- **Hinted multiple-choice traces** — the `good_problems` selection released with Thought Branches,
  itself derived from the faithfulness results of Chua & Evans. Read from the upstream repository,
  not copied into this one.

## Fonts

`plots/fonts/Inter-*.ttf` — Inter by Rasmus Andersson, SIL Open Font License 1.1
(https://github.com/rsms/inter). See `plots/fonts/README.md`.
