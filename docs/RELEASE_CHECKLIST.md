# Public-release checklist

Before making the repository public:

- [ ] Confirm that this exact code version generated the manuscript's final reported results.
- [ ] Confirm that every post-processing rule affecting detections is described in Methods or Supplementary Methods.
- [ ] Confirm the manuscript's timestamp/VFR description matches the implemented logic.
- [ ] Decide whether all study ROI presets should remain public.
- [ ] Remove any study-specific filenames, local paths, credentials, or private data if additional files are added.
- [ ] Add a software license.
- [ ] Update `CITATION.cff` with the final article title and DOI when available.
- [ ] Add the repository URL to the manuscript's code/data-availability statement.
- [ ] Test installation in a clean environment.
- [ ] If known, pin the exact package versions used for the final analysis.
- [ ] Add a small shareable test input or derived signal plus expected output if permitted.
- [ ] Create a tagged release corresponding to the manuscript version (for example `v1.0.0`).
- [ ] Archive the tagged release in a long-term software repository if required by the journal/institution.
