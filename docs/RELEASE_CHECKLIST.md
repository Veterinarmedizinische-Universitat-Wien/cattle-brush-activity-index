# Public-release checklist

Before the final public/software release:

- [ ] Confirm that this exact detector version generated the manuscript's final reported results.
- [ ] Confirm that moving configuration and I/O helpers into `config.py` / `io_utils.py` did not change detector outputs.
- [ ] Confirm that every post-processing rule affecting detections is described in Methods or Supplementary Methods.
- [ ] Confirm the manuscript timestamp/VFR description matches the implemented logic.
- [ ] Confirm the README timing hierarchy (DVR -> filename -> standalone relative) matches the tested release.
- [ ] Decide whether all study ROI presets should remain public.
- [ ] Replace local example paths in `src/config.py` with repository-relative defaults if desired for the final public release.
- [ ] Keep raw/example and annotated video files out of Git unless a deliberately small redistributable sample is chosen.
- [ ] Add the university/institutional data DOI or URL when the study data are deposited.
- [ ] If used, add the annotated demonstration video as manuscript supplementary material and cite it consistently.
- [ ] Add a software license.
- [ ] Update `CITATION.cff` with the final article title, DOI, and archived software DOI when available.
- [ ] Test installation and one complete analysis in a clean environment.
- [ ] If known, pin the package versions used for the final analysis.
- [ ] Create a tagged release corresponding to the manuscript version.
- [ ] Archive the tagged release in a long-term software repository if required by the journal/institution.
