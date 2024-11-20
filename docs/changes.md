# What's New

## 1.0.0rc1

This is a prerelease for the version that will be used by the HBCD project for their first data release.

We plan to not introduce any backwards-incompatible changes for a while.

### 🛠 Breaking Changes

* Rename QSIPrep to QSIRecon by @tsalo in https://github.com/PennLINC/qsirecon/pull/1
* Start removing QSIPrep-specific code and documentation by @tsalo in https://github.com/PennLINC/qsirecon/pull/4
* Remove QSIPrep-specific parameters and Config elements by @tsalo in https://github.com/PennLINC/qsirecon/pull/6
* Drop Docker wrapper by @tsalo in https://github.com/PennLINC/qsirecon/pull/52
* Restructure outputs into BIDS datasets by @tsalo in https://github.com/PennLINC/qsirecon/pull/66
* Combine `mfp` and `mdp` entities into single `param` entity by @tsalo in https://github.com/PennLINC/qsirecon/pull/72
* Replace `--recon-input-pipeline` with `--input-type` by @tsalo in https://github.com/PennLINC/qsirecon/pull/68
* Remove unused `--longitudinal` argument by @tsalo in https://github.com/PennLINC/qsirecon/pull/88
* Make all params and models lower-case by @tsalo in https://github.com/PennLINC/qsirecon/pull/90
* Reorganize atlas management based on XCP-D and BIDS-Atlas by @tsalo in https://github.com/PennLINC/qsirecon/pull/123
* Rename `--freesurfer-input` to `--fs-subjects-dir` by @tsalo in https://github.com/PennLINC/qsirecon/pull/152
* [ENH] Read session-specific anat data by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/151
* Remove underscore versions of all parameters by @tsalo in https://github.com/PennLINC/qsirecon/pull/159
* Make connectivity field names compatible with MATLAB by @tsalo in https://github.com/PennLINC/qsirecon/pull/166

### 🎉 Exciting New Features

* Add ng, perng, parng, and mapcoeffs to Dipy MAPMRI outputs by @tsalo in https://github.com/PennLINC/qsirecon/pull/55
* Add DatasetLinks to dataset_description.json by @tsalo in https://github.com/PennLINC/qsirecon/pull/77
* Pass DWI file metadata to reconstruction workflows by @tsalo in https://github.com/PennLINC/qsirecon/pull/154
* Add HBCD Release1 recon workflow by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/169

### 🐛 Bug Fixes

* Write out QSIRecon pipeline-wise reports by @tsalo in https://github.com/PennLINC/qsirecon/pull/7
* Write out reports to individual reconstruction derivative folders by @tsalo in https://github.com/PennLINC/qsirecon/pull/53
* Add summary reportlets to HTML report by @tsalo in https://github.com/PennLINC/qsirecon/pull/61
* Fix QSIRecon Entrypoint by @smeisler in https://github.com/PennLINC/qsirecon/pull/76
* Use file lock to avoid concurrent edits to dataset_description.json by @cookpa in https://github.com/PennLINC/qsirecon/pull/91
* Compress the tck files by @tsalo in https://github.com/PennLINC/qsirecon/pull/93
* Add missing filename patterns by @tsalo in https://github.com/PennLINC/qsirecon/pull/131
* Fix infant anatomical ingression by @tsalo in https://github.com/PennLINC/qsirecon/pull/126
* Fix extension in ds_fs_5tt_hsvs by @tsalo in https://github.com/PennLINC/qsirecon/pull/165
* Fix extension in ds_qsiprep_5tt_hsvs by @tsalo in https://github.com/PennLINC/qsirecon/pull/172
* Fix space in ds_qsiprep_5tt_hsvs by @tsalo in https://github.com/PennLINC/qsirecon/pull/175
* Make PlotPeaks robust enough that we don't need --writable-tempfs in singularity/apptainer by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/174

### Other Changes

* Remove internal version of LiterateWorkflow by @tsalo in https://github.com/PennLINC/qsirecon/pull/8
* Drop and modify infrastructure files by @tsalo in https://github.com/PennLINC/qsirecon/pull/40
* Remove QSIPrep-specific tests by @tsalo in https://github.com/PennLINC/qsirecon/pull/5
* Drop unused modules, classes, and functions by @tsalo in https://github.com/PennLINC/qsirecon/pull/11
* [CI] Speed up pyafq test by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/57
* [DOCS] Reorganize documentation by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/56
* Collect templates from TemplateFlow instead of package data by @tsalo in https://github.com/PennLINC/qsirecon/pull/49
* Remove unused data files by @tsalo in https://github.com/PennLINC/qsirecon/pull/63
* Replace hyperlinks with BibTeX references by @tsalo in https://github.com/PennLINC/qsirecon/pull/67
* [DOCS] Add scalar tables by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/80
* Move scalar file configs to YAMLs and use a recon scalar workflow by @tsalo in https://github.com/PennLINC/qsirecon/pull/79
* [WIP] Make a system for keeping docs and file names in sync by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/86
* Convert pipeline configs from JSON to YAML by @tsalo in https://github.com/PennLINC/qsirecon/pull/84
* [DOCS] clean up by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/101
* Draft pre-init ingression by @tsalo in https://github.com/PennLINC/qsirecon/pull/102
* [ENH] Update AMICO by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/115
* [FIX] mif2fib by @ameliecr in https://github.com/PennLINC/qsirecon/pull/116
* Fix how bundles to be tracked are selected for DSIstudio autotrack by @ameliecr in https://github.com/PennLINC/qsirecon/pull/121
* [FIX] workflow connection in dsi_studio_gqi by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/142
* [CI]  add hsvs test by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/136
* [FIX] dsi_studio_gqi workflow connect by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/146
* [DOC] Minor addition for skipping connectivity matrices step by @chiuhoward in https://github.com/PennLINC/qsirecon/pull/150
* Add page documenting output structure by @tsalo in https://github.com/PennLINC/qsirecon/pull/156
* Add information about QSIPrep/QSIRecon releases by @tsalo in https://github.com/PennLINC/qsirecon/pull/158
* [CI] Add tests for reading pre-1.0 outputs from qsiprep by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/155
* [ENH] Support outputs from qsiprep 1.0.0rc0 by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/160
* [FIX] get rid of logging error #145 by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/161
* Add UKB ingression by @smeisler in https://github.com/PennLINC/qsirecon/pull/147
* [FIX] set TMPDIR before running plot_peaks by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/163
* [ENH] match to qsiprep nipype/nireports versions by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/167
* Add in Val's text for connectivity matrices by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/173
* Disable test_main in the CI integration tests by @mattcieslak in https://github.com/PennLINC/qsirecon/pull/177

### New Contributors

* @tsalo made their first contribution in https://github.com/PennLINC/qsirecon/pull/1
* @mattcieslak made their first contribution in https://github.com/PennLINC/qsirecon/pull/57
* @cookpa made their first contribution in https://github.com/PennLINC/qsirecon/pull/91
* @dependabot made their first contribution in https://github.com/PennLINC/qsirecon/pull/100
* @ameliecr made their first contribution in https://github.com/PennLINC/qsirecon/pull/116
* @chiuhoward made their first contribution in https://github.com/PennLINC/qsirecon/pull/150

**Full Changelog**: https://github.com/PennLINC/qsirecon/commits/1.0.0rc1
