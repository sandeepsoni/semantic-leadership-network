For running the scripts in this directory, it is assumed that we have *cleaned* data.

This means:

- The data is whitespace corrected and organized into directories.
  - This part of the pipeline is executed on `conair` using scripts in `/hg190/sandeep/projects/aa-digital-cleaning`
  - Particularly see `make_aa_clean.sh`

- A compiled list of 8-gram files to deduplicate is available.
  - This part of the pipeline is also executed on `conair` using scripts in `/hg190/sandeep/projects/aa-digital-cleaning`
  - Particularly see `ngram_based_duplicate_detection.sh`
  - This generated file must be present in `annotations` directory (e.g. `annotations/all.1.ignore`)

- A replica of the scripts from `conair` is also present in `adaptation` (just replace hg190 by hg191) but I'm not fully
  confident that everything is in sync.

