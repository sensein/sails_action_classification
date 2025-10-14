# BIDS Conversion and Preprocessing

These files convert the SAILS home videos collection into a standardized BIDS-compliant dataset.

## `bids.py`

This is the main executable that performs the conversion and processing for a single video; it's called with task IDs from the SLURM schedule to process 
a specific chunk of the total video dataset. It discovers video files and attaches IDs to them, integrates behavioral metadata, creates a BIDS directory structure
 with `/sourcedata` and `/derivatives`, performs video preprocessing with ffmpeg (stabilization, denoising, standardization to 720p), extracts audio, and generates 
 required BIDS metadata in .json and .tsv files. 


## `log_file.ipynb`

After the `bids.py` job array is complete, this file mergers the individual log files from each job (`processing_log.json` and `not_processed.json`) into a summary 
and provides statistics on the number of processed and failed videos and other information and summaries of errors.

## `submit_bids.sh`

This is a SLURM batch script to submit `bids.py` as a job and manages parallel execution of the pipeline - run with `sbatch submit_bids.sh`.

## `data_copy_to_bids.ipynb`

This file populates a `participants.tsv` file with participant IDs using information from the SAILS xlsx data; 
age, session ID, date, duration, data validity score, etc. are used to populate `participants.tsv`.
