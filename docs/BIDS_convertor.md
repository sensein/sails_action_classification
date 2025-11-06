## BIDS Format

For reproducibility, organization, and practicality, sailsprep converts its raw data into the BIDS (Brain Imaging Data Structure) format.
BIDS is a community-driven standard for organizing, naming, and describing neuroimaging and related data (e.g., EEG, fMRI, MEG, behavioral, physiological data, etc.).

During the BIDS conversion pipeline, the raw domestic videos are preprocessed to be standardized, denoised, and reformatted.
Relevant metadata and annotations necessary for downstream analysis are also extracted at this stage.

## Structure

The final BIDS dataset follows the structure below:
```graphql
├── sub-ID1         # Contains raw videos in BIDS format
│   ├── ses-01          # Videos between 12 and 16 months
│   │   └── beh                 # Behavioral data
│   │        ├── sub-ID1_ses-01_task-A_run-01_beh.mp4   # Standardized raw video
│   │        ├── sub-ID1_ses-01_task-A_run-01_beh.tsv   # Manual annotations
│   │        └── sub-ID1_ses-01_task-A_run-01_beh.json  # Info on standardization
│   └── ses-02          # Videos between 34 and 38 months
│       └── beh
├── derivatives
│   └── preprocessed # Contains stabilized, denoised, standardized videos
│       ├── sub-ID1
│       │   ├── ses-01
│       │   │   └── beh
│       │   │        ├── sub-ID1_ses-01_task-A_run-01_audio.json              # Audio extraction info
│       │   │        ├── sub-ID1_ses-01_task-A_run-01_audio.wav               # Extracted audio
│       │   │        ├── sub-ID1_ses-01_task-A_run-01_desc-processed.json     # Video preprocessing info
│       │   │        └── sub-ID1_ses-01_task-A_run-01_desc-processed_beh.mp4  # Preprocessed video
│       │   └── ses-02
│       └── sub-ID2
├── README.md                   # Explains dataset structure and content
├── participants.tsv            # Participant information (e.g., ASD status)
├── participants.json           # Metadata for participants.tsv
└── dataset_description.json    # BIDS dataset description (name, version, etc.)
```
## Execution

To verify that FFmpeg is correctly installed (cf [README.md](../README.md)) and at least version 6.0, run:

```
ffmpeg -version
```

You’ll need to submit the conversion job on Engaging using sbatch.
Make sure you are in the root directory of the repository.

We provide SLURM submission scripts for convenience — simply run the following commands (with the miniforge module deactivated to ensure the correct FFmpeg version is used):
```
jid=$(sbatch --parsable jobs/run_bids_convertor.sh)
sbatch --dependency=afterok:$jid jobs/merge_cleanup.sh
```
