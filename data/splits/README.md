# Dataset Split Metadata

This folder is reserved for small dataset split metadata files.

Raw images and masks are not stored in this repository.

Recommended files that may be generated locally:

- aptos_train.csv
- aptos_val.csv
- aptos_test.csv
- idrid_train.csv
- idrid_test.csv

Each split file should contain image identifiers, labels, and relative paths only.

Do not commit raw images, masks, checkpoints, or large generated artifacts here.

This folder exists to document the expected reproducibility structure without storing the datasets themselves.
