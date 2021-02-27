# Visual Analytics - Spring 2021

This repository contains all of the code and data related to the Spring 2021 module _Visual Analytics_ as part of the bachelor's tilvalg in [Cultural Data Science](https://bachelor.au.dk/en/supplementary-subject/culturaldatascience/) at Aarhus University.

The main course repository is managed by [Ross](https://pure.au.dk/portal/en/persons/ross-deans-kristensenmclachlan(29ad140e-0785-4e07-bdc1-8af12f15856c).html), and can be found [here](https://github.com/CDS-AU-DK/cds-visual).

## Technicalities and Reproducability

All notebooks and scripts contained in this repository are coded and thus depend on requirements defined in the `requirements.txt` file. These can be installed by reproducing the virtual environment `cv101`. To clone the repository and create the venv `cv101`, you can run the following commands in your terminal:

```bash
# clone repository into cds-visual-nd
git clone https://github.com/nicole-dwenger/cds-visual.git cds-visual-nd

# move into directory
cd cds-visual-nd

# create cv101
bash create_vision_venv.sh

# activate cv101 to run the scripts
source cv101/bin/activate
```


## Repo structure

This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data```| A folder to be used for sample datasets that we use in class.
```notebooks``` | This folder contains exploratory and experimental notebooks.
```src``` | Python scripts to be used in class.
```utils``` | Utility functions that are written by Ross, and which we'll use in class.
```assignments``` | Folder structured by assignments, containg scripts and descriptive README files 



## Course overview and readings

A detailed breakdown of the course structure and the associated readings can be found in the [syllabus](syllabus.md). Information about examination and academic regulations can be found in the [_studieordning_](https://eddiprod.au.dk/EDDI/webservices/DokOrdningService.cfc?method=visGodkendtOrdning&dokOrdningId=15952&sprog=en).
