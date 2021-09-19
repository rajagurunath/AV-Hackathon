## Analytics Vidhya Jobathon

### Sales prediction

### Aproach taken so far

- #### Ensembler  for store_id:
    Built model for every store (totally 365 models), appends and submits a prediction

- #### Ensembler for store_type,region,location:
    Built models for different categorical column values and submits the results

- #### Model Blending :
   combines the ensemble results of all the above Ensembler(option 2) using operation like mean or min or max


### Please refer to following github (will be made public once the compition was completed) 

[Github Location ](https://github.com/rajagurunath/AV-Hackathon)

## MLOps
- Added Github Action for Evaluating and Training the models
- Produces Reports
- Create Experiments using new feature branch
- Trains, Evaluates and produces the reports
- Can compare Reports across features

Workflow will be triggered for every push/pull_request, so please have look at the pull_requests and 
[commits](https://github.com/rajagurunath/AV-Hackathon/commits/main)  for the various results across 
experiments and models

Happy to know your feedback/Suggestion

