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

### MLOps

Wanted to try [CML](https://cml.dev/) for long time used this hackathon as a chance to setup end to end training, 
building and comparig reports across branches in pull request.

- Added Github Action for Evaluating and Training the models
- Produces Reports
- Create Experiments using new feature branch
- Trains, Evaluates and produces the reports
- Can compare Reports across features

To view the example reports please have look at all the [pull request](https://github.com/rajagurunath/AV-Hackathon/pulls?q=) and [commit's history](https://github.com/rajagurunath/AV-Hackathon/commits/main)

Once the setup was done, No other work other than experimenting with Data and Different Models and Hyperparameters



Happy to know your feedback/Suggestion

