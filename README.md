## Analytics Vidhya Jobathon

### Sales prediction

### Aproach taken so far

- #### Ensembler  for store_id:
    Built model for every store (totally 365 models), appends and submits a prediction

- #### Ensembler for store_type,region,location:
    Built models for different categorical column values and submits the results

- #### Model Blending :
   combines the ensemble results of all the above Ensembler using operation like mean or min or max

- #### Model combiner:
    Got Crazy and build multiple models for each store using custom defined class called `combiner`
   
### Custom classes Developed for this Hackathon

`scripts/models.py`:

`Ensembler` :
    - Takes model and categorical column unique values and builds one model for each of the categorical column
    - This Class has train,eval, predict functions are used to produce submission and reports (which is then accumented in the github actions)

`combiner`:
    - The input for above ensembler but instead of building single model builds list of model for each of categorical column's unique values.

`Bender`:
   - Combines the ensemble of store_id,type,location,region and make a single submission file.
       
`scripts\features` :
    - LabelEncoder, onehotEncoder, datetime features

### Usage:

- install requirements in virtualenv 
```
virtualenv -p python3 venv
source venv/bin/activate
```
- evaluate & Generate for  models

```
cd scripts/
python main.py --eval

# for plotting (model error for each model of the ensemble)
python main.py --eval --plot-eval
```
- Train models and produce submission file

```
cd scripts/
python main.py
```



## Lesson learnt: 
Used almost Every model [main.py](https://github.com/rajagurunath/AV-Hackathon/blob/feature/decisiontree/scripts/main.py#L9-L18) 😅 but 
unable to reduce the mean squared log error after some saturation point, which always hits me hard 
with brick 🧱 concenrate more on data part instead of fancy algorithms
and packages
    
As side note : Also tried `h2o`,`PyCaret`,`sktime`, `torch` (little bit) in the notebooks, but again the lesson is spent more time on Data 🧱.


## Good News:

- Inspite of concentrating only on Algorithm part and not much on Data part, got some 120th Rank on public leaderboard, 
  and expected to have Rank in the range of above 200 in Private leaderboard, but to my surprise this code base managed to achieve 
  **41th Rank** in [Private leaderboard](https://datahack.analyticsvidhya.com/contest/job-a-thon-september-2021/?utm_source=datahack&utm_medium=flashstrip&utm_campaign=jobathon#LeaderBoard) 🥳 🥳 🕺 (SomeWhat felt good)

### MLOps

Wanted to try [CML](https://cml.dev/) for long time used this hackathon as a chance to setup end to end training, 
building and comparing reports across branches in pull request.

- Added Github Action for Evaluating and Training the models
- Produces Reports
- Create Experiments using new feature branch
- Trains, Evaluates and produces the reports
- Can compare Reports across features


experiments and models
=======
To view the example reports please have look at all the [pull request](https://github.com/rajagurunath/AV-Hackathon/pulls?q=) and [commit's history](https://github.com/rajagurunath/AV-Hackathon/commits/main)

Once the setup was done, No other work, other than experimenting with Data and Different Models 
Hyperparameters and one more important thing `please experiment more with data here after` 🧱



Happy to know your feedback/Suggestion

