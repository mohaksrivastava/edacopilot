# Eval label sample (for review)

Random sample of 30 items from each eval set (Section 15.4), drawn with `random.Random(20260928).sample(...)` -- seed `20260928` (today's date), stated here so the sample is reproducible.

These are ground truth, like the R fixtures: every label below was derived from the same template slots used to build its input text (`evals/build_eval_sets.py`), not asserted separately by hand. Full sets: `evals/intent_utterances.jsonl` (215 items), `evals/question_specs.jsonl` (156 items).

**No live API call has been made anywhere in building or sampling these.** This document is for review before any eval run.

---

## Intent parsing sample (30 of 215)

### #176

**Input:** `plot test_score against income`
**Label:** `{'type': 'show'}`

### #141

**Input:** `switch to the "alt-cleaning" branch`
**Label:** `{'type': 'switch_branch', 'name': 'alt-cleaning'}`

### #72

**Input:** `use alpha = 0.01`
**Label:** `{'type': 'modify', 'alpha': 0.01}`

### #177

**Input:** `plot weight against age`
**Label:** `{'type': 'show'}`

### #202

**Input:** `I don't know`
**Label:** `{'type': 'other'}`

### #194

**Input:** `paired`
**Context:** `{'awaiting_answer': True}`
**Label:** `{'type': 'answer_clarification', 'design': 'paired'}`

### #9

**Input:** `how are wait_minutes and satisfaction related?`
**Label:** `{'type': 'ask_analysis'}`

### #55

**Input:** `accept`
**Label:** `{'type': 'accept', 'persona': None}`

### #25

**Input:** `is there a trend in satisfaction across month?`
**Label:** `{'type': 'ask_analysis'}`

### #211

**Input:** `sure whatever`
**Label:** `{'type': 'other'}`

### #114

**Input:** `go to the profile stage`
**Label:** `{'type': 'goto_stage', 'target_stage': 'profile'}`

### #91

**Input:** `override to kendall_tau`
**Label:** `{'type': 'override', 'function': 'kendall_tau'}`

### #160

**Input:** `write this up as code`
**Label:** `{'type': 'export'}`

### #51

**Input:** `is weight higher in one treatment_group than another?`
**Label:** `{'type': 'ask_analysis'}`

### #182

**Input:** `show me the distribution of satisfaction`
**Label:** `{'type': 'show'}`

### #86

**Input:** `use student_t anyway`
**Label:** `{'type': 'override', 'function': 'student_t'}`

### #131

**Input:** `can you revert`
**Label:** `{'type': 'undo'}`

### #209

**Input:** `maybe`
**Label:** `{'type': 'other'}`

### #119

**Input:** `go to the export stage`
**Label:** `{'type': 'goto_stage', 'target_stage': 'export'}`

### #8

**Input:** `does age correlate with wait_minutes?`
**Label:** `{'type': 'ask_analysis'}`

### #67

**Input:** `go with maverick`
**Label:** `{'type': 'accept', 'persona': 'maverick'}`

### #172

**Input:** `plot income against satisfaction`
**Label:** `{'type': 'show'}`

### #212

**Input:** `huh`
**Label:** `{'type': 'other'}`

### #103

**Input:** `explain the Shapiro-Wilk test`
**Label:** `{'type': 'explain', 'topic': 'the Shapiro-Wilk test'}`

### #167

**Input:** `what have I done`
**Label:** `{'type': 'settings'}`

### #148

**Input:** `skip this`
**Label:** `{'type': 'skip'}`

### #174

**Input:** `plot wait_minutes against weight`
**Label:** `{'type': 'show'}`

### #136

**Input:** `create a branch called sensitivity-check`
**Label:** `{'type': 'branch', 'name': 'sensitivity-check'}`

### #26

**Input:** `has test_score changed over quarter?`
**Label:** `{'type': 'ask_analysis'}`

### #15

**Input:** `how are age and wait_minutes related?`
**Label:** `{'type': 'ask_analysis'}`

---

## QuestionSpec building sample (30 of 156)

### #141

**Input:** `are there outliers in reaction_time?`
**Label:** `{'goal': 'outliers', 'variables': {'outcome': 'reaction_time'}, 'design': 'unknown'}`

### #72

**Input:** `is the average income equal to 0?`
**Label:** `{'goal': 'distribution_fit', 'variables': {'outcome': 'income'}, 'design': 'unknown', 'reference_value': 0}`

### #9

**Input:** `is age different between clinic groups? these are unrelated, independent samples`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'age', 'group': 'clinic'}, 'design': 'independent'}`

### #55

**Input:** `is there a relationship between income and reaction_time?`
**Label:** `{'goal': 'association', 'variables': {'x': 'income', 'y': 'reaction_time'}, 'design': 'unknown'}`

### #25

**Input:** `wait_minutes was measured three times on the same subjects across treatment_group`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'wait_minutes', 'group': 'treatment_group'}, 'design': 'repeated'}`

### #114

**Input:** `give me an overview of reaction_time`
**Label:** `{'goal': 'describe', 'variables': {'outcome': 'reaction_time'}, 'design': 'unknown'}`

### #91

**Input:** `has wait_minutes changed over week?`
**Label:** `{'goal': 'trend', 'variables': {'outcome': 'wait_minutes', 'time': 'week'}, 'design': 'unknown'}`

### #51

**Input:** `is there a relationship between income and wait_minutes?`
**Label:** `{'goal': 'association', 'variables': {'x': 'income', 'y': 'wait_minutes'}, 'design': 'unknown'}`

### #86

**Input:** `is wait_minutes practically the same between treatment_group groups, within a small margin?`
**Label:** `{'goal': 'equivalence', 'variables': {'outcome': 'wait_minutes', 'group': 'treatment_group'}, 'design': 'independent'}`

### #131

**Input:** `are there outliers in age?`
**Label:** `{'goal': 'outliers', 'variables': {'outcome': 'age'}, 'design': 'unknown'}`

### #154

**Input:** `does weight need a transformation before modelling?`
**Label:** `{'goal': 'transform', 'variables': {'outcome': 'weight'}, 'design': 'unknown'}`

### #119

**Input:** `why is wait_minutes missing so often?`
**Label:** `{'goal': 'missingness', 'variables': {'outcome': 'wait_minutes'}, 'design': 'unknown'}`

### #8

**Input:** `is income different between region groups? these are unrelated, independent samples`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'income', 'group': 'region'}, 'design': 'independent'}`

### #67

**Input:** `is the average wait_minutes equal to 0?`
**Label:** `{'goal': 'distribution_fit', 'variables': {'outcome': 'wait_minutes'}, 'design': 'unknown', 'reference_value': 0}`

### #103

**Input:** `summarize age`
**Label:** `{'goal': 'describe', 'variables': {'outcome': 'age'}, 'design': 'unknown'}`

### #136

**Input:** `does satisfaction have any extreme values?`
**Label:** `{'goal': 'outliers', 'variables': {'outcome': 'satisfaction'}, 'design': 'unknown'}`

### #26

**Input:** `satisfaction was measured three times on the same subjects across department`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'satisfaction', 'group': 'department'}, 'design': 'repeated'}`

### #15

**Input:** `is income different between clinic groups? these are unrelated, independent samples`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'income', 'group': 'clinic'}, 'design': 'independent'}`

### #49

**Input:** `is satisfaction different by cohort?`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'satisfaction', 'group': 'cohort'}, 'design': 'ambiguous', 'note': 'wording does not say independent vs. paired/repeated'}`

### #70

**Input:** `is the average weight equal to 50?`
**Label:** `{'goal': 'distribution_fit', 'variables': {'outcome': 'weight'}, 'design': 'unknown', 'reference_value': 50}`

### #7

**Input:** `is reaction_time different between gender groups? these are unrelated, independent samples`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'reaction_time', 'group': 'gender'}, 'design': 'independent'}`

### #61

**Input:** `is there a relationship between wait_minutes and satisfaction?`
**Label:** `{'goal': 'association', 'variables': {'x': 'wait_minutes', 'y': 'satisfaction'}, 'design': 'unknown'}`

### #124

**Input:** `what's going on with the missing values in test_score?`
**Label:** `{'goal': 'missingness', 'variables': {'outcome': 'test_score'}, 'design': 'unknown'}`

### #74

**Input:** `is the average wait_minutes equal to 20?`
**Label:** `{'goal': 'distribution_fit', 'variables': {'outcome': 'wait_minutes'}, 'design': 'unknown', 'reference_value': 20}`

### #148

**Input:** `does wait_minutes need a transformation before modelling?`
**Label:** `{'goal': 'transform', 'variables': {'outcome': 'wait_minutes'}, 'design': 'unknown'}`

### #45

**Input:** `is reaction_time different by region?`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'reaction_time', 'group': 'region'}, 'design': 'ambiguous', 'note': 'wording does not say independent vs. paired/repeated'}`

### #54

**Input:** `is there a relationship between income and weight?`
**Label:** `{'goal': 'association', 'variables': {'x': 'income', 'y': 'weight'}, 'design': 'unknown'}`

### #46

**Input:** `is income different by clinic?`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'income', 'group': 'clinic'}, 'design': 'ambiguous', 'note': 'wording does not say independent vs. paired/repeated'}`

### #57

**Input:** `is there a relationship between age and satisfaction?`
**Label:** `{'goal': 'association', 'variables': {'x': 'age', 'y': 'satisfaction'}, 'design': 'unknown'}`

### #125

**Input:** `why is weight missing so often?`
**Label:** `{'goal': 'missingness', 'variables': {'outcome': 'weight'}, 'design': 'unknown'}`
