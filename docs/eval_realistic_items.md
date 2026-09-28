# Eval realistic items (for review)

Every hand-written realistic item in both eval sets -- 80 intent, 80 QuestionSpec, not a sample. Each label here is a judgment call (typos, vague references, compound requests, requests that push against the product's own rules, and naturally-phrased ambiguous designs), so each carries a note explaining the call and a needs-clarification flag marking whether the correct system behaviour is to ask rather than proceed.

**No live API call has been made anywhere in building or rendering these.**

---

## Intent parsing: realistic items (80)

### #203

**Input:** `waht is a p-vlaue`
**Label:** `{'type': 'explain', 'topic': 'a p-value'}`
**Needs clarification:** no
**Note:** typo; topic recoverable

### #204

**Input:** `shwo me the ledgr`
**Label:** `{'type': 'settings'}`
**Needs clarification:** no
**Note:** typo of 'show me the ledger'

### #205

**Input:** `acept`
**Label:** `{'type': 'accept', 'persona': None}`
**Needs clarification:** no
**Note:** typo of 'accept'

### #206

**Input:** `underdo that`
**Label:** `{'type': 'undo'}`
**Needs clarification:** no
**Note:** typo blend of 'undo'; clear from context

### #207

**Input:** `explian levenes test`
**Label:** `{'type': 'explain', 'topic': "Levene's test"}`
**Needs clarification:** no
**Note:** typo; topic recoverable

### #208

**Input:** `corelate income and age`
**Label:** `{'type': 'ask_analysis'}`
**Needs clarification:** no
**Note:** typo of 'correlate'; still a clear association question

### #209

**Input:** `brnach it as alt-plan`
**Label:** `{'type': 'branch', 'name': 'alt-plan'}`
**Needs clarification:** no
**Note:** typo of 'branch'; name is clear

### #210

**Input:** `swithc to branch mane`
**Label:** `{'type': 'switch_branch', 'name': 'mane'}`
**Needs clarification:** no
**Note:** 'mane' is likely a typo of 'main', but parse_intent extracts the literal name typed; a nonexistent branch is the orchestrator's lookup to report, not an intent-parsing ambiguity

### #211

**Input:** `exprot this`
**Label:** `{'type': 'export'}`
**Needs clarification:** no
**Note:** typo of 'export'

### #212

**Input:** `skpi this`
**Label:** `{'type': 'skip'}`
**Needs clarification:** no
**Note:** typo of 'skip'

### #213

**Input:** `yo let's just go with the prof's pick`
**Label:** `{'type': 'accept', 'persona': 'professor'}`
**Needs clarification:** no
**Note:** 'prof' is an informal alias for professor

### #214

**Input:** `nah not now`
**Label:** `{'type': 'skip'}`
**Needs clarification:** no
**Note:** colloquial decline reads as skip, not a negative answer (nothing is pending)

### #215

**Input:** `can u undo that`
**Label:** `{'type': 'undo'}`
**Needs clarification:** no
**Note:** informal phrasing of undo

### #216

**Input:** `lemme see the ledger`
**Label:** `{'type': 'settings'}`
**Needs clarification:** no
**Note:** informal phrasing, clear intent

### #217

**Input:** `k go for it`
**Label:** `{'type': 'accept', 'persona': None}`
**Needs clarification:** no
**Note:** colloquial go-ahead reads as accept

### #218

**Input:** `sure why not`
**Label:** `{'type': 'accept', 'persona': None}`
**Needs clarification:** no
**Note:** colloquial agreement reads as acceptance

### #219

**Input:** `meh idk`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** genuinely noncommittal; below the confidence floor

### #220

**Input:** `hold up go back a step`
**Label:** `{'type': 'undo'}`
**Needs clarification:** no
**Note:** informal phrasing of undo

### #221

**Input:** `nvm skip it`
**Label:** `{'type': 'skip'}`
**Needs clarification:** no
**Note:** informal phrasing of skip

### #222

**Input:** `yep that's the one`
**Label:** `{'type': 'accept', 'persona': None}`
**Needs clarification:** no
**Note:** colloquial confirmation reads as accept

### #223

**Input:** `undo that last thing`
**Label:** `{'type': 'undo'}`
**Needs clarification:** no
**Note:** vague referent doesn't block the UNDO type itself

### #224

**Input:** `go back to before`
**Label:** `{'type': 'undo'}`
**Needs clarification:** no
**Note:** vague but clearly UNDO

### #225

**Input:** `use that other one instead`
**Label:** `{'type': 'override', 'function': None}`
**Needs clarification:** yes
**Note:** type is clearly override; which function 'that other one' names needs on-screen context, not resolvable from text alone

### #226

**Input:** `explain that`
**Label:** `{'type': 'explain', 'topic': None}`
**Needs clarification:** yes
**Note:** topic pronoun refers to on-screen context

### #227

**Input:** `why not the other test`
**Label:** `{'type': 'explain', 'topic': None}`
**Needs clarification:** yes
**Note:** 'the other test' needs the current card's candidates to resolve

### #228

**Input:** `skip that step`
**Label:** `{'type': 'skip'}`
**Needs clarification:** no
**Note:** vague but clearly SKIP

### #229

**Input:** `show me that plot again`
**Label:** `{'type': 'show'}`
**Needs clarification:** no
**Note:** type is clear; which plot is a rendering detail, not an intent-parsing ambiguity

### #230

**Input:** `go with the other one`
**Label:** `{'type': 'accept', 'persona': None}`
**Needs clarification:** yes
**Note:** which proposal 'the other one' means needs on-screen context

### #231

**Input:** `override to the other method`
**Label:** `{'type': 'override', 'function': None}`
**Needs clarification:** yes
**Note:** method unresolvable from text alone

### #232

**Input:** `switch back to the first branch`
**Label:** `{'type': 'switch_branch', 'name': None}`
**Needs clarification:** yes
**Note:** 'the first branch' needs a branch-list to resolve to a name

### #233

**Input:** `is income different by gender and also does age relate to income?`
**Label:** `{'type': 'ask_analysis'}`
**Needs clarification:** yes
**Note:** compound request; the system should address one question at a time and ask which first

### #234

**Input:** `what's a p-value and why does it matter here`
**Label:** `{'type': 'explain', 'topic': 'a p-value'}`
**Needs clarification:** no
**Note:** both halves are explanatory; one explain response can reasonably cover both

### #235

**Input:** `compare income across regions, also check for outliers`
**Label:** `{'type': 'ask_analysis'}`
**Needs clarification:** yes
**Note:** two different goals (compare_groups and outliers) bundled together

### #236

**Input:** `undo that and then show me the ledger`
**Label:** `{'type': 'undo'}`
**Needs clarification:** yes
**Note:** two actions requested; only the first executes this turn (rule 3)

### #237

**Input:** `accept the professor's pick and export the notebook`
**Label:** `{'type': 'accept', 'persona': 'professor'}`
**Needs clarification:** yes
**Note:** export should wait for its own turn

### #238

**Input:** `is satisfaction related to training and does it differ by department`
**Label:** `{'type': 'ask_analysis'}`
**Needs clarification:** yes
**Note:** association vs. compare_groups: two different questions

### #239

**Input:** `what is Cohen's d and how is it different from Hedges' g`
**Label:** `{'type': 'explain', 'topic': "Cohen's d"}`
**Needs clarification:** no
**Note:** one coherent explain request even though it names two terms

### #240

**Input:** `go to missingness then outliers`
**Label:** `{'type': 'goto_stage', 'target_stage': 'missingness'}`
**Needs clarification:** yes
**Note:** two stage requests; only the first should be acted on this turn

### #241

**Input:** `override to welch_t and set alpha to 0.01`
**Label:** `{'type': 'override', 'function': 'welch_t'}`
**Needs clarification:** yes
**Note:** two distinct requests bundled together

### #242

**Input:** `branch this as alt-plan and switch to it`
**Label:** `{'type': 'branch', 'name': 'alt-plan'}`
**Needs clarification:** yes
**Note:** branch and switch are two actions; the switch needs its own confirmation

### #243

**Input:** `just run a t-test, skip the assumption checks`
**Label:** `{'type': 'ask_analysis'}`
**Needs clarification:** no
**Note:** reads as a normal analysis request with a method preference; 'skip the checks' cannot be honored (rule 2) regardless of intent type -- eligibility is always deterministic downstream

### #244

**Input:** `I don't care about assumptions just give me a p-value`
**Label:** `{'type': 'ask_analysis'}`
**Needs clarification:** no
**Note:** same as above: the request is heard as a normal analysis question; assumptions still run

### #245

**Input:** `don't bother with corrections, just show me the raw p-value`
**Label:** `{'type': 'ask_analysis'}`
**Needs clarification:** no
**Note:** multiple-testing correction is not controlled by intent parsing; applies regardless

### #246

**Input:** `skip validity, run it anyway`
**Label:** `{'type': 'override', 'function': None}`
**Needs clarification:** yes
**Note:** 'run it anyway' suggests an existing proposal (OVERRIDE), but no function is named

### #247

**Input:** `ignore normality, use the parametric test`
**Label:** `{'type': 'override', 'function': None}`
**Needs clarification:** yes
**Note:** 'the parametric test' names no specific function (could be several)

### #248

**Input:** `just tell me if it's significant`
**Label:** `{'type': 'ask_analysis'}`
**Needs clarification:** no
**Note:** impatience, not a different request; full diagnostics still show regardless

### #249

**Input:** `don't ask me about design, just pick one`
**Label:** `{'type': 'ask_analysis'}`
**Needs clarification:** yes
**Note:** explicitly declines the design question; Section 7.1's rule is that the system never guesses even when asked to -- this still must surface as a clarifying question

### #250

**Input:** `turn off the friction, I know what I'm doing`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** not a recognizable action; likely refers to Section 7.5's override friction, which is not a togglable setting

### #251

**Input:** `force it through`
**Label:** `{'type': 'override', 'function': None}`
**Needs clarification:** yes
**Note:** clearly an override intent but no function named

### #252

**Input:** `just accept whatever, I trust the model`
**Label:** `{'type': 'accept', 'persona': None}`
**Needs clarification:** no
**Note:** reads as an unconditional accept of whatever is currently on screen

### #253

**Input:** `yes I think income matters`
**Context:** `{'awaiting_answer': False}`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** 'yes' would misparse as answer_clarification if a question were pending; none is here, and the rest of the sentence is too vague to be an analysis request

### #254

**Input:** `paired programming session today`
**Context:** `{'awaiting_answer': False}`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** 'paired' coincidentally matches a design keyword; sentence is unrelated small talk

### #255

**Input:** `independent contractor data is in this column`
**Context:** `{'awaiting_answer': False}`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** 'independent' coincidentally matches a design keyword; this is a description, not an answer

### #256

**Input:** `repeated exposure to the treatment`
**Context:** `{'awaiting_answer': False}`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** 'repeated' coincidentally matches a design keyword; unrelated phrase

### #257

**Input:** `correct me if I'm wrong but income seems skewed`
**Context:** `{'awaiting_answer': False}`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** 'correct' coincidentally matches a confirmation keyword; the sentence hedges a different claim

### #258

**Input:** `no idea what this column means`
**Context:** `{'awaiting_answer': False}`
**Label:** `{'type': 'explain', 'topic': None}`
**Needs clarification:** yes
**Note:** 'no' would misparse as a negative answer if a question were pending; here it's part of 'no idea', an implicit request to explain something

### #259

**Input:** `not quite sure what to ask`
**Context:** `{'awaiting_answer': False}`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** 'not quite' coincidentally matches a negative-confirmation phrase; genuinely just unsure

### #260

**Input:** `matched pairs sounds like a good idea for this study design`
**Context:** `{'awaiting_answer': False}`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** 'matched' coincidentally matches a design keyword describing a hypothetical, not an answer

### #261

**Input:** `same subjects appear in both files, is that a problem?`
**Context:** `{'awaiting_answer': False}`
**Label:** `{'type': 'explain', 'topic': None}`
**Needs clarification:** yes
**Note:** a genuine, on-topic question about data structure, but too general to be an analysis request; 'same subjects' coincidentally matches a design keyword by accident

### #262

**Input:** `different people have different opinions on this`
**Context:** `{'awaiting_answer': False}`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** 'different people' coincidentally matches a design keyword; unrelated statement

### #263

**Input:** `asdkfj`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** noise

### #264

**Input:** `🤷`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** an emoji, no text content

### #265

**Input:** `...`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** no content

### #266

**Input:** `so`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** filler word, no request

### #267

**Input:** `anyway`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** filler word, no request

### #268

**Input:** `lol ok`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** acknowledgement with no request

### #269

**Input:** `did that work?`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** asks about system state, not a new request; 'that' is unresolvable

### #270

**Input:** `is this thing on`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** no request content

### #271

**Input:** `test test`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** noise

### #272

**Input:** `hello`
**Label:** `{'type': 'other'}`
**Needs clarification:** yes
**Note:** greeting, no request

### #273

**Input:** `plot icnome agsinst age`
**Label:** `{'type': 'show'}`
**Needs clarification:** no
**Note:** typos in column names don't change the SHOW type

### #274

**Input:** `waht is icnome distribtuion`
**Label:** `{'type': 'explain', 'topic': 'income distribution'}`
**Needs clarification:** no
**Note:** phrased as 'what is X' despite typos; reads as wanting a description, not an explicit plot

### #275

**Input:** `shw me a chart of waeght vs hieght`
**Label:** `{'type': 'show'}`
**Needs clarification:** no
**Note:** typos; SHOW keyword ('chart') still present

### #276

**Input:** `grpah income by regoin`
**Label:** `{'type': 'show'}`
**Needs clarification:** no
**Note:** typo of 'graph'

### #277

**Input:** `vizualize the sasticfaction scores`
**Label:** `{'type': 'show'}`
**Needs clarification:** no
**Note:** typo of 'visualize'

### #278

**Input:** `hist of test_score`
**Label:** `{'type': 'show'}`
**Needs clarification:** no
**Note:** 'hist' shorthand for histogram, a SHOW keyword in spirit

### #279

**Input:** `scaterplot age vs incone`
**Label:** `{'type': 'show'}`
**Needs clarification:** no
**Note:** typo of 'scatterplot'

### #280

**Input:** `waht does cohens d actualy mean`
**Label:** `{'type': 'explain', 'topic': "Cohen's d"}`
**Needs clarification:** no
**Note:** typos, clear intent

### #281

**Input:** `y is the shapiro test failing`
**Label:** `{'type': 'explain', 'topic': 'the Shapiro-Wilk test'}`
**Needs clarification:** no
**Note:** 'y' texting shorthand for 'why'

### #282

**Input:** `explain waht a type 1 error is`
**Label:** `{'type': 'explain', 'topic': 'a type I error'}`
**Needs clarification:** no
**Note:** typo, clear intent

---

## QuestionSpec building: realistic items (80)

### #157

**Input:** `is icnome different by gender`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'income', 'group': 'gender'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Expected correction note:** Using `income` (you wrote 'icnome')
**Note:** 'icnome' resolves unambiguously to 'income' (match_column, default thresholds); needs_clarification stays True for a different reason now -- design is still unstated

### #158

**Input:** `compare revenue across regions`
**Label:** `{'goal': 'compare_groups', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** 'revenue' matches nothing in the column universe above the floor threshold; validate_spec raises InvalidSpecError here, not an ambiguity -- nothing plausible to offer

### #159

**Input:** `does age relate to icome`
**Label:** `{'goal': 'association', 'variables': {'x': 'age', 'y': 'income'}, 'design': 'unknown'}`
**Needs clarification:** no
**Expected correction note:** Using `income` (you wrote 'icome')
**Note:** 'icome' resolves unambiguously to 'income'; association has no design concept, so nothing else is left to ask once the column is corrected

### #160

**Input:** `is there a difference in slaary by department`
**Label:** `{'goal': 'compare_groups', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** 'slaary' matches nothing in the column universe above the floor threshold; InvalidSpecError, same reasoning as 'revenue' above

### #161

**Input:** `check the weit column for outliers`
**Label:** `{'goal': 'outliers', 'variables': {'outcome': 'weight'}, 'design': 'unknown'}`
**Needs clarification:** no
**Expected correction note:** Using `weight` (you wrote 'weit')
**Note:** 'weit' resolves unambiguously to 'weight'; outliers has no design concept

### #162

**Input:** `summarize the icnome column`
**Label:** `{'goal': 'describe', 'variables': {'outcome': 'income'}, 'design': 'unknown'}`
**Needs clarification:** no
**Expected correction note:** Using `income` (you wrote 'icnome')
**Note:** 'icnome' resolves unambiguously to 'income'; describe has no design concept

### #163

**Input:** `is bonus related to performance`
**Label:** `{'goal': 'association', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** neither 'bonus' nor 'performance' matches anything in the column universe above the floor threshold; InvalidSpecError on whichever role is checked first

### #164

**Input:** `why is the sallary column missing so much data`
**Label:** `{'goal': 'missingness', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** 'sallary' matches nothing above the floor threshold; InvalidSpecError

### #165

**Input:** `should we transform the icome variable`
**Label:** `{'goal': 'transform', 'variables': {'outcome': 'income'}, 'design': 'unknown'}`
**Needs clarification:** no
**Expected correction note:** Using `income` (you wrote 'icome')
**Note:** 'icome' resolves unambiguously to 'income'; transform has no design concept

### #166

**Input:** `compare test scores by depatment`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'test_score', 'group': 'department'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Expected correction note:** Using `test_score` (you wrote 'test scores'); using `department` (you wrote 'depatment')
**Note:** both near-misses resolve unambiguously (match_column tolerates the space/typo); needs_clarification stays True because design is still unstated, not because of the columns

### #167

**Input:** `is it different between the groups?`
**Label:** `{'goal': 'compare_groups', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** no outcome or group column named; build_question_spec refuses to guess (Section 10.5)

### #168

**Input:** `does that column relate to this one?`
**Label:** `{'goal': 'association', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** no columns named

### #169

**Input:** `compare the two groups`
**Label:** `{'goal': 'compare_groups', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** no columns named

### #170

**Input:** `is there a trend in it over time?`
**Label:** `{'goal': 'trend', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** no column named

### #171

**Input:** `why is that missing so much?`
**Label:** `{'goal': 'missingness', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** no column named

### #172

**Input:** `are there outliers in this one?`
**Label:** `{'goal': 'outliers', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** no column named

### #173

**Input:** `should this be transformed?`
**Label:** `{'goal': 'transform', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** no column named

### #174

**Input:** `summarize it`
**Label:** `{'goal': 'describe', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** no column named

### #175

**Input:** `is the average the same as before?`
**Label:** `{'goal': 'distribution_fit', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** no column named, and 'before' is not a numeric reference_value

### #176

**Input:** `is it practically equivalent between them?`
**Label:** `{'goal': 'equivalence', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** no columns named

### #177

**Input:** `is income different by gender, and also is age different by region?`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'income', 'group': 'gender'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** compound; the second clause is an entirely separate spec that must wait its turn

### #178

**Input:** `compare income by gender and also check if it's related to age`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'income', 'group': 'gender'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** second clause is a different goal (association)

### #179

**Input:** `is satisfaction different by department, same employees before and after the change, and also does it correlate with tenure`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'satisfaction', 'group': 'period'}, 'design': 'paired'}`
**Needs clarification:** yes
**Note:** design IS stated for the first clause, but a second, unrelated question is bundled in

### #180

**Input:** `summarize income and also tell me about missingness in age`
**Label:** `{'goal': 'describe', 'variables': {'outcome': 'income'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** second clause is a different goal entirely

### #181

**Input:** `does wait_minutes trend over month, and is it different by clinic`
**Label:** `{'goal': 'trend', 'variables': {'outcome': 'wait_minutes', 'time': 'month'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** second clause is a different goal (compare_groups)

### #182

**Input:** `are there outliers in weight and also should it be transformed`
**Label:** `{'goal': 'outliers', 'variables': {'outcome': 'weight'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** second clause (transform) is a distinct goal

### #183

**Input:** `is income equal to 50000 on average, or does it differ by region`
**Label:** `{'goal': 'distribution_fit', 'variables': {'outcome': 'income'}, 'design': 'unknown', 'reference_value': 50000}`
**Needs clarification:** yes
**Note:** 'or' presents two different goals as alternatives; can't resolve which to run without asking

### #184

**Input:** `compare test_score by cohort and also by department`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'test_score', 'group': 'cohort'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** a second group column suggests two separate questions

### #185

**Input:** `is reaction_time related to age, and also is it practically the same across gender`
**Label:** `{'goal': 'association', 'variables': {'x': 'reaction_time', 'y': 'age'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** second clause is a different goal (equivalence)

### #186

**Input:** `why is satisfaction missing so often, and is it different by region too`
**Label:** `{'goal': 'missingness', 'variables': {'outcome': 'satisfaction'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** second clause is a different goal (compare_groups)

### #187

**Input:** `just compare income by gender, independent groups, don't ask me anything else`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'income', 'group': 'gender'}, 'design': 'independent'}`
**Needs clarification:** no
**Note:** despite the pushback framing, design IS explicitly stated; nothing further needs asking

### #188

**Input:** `skip the design question, just assume independent`
**Label:** `{'goal': 'compare_groups', 'variables': {}, 'design': 'independent'}`
**Needs clarification:** yes
**Note:** design is stated, but no outcome/group column is ever named

### #189

**Input:** `I don't want to specify design, just run something`
**Label:** `{'goal': 'compare_groups', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** explicitly declines to answer; the system cannot silently default (Section 7.1) even when asked to skip the question

### #190

**Input:** `don't make me pick a design, just use whichever the data supports`
**Label:** `{'goal': 'compare_groups', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** design is a data-collection fact the data itself cannot reveal; still needs asking

### #191

**Input:** `just tell me if income differs by gender, I don't care about the details`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'income', 'group': 'gender'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** 'I don't care about the details' does not supply the missing design information

### #192

**Input:** `run the simplest test on income by region, whatever that is`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'income', 'group': 'region'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** defers method choice to the eligibility engine (fine) but design is still unstated

### #193

**Input:** `compare wait_minutes across clinics, I already know they're independent so don't ask`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'wait_minutes', 'group': 'clinic'}, 'design': 'independent'}`
**Needs clarification:** no
**Note:** design is explicitly asserted by the user; nothing is left to ask

### #194

**Input:** `just give me a number for how different income is by gender`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'income', 'group': 'gender'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** 'a number' (an effect size) doesn't resolve the design ambiguity

### #195

**Input:** `forget the assumptions, is age different by department`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'age', 'group': 'department'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** 'forget the assumptions' cannot be honored (rule 2); design is still unstated regardless

### #196

**Input:** `I already told you it's independent, is test_score different by cohort`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'test_score', 'group': 'cohort'}, 'design': 'independent'}`
**Needs clarification:** no
**Note:** taken at face value within this single utterance; a stateful system would also check its own history, which one isolated eval item cannot represent

### #197

**Input:** `did income change by region`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'income', 'group': 'region'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** no cue distinguishing independent regions from the same regions tracked over time

### #198

**Input:** `is satisfaction linked to the training`
**Label:** `{'goal': 'compare_groups', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** ambiguous whether 'linked to' means trained-vs-untrained (compare_groups) or a continuous association -- the goal itself, not just the design, needs clarifying

### #199

**Input:** `how does performance compare before and after`
**Label:** `{'goal': 'compare_groups', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** 'before and after' suggests paired, but could be two independent snapshots; 'performance' is also not a column name available here

### #200

**Input:** `is wait time different depending on which clinic you go to`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'wait_minutes', 'group': 'clinic'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** strongly implies independent groups but never says patients aren't shared across clinics; Section 7.1's strict rule still applies

### #201

**Input:** `does the treatment change satisfaction`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'satisfaction', 'group': 'treatment_group'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** 'the treatment' suggests before/after (paired), but 'treatment_group' as a noun suggests independent groups -- reads both ways

### #202

**Input:** `is age related to how long someone waits`
**Label:** `{'goal': 'association', 'variables': {'x': 'age', 'y': 'wait_minutes'}, 'design': 'unknown'}`
**Needs clarification:** no
**Note:** association has no design ambiguity; clean natural phrasing

### #203

**Input:** `do the three clinics differ in wait times`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'wait_minutes', 'group': 'clinic'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** 'the three clinics' strongly implies independent groups, but the wording doesn't say so explicitly enough to skip asking, per Section 7.1

### #204

**Input:** `has morale gone up since the reorg`
**Label:** `{'goal': 'trend', 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** could be a trend over time or a before/after paired comparison; 'morale' is also not a column name available here

### #205

**Input:** `is there a difference by region, same customers surveyed each quarter`
**Label:** `{'goal': 'compare_groups', 'variables': {'group': 'region'}, 'design': 'repeated'}`
**Needs clarification:** yes
**Note:** design IS stated (repeated), but the outcome column is never named

### #206

**Input:** `does income vary across departments, some people moved between departments during the year`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'income', 'group': 'department'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** flags unstable group membership, a genuine complication beyond simple independent/paired -- worth surfacing as its own design conversation, not defaulting to independent

### #207

**Input:** `is icnome diferent between gendre groups, independent smaples`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'income', 'group': 'gender'}, 'design': 'independent'}`
**Needs clarification:** no
**Note:** typos throughout, but the independent-design cue survives

### #208

**Input:** `compaer wait_minutes acros clinics, unrelated patients`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'wait_minutes', 'group': 'clinic'}, 'design': 'independent'}`
**Needs clarification:** no
**Note:** typos; structure clear

### #209

**Input:** `sam subjects mesured twice for staisfaction`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'satisfaction', 'group': 'period'}, 'design': 'paired'}`
**Needs clarification:** no
**Note:** 'sam subjects' typo of 'same subjects' still signals paired

### #210

**Input:** `is their a realationship between age nad income`
**Label:** `{'goal': 'association', 'variables': {'x': 'age', 'y': 'income'}, 'design': 'unknown'}`
**Needs clarification:** no
**Note:** typos; structure clear

### #211

**Input:** `waht is the trned in wait_minutes over the monht`
**Label:** `{'goal': 'trend', 'variables': {'outcome': 'wait_minutes', 'time': 'month'}, 'design': 'unknown'}`
**Needs clarification:** no
**Note:** typos; structure clear

### #212

**Input:** `shoud test_score be log transfromed`
**Label:** `{'goal': 'transform', 'variables': {'outcome': 'test_score'}, 'design': 'unknown'}`
**Needs clarification:** no
**Note:** typos; structure clear

### #213

**Input:** `r there outlier in reaction_time`
**Label:** `{'goal': 'outliers', 'variables': {'outcome': 'reaction_time'}, 'design': 'unknown'}`
**Needs clarification:** no
**Note:** shorthand/typos; structure clear

### #214

**Input:** `y is weigth missing so ofen`
**Label:** `{'goal': 'missingness', 'variables': {'outcome': 'weight'}, 'design': 'unknown'}`
**Needs clarification:** no
**Note:** 'y' shorthand for 'why'; typos elsewhere; structure clear

### #215

**Input:** `is the avg incom eqaul to 50000`
**Label:** `{'goal': 'distribution_fit', 'variables': {'outcome': 'income'}, 'design': 'unknown', 'reference_value': 50000}`
**Needs clarification:** no
**Note:** typos; structure clear

### #216

**Input:** `smae subjects, repeated mesures of satisfaction across departmnet`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'satisfaction', 'group': 'department'}, 'design': 'repeated'}`
**Needs clarification:** no
**Note:** typos; structure clear

### #217

**Input:** `build me a regression model predicting income from age and region`
**Label:** `{'goal': None, 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** regression modelling is out of scope (Section 1.2, 17.1); correct behavior is to say so, not to force it into one of the nine goals

### #218

**Input:** `forecast next quarter's income`
**Label:** `{'goal': None, 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** forecasting is beyond EDA diagnostics; TREND describes past change, it does not forecast

### #219

**Input:** `cluster the customers into segments`
**Label:** `{'goal': None, 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** clustering/segmentation is a modelling task, out of scope

### #220

**Input:** `build a dashboard of all the key metrics`
**Label:** `{'goal': None, 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** not an EDA question at all

### #221

**Input:** `impute the missing income values with the mean`
**Label:** `{'goal': 'missingness', 'variables': {'outcome': 'income'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** asks for an action (impute) rather than a diagnostic question; missingness as a goal covers investigating missingness, the imputation method choice happens later in that stage's flow

### #222

**Input:** `remove the outliers from weight`
**Label:** `{'goal': 'outliers', 'variables': {'outcome': 'weight'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** asks for an action (remove), not a diagnostic question -- same reasoning as imputation above

### #223

**Input:** `normalize the income column`
**Label:** `{'goal': 'transform', 'variables': {'outcome': 'income'}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** commands a specific transform rather than asking whether one is needed; close enough to the transform goal to route there, but still needs a turn to confirm which transform

### #224

**Input:** `tell me which columns are most important for predicting churn`
**Label:** `{'goal': None, 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** feature importance / predictive modelling is out of scope; also no 'churn' column exists

### #225

**Input:** `what's the best machine learning model for this data`
**Label:** `{'goal': None, 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** modelling recommendation is entirely out of scope (Section 1.2)

### #226

**Input:** `automatically clean this dataset for me`
**Label:** `{'goal': None, 'variables': {}, 'design': 'unknown'}`
**Needs clarification:** yes
**Note:** 'automatically ... for me' conflicts with rule 3; also too vague to map to one goal

### #227

**Input:** `I want to know if wait times differ between clinic A, B, and C -- these are three different clinics, not the same patients`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'wait_minutes', 'group': 'clinic'}, 'design': 'independent'}`
**Needs clarification:** no
**Note:** design explicitly and naturally stated

### #228

**Input:** `we surveyed the same 50 employees before and after the training on satisfaction, did it change`
**Label:** `{'goal': 'compare_groups', 'variables': {'outcome': 'satisfaction', 'group': 'period'}, 'design': 'paired'}`
**Needs clarification:** no
**Note:** paired design explicitly and naturally stated

### #229

**Input:** `each department reported its average test score for three consecutive quarters, is there a trend`
**Label:** `{'goal': 'trend', 'variables': {'outcome': 'test_score', 'time': 'quarter'}, 'design': 'unknown'}`
**Needs clarification:** no
**Note:** clean trend question

### #230

**Input:** `is there a correlation between how long someone waits and how satisfied they are afterward`
**Label:** `{'goal': 'association', 'variables': {'x': 'wait_minutes', 'y': 'satisfaction'}, 'design': 'unknown'}`
**Needs clarification:** no
**Note:** clean association question, paraphrased naturally

### #231

**Input:** `our target average income for this cohort was 45000, are we hitting that`
**Label:** `{'goal': 'distribution_fit', 'variables': {'outcome': 'income'}, 'design': 'unknown', 'reference_value': 45000}`
**Needs clarification:** no
**Note:** clean distribution_fit question

### #232

**Input:** `we want to confirm reaction_time is basically unchanged between the treatment and control groups, independent participants in each`
**Label:** `{'goal': 'equivalence', 'variables': {'outcome': 'reaction_time', 'group': 'treatment_group'}, 'design': 'independent'}`
**Needs clarification:** no
**Note:** clean equivalence question, design stated

### #233

**Input:** `can you give me a quick summary of the weight column before we go further`
**Label:** `{'goal': 'describe', 'variables': {'outcome': 'weight'}, 'design': 'unknown'}`
**Needs clarification:** no
**Note:** clean describe question

### #234

**Input:** `age has a lot of blanks, want to understand why before we proceed`
**Label:** `{'goal': 'missingness', 'variables': {'outcome': 'age'}, 'design': 'unknown'}`
**Needs clarification:** no
**Note:** clean missingness question, paraphrased naturally ('blanks' for missing values)

### #235

**Input:** `test_score looks like it might have a few extreme values skewing things, can you check`
**Label:** `{'goal': 'outliers', 'variables': {'outcome': 'test_score'}, 'design': 'unknown'}`
**Needs clarification:** no
**Note:** clean outliers question, paraphrased naturally

### #236

**Input:** `income is heavily skewed, would a transformation help before we model it`
**Label:** `{'goal': 'transform', 'variables': {'outcome': 'income'}, 'design': 'unknown'}`
**Needs clarification:** no
**Note:** clean transform question
