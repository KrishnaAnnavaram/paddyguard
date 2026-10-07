# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

This section gives the technical names and the technical verbs of paddyguard. The README uses each term with only this meaning.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **image** | One leaf photo, identified by `image_id` | picture, sample, frame |
| **class** | One disease or health label, from the sorted class list | category, disease (for a label) |
| **field** | The group of images from one farm, plot or plant (`field_id`) | site, location (for a group) |
| **near-duplicate** | An image with a thumbnail signature close to another image of the same class | copy, twin |
| **split group** | Images joined by a shared field or a near-duplicate link | cluster, block |
| **split** | `train`, `val` or `test` | fold (for a split), partition |
| **weather window** | The `lag_days` days before the date of an image | lookback, history |
| **weather features** | The 8 values from the weather window (`w_tmean` … `w_wind`) | weather vector, climate data |
| **provider** | The source of daily weather: `offline` or `open-meteo` | API, service |
| **augmentation** | A weather-style change of a training image (fog, haze, rain, brightness, shadow, motion blur) | effect, filter, transform |
| **pixel contract** | float32 RGB values in [0, 1] | scale, range |
| **image features** | Colour histograms, lesion-colour shares and texture statistics of an image | handcrafted features, descriptors |
| **variant** | `image`, `weather` or `image+weather` in the ablation | model type, setting |
| **ablation** | The comparison of the variants on the same splits and seeds | experiment, study |
| **fusion** | The way a torch model uses weather: `none`, `late` or `film` | combination, merge |
| **backbone** | The image network before the head | encoder, base model |
| **robustness test** | The image model on augmented test images | stress test |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **validate** | Check the metadata, the files, the near-duplicates and the weather keys |
| **group** | Join images that share a field or a near-duplicate link |
| **split** | Put each split group in exactly one of `train`, `val`, `test` |
| **augment** | Apply one augmentation to a training image, on the fly |
| **select** | Choose the C value of a variant on `val` |
| **evaluate** | Measure a model once on the `test` images |
| **fine-tune** | Train all layers of a pretrained backbone at a low learning rate |
| **cache** | Keep a weather answer in a CSV file |
