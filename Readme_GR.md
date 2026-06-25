# Λογική Υλοποίησης και Επέκτασης του Tokenizer

Στόχος αυτού του σταδίου είναι η προσαρμογή του υπάρχοντος tokenizer (`swiss-ai/Apertus-8B-Instruct-2509`) ώστε να υποστηρίζει καλύτερα την ελληνική γλώσσα, να μειώσει τον κατακερματισμό (fragmentation) των ελληνικών κειμένων σε πολλά υπο-τμήματα (subtokens) και να βελτιστοποιήσει την αναπαράσταση ειδικής ορολογίας, όπως αυτής του GlossAPI.

Η διαδικασία χωρίζεται στα εξής βασικά βήματα, χωρίς να περιλαμβάνει πρακτικές περαιτέρω εκπαίδευσης (CPT ή SFT):

## 1. Εξαγωγή Στατιστικών Λέξεων από Κείμενα (Word Statistics Extraction)
Αρχικά, αναλύεται ένα μεγάλο ελληνικό corpus (όπως το ελληνικό τμήμα του `FineWeb2-HQ`) για την εύρεση της συχνότητας των λέξεων.  
Αυτό γίνεται μέσω του εργαλείου `vocabularyGen/countWords.py` το οποίο με ένα πέρασμα (streaming):
* Μετράει την εμφάνιση απλών λέξεων (words).
* Εξάγει λέξεις που βρίσκονται μέσα σε χωρία με εισαγωγικά (quoted words).
* Καταγράφει λέξεις που ξεκινούν με κεφαλαίο (capitalized words: ονόματα, χώρες, κλπ.).
Τα δεδομένα καταγράφονται σε βάσεις δεδομένων SQLite, ώστε να αποφευχθεί το υπερβολικό φόρτωμα της μνήμης (RAM) από το τεράστιο μέγεθος των corpus.

## 2. Επιλογή Υποψήφιων Tokens (Candidate Selection)
Η εισαγωγή νέου λεξιλογίου δεν γίνεται αδιάκριτα. Γίνεται χρήση του εργαλείου `vocabularyGen/selectTokenizerCandidates.py` για την ακριβή επιλογή:
* **Ανάλυση Κατακερματισμού:** Ελέγχεται πώς ο αρχικός (base) tokenizer τεμαχίζει την κάθε υποψήφια λέξη σε subtokens.
* **Φιλτράρισμα:** Επιλέγονται λέξεις που εμφανίζονται αρκετά συχνά στα δεδομένα *αλλά* ταυτόχρονα κατακερματίζονται πολύ από τον παλιό tokenizer (π.χ. χρειάζονται 4-5 tokens). Λέξεις με τεράστια συχνότητα περνούν από αυστηρότερα κριτήρια για να μην χαλάσει η ισορροπία του γενικού λεξιλογίου.
* **Ενοποίηση Τύπων:** Γίνεται συγχώνευση (case-folding) κεφαλαίων/πεζών αναλογικά με τις ανάγκες, εκτός αν ρυθμιστεί διαφορετικά.
* **Curated Στατικά Tokens:** Πέρα από τα tokens του corpus, προστίθενται στατικές λίστες από curated tokens σημαντικά για το domain-specific περιβάλλον μας (GlossAPI).
Το τελικό αρχείο επιλεγμένων λέξεων (`selected_tokens_v1.txt`) περιλαμβάνει πλέον τις λέξεις συνήθως με ένα αρχικό κενό διάστημα, ώστε να ταιριάζουν στα όρια (word boundaries) του tokenizer.

## 3. Επέκταση Tokenizer και Έξυπνη Αρχικοποίηση (Alignment & Initialization)
Έχοντας το επιλεγμένο λεξιλόγιο, προχωράμε στην επέκταση:
* Μέσω του `scripts/extend_apertus_tokenizer.py` προσθέτουμε τα νέα tokens στον base tokenizer, δημιουργώντας τη νέα έκδοση (π.χ. `apertus-greek-v1`).
* **Resizing Model Embeddings:** Το ίδιο το μοντέλο φορτώνεται για να αυξηθεί το μέγεθος του πίνακα των embeddings ώστε να περιλάβει το νέο αυξημένο μέγεθος του λεξιλογίου.
* **Mean Initialization:** Τα καινούρια tokens δεν αρχικοποιούνται από το μηδέν ή με τυχαίο θόρυβο. Το script ελέγχει τα subtokens που θα δίνονταν προηγουμένως για τη συγκεκριμένη λέξη, εξάγει τα embeddings τους από το αρχικό μοντέλο, υπολογίζει τον **μέσο όρο** τους, και τα τοποθετεί στο νέο, ενιαίο token. Αυτή η πρακτική μεταφέρει μερική γνώση της έννοιας εξαρχής στο νέο token.

Το αποτέλεσμα είναι η δημιουργία ενός σταθερού (persistent) aligned αρχικού checkpoint (μοντέλο και εκτεταμένος tokenizer), το οποίο θα μπει στην αναμονή, έτοιμο για τα μεταγενέστερα εκπαιδευτικά στάδια.

```bash
./run_uenv.sh python scripts/extend_apertus_tokenizer.py \
  --token-file artifacts/vocab_candidates/selected_tokens_v1.txt \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --model-output-dir "${SCRATCH}/apertus-greek-tokenizer-v1" \
  --torch-dtype bfloat16 \
  --trust-remote-code \
  --untied-output-init-strategy mean \
  --overwrite
```
* προσοχη  πρεπει να μπει το --untied-output-init-strategy mean. Αυτό θα αρχικοποιήσει κάθε νέα γραμμή του lm_head ως τον μέσο όρο των output embeddings των subtokens που αντικαθιστά — ακριβώς όπως γίνεται ήδη by default για τα input embeddings.
Το Tokenizer επιτυνγχανει καλη μείωση του fragmentation, όπως φαίνεται στο παράδειγμα:
```
"Η εκπαίδευση είναι απαραίτητη για την ανάπτυξη."
  Base: 19 tokens → ['Η', ' εκ', 'πα', 'ί', 'δ', 'ευ', 'ση', ...]
  Ext:  10 tokens → ['Η', ' εκπαίδευση', ' είναι', ' απαραίτητη', ...]
  Μείωση: 47.4%
```

έλεγχος
```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --trained-model "${SCRATCH}/apertus-greek-tokenizer-v1" \
  --output-json artifacts/reports/greek_mmlu_init_eval.json
```
```json
{
  "output_json": "artifacts/reports/greek_mmlu_init_eval.json",
  "base_report_cache": "artifacts/reports/greek_mmlu_base_eval.json",
  "base_report_cache_hit": true,
  "krikri_report_cache": "artifacts/reports/greek_mmlu_krikri_eval.json",
  "krikri_report_cache_hit": true,
  "base_accuracy": 0.647005772005772,
  "krikri_accuracy": 0.6746632996632996,
  "trained_accuracy": 0.5928932178932179,
  "accuracy_delta": -0.05411255411255411
}
```
- Εάν η ευθυγραμμισμένη αρχική τιμή είναι ήδη πολύ χειρότερη από τη βάση, σταματήστε εδώ και εντοπίστε σφάλματα στην επέκταση tokenizer ή στην αρχικοποίηση ενσωμάτωσης. Στο παραδειγμα
το accuracy έχει πέσει από 64.7% σε 59.3%, κάτι που είναι σημαντικό, αλλά όχι καταστροφικό (catastrophic). Εάν η πτώση ήταν πολύ μεγαλύτερη, θα ήταν ένδειξη ότι κάτι πήγε στραβά στη διαδικασία επέκτασης ή αρχικοποίησης.
Θα πρεπει να γίνει ομως smoke test.
---



## 4. Συνεχής Προ-εκπαίδευση (Continued Pre-Training - CPT)

Μετά τη δημιουργία του αρχικού aligned checkpoint (με το νέο λεξιλόγιο), το μοντέλο χρειάζεται περαιτέρω εκπαίδευση για να αποδώσει νόημα στα νέα tokens και να καλύψει κενά γνώσης. Η διαδικασία αυτή δεν γίνεται τυφλά σε αχανή δεδομένα, αλλά με στοχευμένη μεθοδολογία.

### 4.1 Μείγμα Εκπαίδευσης και "English Anchor"
Για να αποφύγουμε την απώλεια ικανοτήτων συλλογισμού (catastrophic forgetting), τα μείγματα δεδομένων περιλαμβάνουν μια "άγκυρα" αγγλικών δεδομένων. 

Για καθε dataset υπολογιζεται η αναλογία που θα έχει στο μείγμα, με βάση:
* **Συχνότητα Εμφάνισης:** Πόσο συχνά εμφανίζεται το dataset στο training corpus.
* **Περπλεξία (Perplexity):** Πόσο "ξένο" είναι το dataset για το μοντέλο (υψηλή περπλεξία σημαίνει μεγαλύτερο κενό γνώσης).
* **Θεματική Καινοτομία (Novelty):** Πόσο μοναδικό είναι το περιεχόμενο του dataset σε σχέση με τα ήδη υπάρχοντα δεδομένα.


Μια γενική αναλογία είναι:
* **90% Ελληνικά κείμενα**
* **10% Αγγλικά κείμενα** (από το FineWeb-HQ)
Με αυτόν τον τρόπο το μοντέλο ενσωματώνει την ελληνική γλώσσα χωρίς να αλλοιώνεται η βασική του λογική δομή.

### smoke

```bash
SMOKE_TEST=1 \
SKIP_WARMUP=1 \
ENGLISH_PROBABILITY=0.7 \
GREEK_PROBABILITY=0.3 \
SMOKE_FULL_STEPS=1000 \
SMOKE_FULL_WARMUP_STEPS=50 \
MODEL_PATH="${SCRATCH}/apertus-greek-tokenizer-v1/" \
OUTPUT_DIR="${SCRATCH}/apertus-greek-cpt-smoke-1k" \
sbatch scripts/run_apertus_greek_cpt_clariden.sh --time=02:00:00 --partition=normal --gpus-per-node=4
```
Τι θα κάνει το smoke test:
SKIP_WARMUP=1 — πάμε κατευθείαν σε full training, αποφεύγουμε το embedding-only στάδιο που μπερδεύει
10.000 steps full phase με cosine schedule, 200 warmup steps
~41M tokens συνολικά (10000 × 4 GPU × 1024 seq)
~2 ώρες εκτίμηση χρόνου (1 ώρα στην πράξη)

Αυτό το smoke test είναι σχεδιασμένο για να ελέγξει την ομαλή λειτουργία της διαδικασίας CPT με το νέο tokenizer, χωρίς να στοχεύει σε σημαντική βελτίωση απόδοσης. Αν όλα πάνε καλά, θα προχωρήσουμε στην πλήρη CPT εκπαίδευση με στοχευμένα δεδομένα.



αξιολόγιση
```bash
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --trained-model "${SCRATCH}/apertus-greek-cpt-smoke-1k/final" \
  --output-json artifacts/reports/greek_mmlu_smoke_1k_eval.json
```

./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --trained-model "${SCRATCH}/apertus-greek-cpt-smoke-5k-bal/final" \
  --output-json artifacts/reports/greek_mmlu_smoke_5k_bal_eval.json

Αποτέλεσμα smoke	Ενέργεια
>62-63% accuracy	✅ Προχώρα σε production CPT
60-62%	⚠️ Οριακό — ξανατσέκαρε το CPT recipe (lr, warmup, mixture)
<60%	❌ Σταμάτα — debug το tokenizer extension (λιγότερα tokens; καλύτερο initialization;)

```json

```


### 4.2 Εμπλουτισμός με Στοχευμένα Δεδομένα (GlossAPI)
Ενώ τα γενικά σύνολα (π.χ. `FineWeb2-HQ`) προσφέρουν όγκο, υστερούν σε εξειδικευμένη εντοπιότητα. Για τον αποτελεσματικό εμπλουτισμό χρησιμοποιούμε τα σύνολα δεδομένων του **GlossAPI**, τα οποία επιλέγονται προσδιορίζοντας τον δείκτη **Περπλεξίας (Perplexity - PPL)** του μοντέλου σε αυτά:
* **Identification of Knowledge Gaps:** Αξιολογούμε την "έκπληξη" (αδυναμία πρόβλεψης) του Apertus στα dataset. Σύνολα με υψηλή περπλεξία σε συνδυασμό με θεματική καινοτομία (novelty) έχουν προτεραιότητα.
* **Top Candidates:** Σε αυτά εντάσσονται datasets όπως:
  * `glossAPI/modern-greek-dictionary`: Παρουσιάζει πολύ υψηλό perplexity gap, δείχνοντας ότι το μοντέλο έχει πραγματικό κενό στην αντίστοιχη ορολογία/δομή.
  * `glossAPI/eurlex-greek-legislation` & `glossAPI/Ekklisiastika_Keimena`: Εμφανίζουν εξαιρετική ποιότητα (QP) και υψηλή θεματική καινοτομία, άρα είναι εξαιρετικά για domain-specific γνώση.
  * `glossAPI/openarchives.gr`, `artoszois`, `Ellinika_Keimena_Project_Gutenberg`: Ισορροπημένα σύνολα μεταξύ ποιότητας αρχείου, καινοτομίας, και κενού γνώσης του μοντέλου (gap).

### 4.3 Smoke Tests & Targeted CPT Probe

Η διαδικασία εκπαίδευσης πρέπει να προσπελάσει πρώτα μια "πύλη" αξιολόγησης (Validation Gate):
1. **Δημιουργία Targeted Probe Dataset**: Αρχικά εξάγουμε ένα μικρό dataset (π.χ. 1GB από curated υλικό GlossAPI) και τρέχουμε ένα πολύ σύντομο CPT (έως ~100 steps).
2. **Αξιολόγηση Smoke Test**: Το μοντέλο δοκιμάζεται σε benchmarks (π.χ. GreekMMLU). Αν η απόδοση πέσει αξιοσημείωτα (regression), η προσέγγιση ελέγχεται ξανά (δεν προχωράμε στο Production). Αν η απόδοση μείνει σταθερή ή βελτιωθεί, παίρνουμε το πράσινο φως.



### 4.4 Production CPT και Checkpoint Promotion
Η πραγματική CPT εκπαίδευση φτάνει τα επιδωκόμενα (production) μεγέθη, προσπελαύνοντας το μείγμα English Anchor και Custom Greek Corpuses (κοντά στο 1B tokens):
* Κατά την εκπαίδευση, αποθηκεύονται διαρκώς **ενδιάμεσα checkpoints**.
* Το τελικό (`final`) checkpoint δεν προωθείται τυφλά ως το "καλύτερο". Τρέχουμε το εργαλείο αξιολόγησης (π.χ. `tools/evaluateCptCheckpoints.py`) σε όλη τη σειρά lineage (checkpoint sweep).
* Προωθείται για το επόμενο στάδιο (**SFT**) εκείνο το ενδιάμεσο η τελικό checkpoint που φέρει το υψηλότερο συνολικό σκορ στα benchmarks (MMLU score, Secondary School, κλπ), το οποίο γίνεται ο νέος **Champion**.

**Χρυσός Κανόνας**: Διατηρούμε πάντα ένα καθαρό lineage: *Aligned Init Checkpoint* ➔ *Best CPT Champion Checkpoint* ➔ *SFT*.







ΣΦΤ
SMOKE_TEST=1 \
SMOKE_MAX_STEPS=100 \
SMOKE_TRAIN_SAMPLES=4096 \
MODEL_PATH="${SCRATCH}/apertus-greek-tokenizer-v1/" \
DATASET_NAME="${SCRATCH}/greek-mmlu-sft-parquet/train.parquet" \
OUTPUT_DIR="${SCRATCH}/apertus-greek-sft-greekmmlu" \
MAX_SEQ_LENGTH=1024 \
LEARNING_RATE=2e-5 \
NUM_TRAIN_EPOCHS=1 \
OVERWRITE_OUTPUT_DIR=1 \
sbatch SFT/run_apertus_greek_sft_clariden.sh --time=01:00:00 --partition=dev