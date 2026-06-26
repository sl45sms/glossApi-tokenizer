
Η επέκταση του tokenizer (vocab extension) για μορφολογικά πλούσιες γλώσσες όπως η Ελληνική συνήθως σκοντάφτει είτε στην απλοϊκή μέθοδο του μέσου όρου (που καταστρέφει τη γεωμετρία του latent space) είτε στο distillation (που είναι υπολογιστικά ασύμφορο και απαιτεί τεράστιο overhead).

Για το Apertus, χρειαζόμαστε μια μέθοδο που να εκμεταλλεύεται την υπάρχουσα «νοημοσύνη» του base μοντέλου, αλλά να μεταφέρει τη δομή της νέας γλώσσας με μαθηματική ακρίβεια.

Θα δοκιμάσουμε να δημιουργήσουμε μια νέα αρχιτεκτονική: τη Συστατική Γεωμετρική Ευθυγράμμιση (Compositional Geometric Alignment - CGA). 

Η μέθοδος αυτή χωρίζεται σε δύο φάσεις, τη Μορφολογική Κατάτμηση και τη Γεωμετρική Προβολή Anchor-to-Anchor.

1. Μορφολογικά Καθοδηγούμενο Vocab Extension (The Vocabulary)
Αντί για ένα τυφλό frequency-based BPE (Byte Pair Encoding) που σπάει τις ελληνικές λέξεις σε τυχαία συμπλέγματα χαρακτήρων λόγω πολυτονικού ή καταλήξεων, εισάγουμε το **Morpho-BPE.Rule-based Anchors**. Πριν τρέξει ο αλγόριθμος BPE στο ελληνικό corpus του GlossAPI, «κλειδώνουμε» ως αδιαίρετα tokens τις βασικές παραγωγικές καταλήξεις, θέματα και συνδέσμους της γλώσσας (π.χ. -ουμε, -οντας, -ότητα, προς, κατά).

Syllable Constraints: Επιβάλλουμε περιορισμούς ώστε οι ελάχιστες υπομονάδες (sub-tokens) να συμπίπτουν με συλλαβικές δομές της στοχευμένης γλώσσας.Έτσι, αποφεύγουμε το φαινόμενο μια λέξη να σπάει σε tokens που δεν έχουν καμία σημασιολογική ή συντακτική αυτοτέλεια.

2. Γεωμετρική Ευθυγράμμιση & Προβολή (The Embeddings) Αυτός είναι ο πυρήνας της ιδέας που αντικαθιστά τον μέσο όρο και το distillation. Αντί να προσπαθήσουμε να «μαντέψουμε» τα νέα embeddings, θα ευθυγραμμίσουμε έναν εξωτερικό, στατικό αλλά γεωμετρικά πλούσιο χώρο (π.χ. ένα εξειδικευμένο FastText ή Word2Vec μοντέλο εκπαιδευμένο σε τεράστιο ελληνικό corpus) με τον high-dimensional χώρο του base μοντέλου της Swiss AI.

Το Μαθηματικό Μοντέλο 
Εύρεση Anchor Tokens: Εντοπίζουμε ένα κοινό σύνολο $N$ λέξεων/tokens που υπάρχουν ήδη τόσο στον tokenizer του base μοντέλου (έστω και σπασμένα) όσο και στο ελληνικό στατικό μοντέλο (π.χ. διεθνείς όροι, βασικές έννοιες, κοινά tokens).

Ορθογώνια Προβολή (Orthogonal Procrustes): Έστω $X \in \mathbb{R}^{N \times D_{base}}$ οι φορείς (embeddings) των anchors στο base μοντέλο και $Y \in \mathbb{R}^{N \times D_{target}}$ οι φορείς τους στο ελληνικό στατικό μοντέλο. Αναζητούμε έναν πίνακα μετασχηματισμού $W$ που να ελαχιστοποιεί την απόσταση:$$\min_{W} \| XW - Y \|_F \quad \text{subject to} \quad W^T W = I$$Η λύση προκύπτει άμεσα μέσω Singular Value Decomposition (SVD):$$X^T Y = U \Sigma V^T \implies W = U V^T$$ 

Αρχικοποίηση Νέων Tokens: Για κάθε νέο ελληνικό token που προσθέτουμε, παίρνουμε το embedding του από τον target χώρο ($y_{new}$) και το προβάλλουμε αντίστροφα στον χώρο του LLM:$$x_{new} = y_{new} W^T$$Γιατί είναι πιο αποδοτικό: 
Έναντι του Μέσου Όρου: Ο μέσος όρος υπο-tokens συχνά καταλήγει σε «νεκρές ζώνες» (low-density areas) του embedding space. Η προβολή Procrustes διατηρεί αναλλοίωτη την τοπολογική και σημασιολογική δομή (cosine similarities) των ελληνικών εννοιών.

Έναντι του Distillation: 
Δεν απαιτείται κανένα forward pass από το LLM, ούτε backpropagation, ούτε GPU clusters για την αρχικοποίηση. Η εύρεση του $W$ γίνεται σε δευτερόλεπτα στην CPU με απλή γραμμική άλγεβρα.

3. Μορφολογικός Τανυστής (Compositional Residuals)Για λέξεις που είναι σύνθετες ή παράγωγες και δεν υπάρχουν ούτε στο στατικό μοντέλο, εφαρμόζουμε έναν Συστατικό Τανυστή.

 Αν το νέο token είναι το ανθρωπότητα, και έχουμε ήδη το θέμα άνθρωπ- και την κατάληξη -ότητα, το νέο embedding υπολογίζεται ως:$$E_{\text{new}} = W^T \left( f(E_{\text{root}}) \otimes g(E_{\text{suffix}}) \right)$$ Όπου $f$ και $g$ είναι απλοί, linear προβολείς των δομικών στοιχείων, διασφαλίζοντας ότι η γραμματική δομή της γλώσσας μεταφράζεται σε ακριβή γεωμετρική μετατόπιση μέσα στον vector space. 
 
 Γενίκευση για Κάθε Γλώσσα
 
 Η μέθοδος αυτή είναι πλήρως αγνωστικιστική ως προς τη γλώσσα (language-agnostic). Για να εφαρμοστεί, για παράδειγμα, στα Φινλανδικά ή στα Τουρκικά (που είναι επίσης εξαιρετικά συγκολλητικές γλώσσες), απαιτούνται μόνο δύο πράγματα: 
 Ένα corpus της γλώσσας-στόχου για τη δημιουργία του Morpho-BPE.
 Ένας στατικός zero-shot embedding χώρος της γλώσσας αυτής για να τρέξει ο μετασχηματισμός Procrustes.

---

## Υλοποίηση 

Έγινε σε 5 αρχεία Python, όλα στο `vocab-extension/`:

### Χάρτης modules (Module map)

| Module | Ρόλος |
|---|---|
| `morpho_bpe.py` | Μορφολογικά anchors (προθήματα, επιθήματα, θέματα, forced tokens), έλεγχος συλλαβικής δομής, σκοράρισμα υποψήφιων tokens |
| `fasttext_utils.py` | Λήψη/φόρτωση ελληνικών FastText διανυσμάτων (`cc.el.300`), εξαγωγή anchor tokens, κατασκευή embedding matrices |
| `geometric_alignment.py` | Orthogonal Procrustes μέσω SVD, προβολή base↔target, αρχικοποίηση embeddings μοντέλου |
| `compositional_residuals.py` | Μορφολογική αποσύνθεση (πρόθημα+ρίζα+επίθημα), τανυστική σύνθεση embeddings για σύνθετες λέξεις |
| `cga_pipeline.py` | Πλήρες CLI pipeline: tokenizer extension → FastText → Procrustes → σύνθεση → αποθήκευση μοντέλου |

### Γρήγορη εκκίνηση (Quick start)

```bash
# 1. Μόνο tokenizer (δεν χρειάζεται GPU)
./run_uenv.sh python vocab-extension/cga_pipeline.py \
    --base-tokenizer artifacts/tokenizers/apertus-base \
    --token-file artifacts/vocab_candidates/selected_tokens_v1.txt \
    --output-dir artifacts/tokenizers/apertus-greek-cga-v1 \
    --overwrite

# 2. Πλήρες CGA με αρχικοποίηση μοντέλου (χρειάζεται GPU μνήμη)
#    Το --fasttext-use-subword συνιστάται: δίνει CGA-projected embeddings
#    ακόμα και για σύνθετες λέξεις που δεν υπάρχουν στο απλό .vec.gz.
./run_uenv.sh python vocab-extension/cga_pipeline.py \
    --base-tokenizer artifacts/tokenizers/apertus-base \
    --token-file artifacts/vocab_candidates/selected_tokens_v1.txt \
    --base-model swiss-ai/Apertus-8B-Instruct-2509 \
    --output-dir artifacts/tokenizers/apertus-greek-cga-v1 \
    --model-output-dir "${SCRATCH}/apertus-greek-cga-v1" \
    --trust-remote-code --torch-dtype bfloat16 \
    --fasttext-use-subword \
    --save-alignment artifacts/reports/cga_alignment_W.npz \
    --overwrite

# 3. Με PCA bridge (για πειραματισμό — δες σημείωση παρακάτω)
./run_uenv.sh python vocab-extension/cga_pipeline.py \
    --base-tokenizer artifacts/tokenizers/apertus-base \
    --token-file artifacts/vocab_candidates/selected_tokens_v1.txt \
    --base-model swiss-ai/Apertus-8B-Instruct-2509 \
    --output-dir artifacts/tokenizers/apertus-greek-cga-v1 \
    --model-output-dir "${SCRATCH}/apertus-greek-cga-v1" \
    --trust-remote-code --torch-dtype bfloat16 \
    --pca-bridge \
    --overwrite

# 4. Αξιολόγηση (μετά το pipeline)
./run_uenv.sh python evaluation/evaluate_greek_mmlu.py \
  --base-model swiss-ai/Apertus-8B-Instruct-2509 \
  --trained-model "${SCRATCH}/apertus-greek-cga-v1" \
  --output-json artifacts/reports/greek_mmlu_cga_eval.json
```

### Βασικές παράμετροι (Key options)

| Παράμετρος | Λειτουργία |
|---|---|
| `--fasttext-use-subword` | **Συνιστάται**: Χρήση `.bin` μοντέλου με subword πληροφορία. Επιτρέπει CGA projection ακόμα και για σύνθετες λέξεις εκτός λεξιλογίου. |
| `--pca-bridge` | Μείωση διαστάσεων μέσω PCA πριν το Procrustes (4096d→300d). **Πειραματικό**: ο intrinsic dimensionality των LLM embeddings είναι >>300, το PCA variance retained ήταν μόλις 36%. |
| `--pca-dim 300` | Διάσταση-στόχος για PCA (default: διάσταση FastText) |
| `--min-anchors 200` | Ελάχιστος αριθμός κοινών anchors για Procrustes (λιγότερα → mean fallback) |
| `--morpho-min-score 0.2` | Ελάχιστο σκορ μορφολογικής συνοχής (0–1) για να κρατηθεί ένα υποψήφιο token |
| `--no-morpho-filter` | Παράκαμψη μορφολογικού φιλτραρίσματος |
| `--no-compositional` | Παράκαμψη compositional residual υπολογισμού |
| `--save-alignment path.npz` | Αποθήκευση του Procrustes W, των μέσων, και των anchors για μελλοντική επαναχρησιμοποίηση |
| `--untied-output-init-strategy zero` | Στρατηγική για νέες γραμμές του `lm_head`: `zero` (συντηρητική), `mean`, ή `keep-resized` |

### Συμπεριφορά υποβάθμισης (Fallback behavior)

Το pipeline υποβαθμίζεται ομαλά:

1. **Επαρκή anchors (≥10 κοινά)**: Πλήρες CGA: Procrustes → FastText projection για νέα tokens → compositional residuals για σύνθετες λέξεις.
2. **Λίγα anchors**: Πτώση σε mean-pooling initialization (ίδια συμπεριφορά με το `extend_apertus_tokenizer.py --base-model`).

### Αναμενόμενο pre-CPT regression

Όλες οι μέθοδοι επέκτασης tokenizer (CGA, mean-init, distill) εμφανίζουν πτώση ~5-6% στο GreekMMLU πριν από CPT. Αυτό είναι αναπόφευκτο: ο extended tokenizer παράγει διαφορετικές ακολουθίες tokens από αυτές που είδε το μοντέλο κατά το pretraining. Το πραγματικό τεστ είναι **μετά από CPT** — εκεί φαίνεται ποια μέθοδος δίνει καλύτερο starting point.

### Εξαρτήσεις (Dependencies)

- Βασικές: `torch`, `numpy`, `transformers`, `tokenizers`
- Προαιρετικές: `fasttext` (για `--fasttext-use-subword`· αλλιώς χρησιμοποιεί απλό `.vec.gz`)
- Το ελληνικό μοντέλο FastText κατεβάζεται αυτόματα στην πρώτη χρήση στο `$SCRATCH/fasttext/` ή `~/.cache/fasttext/`
