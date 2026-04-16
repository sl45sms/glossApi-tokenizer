To train with full ~80GB el_Grek corpus from fineweb-2-hq can takes around 70Hours on 4 nodes with 16 A100 80GB GPUs, that ofcource will produse the best results.

Estimate on four Clariden Nodes (16x GH200)
|--|80GB Full CPT|100MB Targeted CPT|1GB Curated CPT|
|---|---|---|---|
|Time (4x GH200)|~2-4 Days|~30 Minutes|~3-5 Hours|
|Embedding Quality|Excellent|Good|Very Good|
|Language Flow|Native|Slightly "broken"|Natural|
|Overfitting Risk|None|High|Low|

However, for quick iteration and benchmarking, we can use the 100MB targeted CPT dataset, which can produce good results in around 30 minutes on 4 nodes with 16 A100 80GB GPUs.

To filter 80GB of text searching for 20,000 different tokens, the classical approach with `if token in text:` would take weeks. 

We will use a much more efficient approach based on the Aho-Corasick algorithm (via the `pyahocorasick` library) or a set-based approach after a quick split. Since you are on Clariden, your processor (Grace CPU) is very powerful, so we will leverage multiprocessing.

**Filtering Strategy**

**Aho-Corasick Automaton:** We construct a "tree" with all 20,000 words. This enables searching for all words simultaneously with a single pass through each line of text (O(n) time complexity).

**Global Counter:** We maintain an array that counts how many times we have found each word.

**Early Exit:** When a line contains words we still "need" (i.e., they have < 50 occurrences), we save it.

This approach is very efficient and can filter the 80GB dataset in a few hours on a single node, which is a huge improvement over the naive method.

