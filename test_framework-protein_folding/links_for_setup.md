# Σημειώσεις

2026-01-07
==========

[Κώδικας από τη σελίδα](https://huggingface.co/docs/transformers/main/en/model_doc/esm#transformers.EsmForMaskedLM) στο κάτω μέρος:

```py
from transformers import AutoTokenizer, EsmForProteinFolding
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
inputs = tokenizer(["MLKNVQVQLV"], return_tensors="pt", add_special_tokens=False)  # A tiny random peptide
outputs = model(**inputs)
folded_positions = outputs.positions
```

> το διάνυσμα προκύπτει από τον μέσο όρο (mean pooling) των αναπαραστάσεων του τελευταίου επιπέδου

Έχω και τα δεδομένα εισόδου, περίπου. ~~Πρέπει να κοπούν σε κομμάτια, κάπως έτσι:~~ Τότε τα πήρα από σελίδα που δεν κατέγραψα, τώρα βρήκα και [καλύτερο link](https://www.uniprot.org/help/downloads).

```py
from sklearn.model_selection import train_test_split

train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels, test_size=0.25, shuffle=True)
```

[που κι αυτό το πήρα απ' τη σελίδα.](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_language_modeling-tf.ipynb#scrollTo=366147ad)

Φαίνεται ότι το `facebook/esmfold_v1` είναι φάκελος και όχι αρχείο που πρέπει να κατεβάσω. Μάλλον το αντικαθιστώ με
`facebook/esm2_t6_8M_UR50D` στην περίπτωσή μου.

Δοκιμάζω `git clone`... Δεν πιάνει. Κατεβάζω τα binary αρχεία και τα βάζω σε φάκελο,
περιμένω ότι θα πρέπει να δώσω το όνομα αυτού του φακέλου στα `from_pretrained`. Λάθος.

Τελικά: το κατεβάζει μόνο του. Ο κώδικας απλά αναφέρει `facebook/esm2_t6_8M_UR50D`
και όταν εκτελεστεί, εγκαθίσταται από μόνο του, έτσι απλά.

Απέτυχε το

```py
model = EsmForProteinFolding.from_pretrained("facebook/esm2_t6_8M_UR50D")
```

λέγοντας

```
  File "/home/me/Documents/Code/Since_2025-06/nn_clustering/3_protein_folding/venv/lib/python3.13/site-packages/transformers/models/esm/modeling_esmfold.py", line 2009, in __init__
    if self.config.esmfold_config.fp16_esm:
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'fp16_esm'
```

Δούλεψε, όμως το

```py
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
```

από τη [μικρή βοήθεια στα transformers.](https://huggingface.co/facebook/esm2_t6_8M_UR50D?library=transformers)

Τελικά, μάλλον protein folding θέλω, αλλά το `EsmForProteinFolding` απαιτεί κάποιο
`self.config.esmfold_config` το οποίο υποθέτω ότι το δίνω έμμεσα, ίσως φτιάχνοντας
κάποιο αρχείο. Και πάλι, διαφώνησε το ChatGPT ότι αυτό είναι το σωστό μοντέλο.

Διαβάζοντας τη [σελίδα του transformers](https://pypi.org/project/transformers/) είδα ότι
λέγεται

```sh
pip install "transformers[torch]"
```

Παρατηρώ ότι ίσως πιάνει με `facebook/esmfold_v1` αλλά κατεβαίνει πολύ αργά.

Τελικά, όπως με βοήθησε να καταλάβω και το ChatGPT και δοκιμές στη γραμμή εντολών,
θέλω `EsmModel` αντί για `EsmForProteinFolding`.

Το ChatGPT μού έγραψε και έναν αλγόριθμο για input. Έχω input και output. Μένει:

1. Blast
2. αποθήκευση για input στους αλγορίθμους μας
3. διόρθωση command line χρήσης των προγραμμάτων
4. shell script που τα συνδυάζει

Βρήκα και [καλύτερο link](https://www.uniprot.org/help/downloads) για κατέβασμα πρωτεϊνών.

Για το `requirements.txt` κατάργησα το παλιό και έτρεξα `pip freeze > requirements.txt`
