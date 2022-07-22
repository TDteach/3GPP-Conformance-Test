from paddlenlp.datasets import load_dataset

train_ds, dev_ds, public_test_ds = load_dataset("fewclue", name="tnews", splits=("train_0", "dev_0", "test_public"))