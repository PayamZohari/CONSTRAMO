import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

from collections import Counter

from synthcity.plugins import Plugins


def balance_with_ctgan_multiomics(df, labels):

    y = labels.copy()

    combined = df.copy()
    combined["target"] = y.values

    counts = Counter(y)

    max_class = max(counts.values())

    synthetic_rows = []

    for cls in counts:

        current = counts[cls]

        if current == max_class:
            continue

        needed = max_class - current

        class_df = combined[combined["target"] == cls]

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(class_df)

        model = CTGANSynthesizer(metadata)

        model.fit(class_df)

        samples = model.sample(num_rows=needed)

        synthetic_rows.append(samples)

    if synthetic_rows:

        synth_df = pd.concat(synthetic_rows)

        combined = pd.concat(
            [combined, synth_df],
            ignore_index=True
        )

    y_new = combined["target"]

    X_new = combined.drop(
        columns=["target"]
    )

    return X_new, y_new


def balance_with_diffusion_multiomics(df, labels):

    combined = df.copy()

    combined["target"] = labels.values

    plugin = Plugins().get(
        "ddpm"
    )

    plugin.fit(combined)

    counts = Counter(labels)

    max_count = max(counts.values())

    synthetic_parts = []

    for cls in counts:

        current = counts[cls]

        if current == max_count:
            continue

        needed = max_count - current

        samples = plugin.generate(
            count=needed
        ).dataframe()

        samples["target"] = cls

        synthetic_parts.append(samples)

    if synthetic_parts:

        synth_df = pd.concat(
            synthetic_parts
        )

        combined = pd.concat(
            [combined, synth_df],
            ignore_index=True
        )

    y_new = combined["target"]

    X_new = combined.drop(
        columns=["target"]
    )

    return X_new, y_new