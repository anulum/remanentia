# Recall Gate

Read-side honesty gate for stored findings and synthesized recall answers.

The gate folds finding axes into `validated`, `boundary`, or `refuted`, then
applies the synthesis floor: a synthesized answer never renders above its
weakest input, and any `refuted` input floors the whole synthesis to `refuted`.
Unknown axis values render at `boundary` rather than raising or validating.

::: recall_gate
