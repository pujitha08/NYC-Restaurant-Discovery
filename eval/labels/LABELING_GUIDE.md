# Labeling guide

You have one file: `labels_<your name>.csv`, with 120 rows. Each row is one restaurant for one search request.
Please return it by **Friday 10 AM**. It takes about **1.5–2 hours**.

## Open and save the file

- **Google Sheets:** File → Import → Upload, then choose "Replace spreadsheet". When you're done:
  File → Download → Comma-separated values (.csv).
- **Excel:** open the file directly. When you're done: File → Save As → **CSV UTF-8 (Comma delimited)**.
- Keep the file name. Only edit the `label` and `note` columns. Don't delete, add or reorder rows, and don't
  change `sample_id`.

## What to do for each row

1. Read the request in `query`, then the reviews in `evidence_1` to `evidence_6`.
2. In `label`, enter **0, 1 or 2**:
   - **2:** the reviews clearly show the restaurant meets the need.
   - **1:** it probably meets the need, or meets only part of it.
   - **0:** no evidence for the need, or evidence against it.
3. If you're unsure, add a short reason in `note`.

## Rules

- Follow [`eval/RUBRIC.md`](../RUBRIC.md); it has the details and examples, especially for allergy questions.
- Use **only the reviews shown in the row**. Don't look the restaurant up on Google Maps or anywhere else.
- **No evidence found = 0.** If no review shown says anything about the need (or there are none), label 0 and
  write "no evidence found" in `note`.
- A review saying a dish *contains* the allergen, or just mentioning the allergy, is not evidence the restaurant
  is safe. Warnings and bad experiences count against it.
- **Label on your own.** Don't discuss rows with the other judges, and don't look at anyone else's file until all
  three are returned. Disagreements are expected; we compare them afterwards.

Send the finished CSV back to Kang.
