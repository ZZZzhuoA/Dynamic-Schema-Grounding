# Manual audit of the 15 unresolved corrections

All 15 SQL strings pass SQLite `EXPLAIN QUERY PLAN`. This only establishes syntactic and schema
validity; it does not establish semantic correctness.

| Correction | Database | Proposed original | Decision | Audit finding |
|---:|---|---:|---|---|
| 162 | car_retails | 1577 | Approve | Strong evidence and SQL agreement; question is a paraphrase. |
| 378 | cookbook | 8884 | Approve | `at the minimum` correctly changes `COUNT(*)` to `min_qty`. |
| 684 | hockey | 7799 | Hold | Alignment is strong, but SQL groups by `shootCatch`, so it finds the hand with the largest aggregate shutouts rather than the individual goalie with the most shutouts. |
| 686 | hockey | 7812 | Hold | Alignment and SQL identity are certain, but joining `Coaches` and `Teams` only by `tmID` multiplies seasons. The join also needs the relevant year/league keys before summing points by coach. |
| 890 | mondial_geo | 8280 | Hold | Occurrence order suggests this mapping, but the corrected question is corrupted (`crcentage of Asia?`) and duplicates correction 891. Repair the question before use. |
| 891 | mondial_geo | 8281 | Approve | Exact question occurrence and valid Egypt/Asia area-ratio correction. |
| 1034 | movie_3 | 9375 | Hold | Question and SQL use `Daisy Menagerie`, while evidence still says `Destiny Saturday`. Repair evidence first. |
| 1143 | music_tracker | none | Discard duplicate | It duplicates correction 1142, which already maps to the only matching original record 2077. |
| 1293 | public_review_platform | 4048 | Hold | Alignment is certain, but the SQL averages all elite-year rows. “Time to upgrade” should normally use each user's first elite year. |
| 1303 | public_review_platform | 4067 | Approve | The corrected AM-to-PM condition represents businesses open across noon and is consistent with the revised evidence. |
| 1311 | public_review_platform | 4043 | Hold | Alignment is clear, but SQL unnecessarily restricts to ten businesses and uses a complex `HAVING` threshold instead of directly ranking categories and limiting to three. |
| 1532 | simpson_episodes | 4186 | Approve | Fully consistent average-height clarification. |
| 1576 | simpson_episodes | 4265 | Hold | The denominator counts only joined award rows, not all episodes as requested; duplicate award rows may also inflate the numerator. |
| 2020 | university | 8023 | Hold | The SQL filters `university_year.year = 2013` but does not join/filter the ranking row's year, allowing a score of 98 from another year. |
| 2299 | world | 7884 | Approve | Strong semantic match and the corrected SQL fixes the original ordering logic. |

## Approved mapping summary

Six corrections are approved for immediate use: 162, 378, 891, 1303, 1532, and 2299.

Eight records have a confident source alignment but require content repair before they should enter
training: 684, 686, 890, 1034, 1293, 1311, 1576, and 2020.

Correction 1143 should not be mapped because it is an internal duplicate of correction 1142.
