## Example 1

**DB:** `california_schools`

**Question:** What is the highest eligible free rate for K-12 students in the schools in Alameda County?

**Evidence:** Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `frpm.Free Meal Count (K-12)` (column), score=3.0416
- `frpm.Enrollment (K-12)` (column), score=2.6651
- `schools.County` (column), score=2.3357
- `schools` (table), score=1.1271
- `schools.CDSCode` (column), score=0.2486
- `satscores` (table), score=-0.4100
- `frpm` (table), score=-0.6339
- `frpm.Percent (%) Eligible Free (K-12)` (column), score=-0.9255
- `schools.School` (column), score=-1.0925
- `frpm.FRPM Count (K-12)` (column), score=-1.1080
- `schools.District` (column), score=-1.2896
- `schools.AdmEmail2` (column), score=-1.3757
- `frpm.Percent (%) Eligible FRPM (K-12)` (column), score=-1.5357
- `schools.State` (column), score=-1.6139
- `schools.AdmEmail1` (column), score=-1.6392

## Example 2

**DB:** `california_schools`

**Question:** Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.

**Evidence:** Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `frpm.Free Meal Count (Ages 5-17)` (column), score=3.2305
- `frpm.Enrollment (Ages 5-17)` (column), score=2.9535
- `schools` (table), score=1.3066
- `schools.CDSCode` (column), score=0.3666
- `satscores` (table), score=-0.2663
- `frpm` (table), score=-0.4193
- `frpm.Percent (%) Eligible Free (Ages 5-17)` (column), score=-0.7920
- `frpm.FRPM Count (Ages 5-17)` (column), score=-0.8136
- `schools.School` (column), score=-1.1756
- `schools.District` (column), score=-1.2731
- `frpm.Percent (%) Eligible FRPM (Ages 5-17)` (column), score=-1.3179
- `schools.AdmEmail2` (column), score=-1.3572
- `schools.AdmEmail1` (column), score=-1.5386
- `schools.State` (column), score=-1.7503
- `schools.DOC` (column), score=-1.8239

## Example 3

**DB:** `california_schools`

**Question:** Please list the zip code of all the charter schools in Fresno County Office of Education.

**Evidence:** Charter schools refers to `Charter School (Y/N)` = 1 in the table fprm

**Gold labels:**

- `frpm`
- `schools`
- `frpm.CDSCode`
- `schools.CDSCode`
- `schools.District`
- `schools.School`
- `schools.Zip`
- `schools.Charter`

**Top predictions:**

- `schools.School` (column), score=3.3323
- `schools.Charter` (column), score=3.1694
- `frpm.Charter School (Y/N)` (column), score=3.0099
- `schools` (table), score=2.9100
- `schools.County` (column), score=2.7349
- `schools.Zip` (column), score=2.3880
- `satscores` (table), score=0.9541
- `frpm` (table), score=0.6602
- `schools.CDSCode` (column), score=0.5000
- `frpm.School Code` (column), score=0.1190
- `schools.CharterNum` (column), score=-0.4205
- `schools.District` (column), score=-0.8706
- `schools.AdmEmail2` (column), score=-1.0528
- `frpm.Charter School Number` (column), score=-1.0886
- `frpm.County Code` (column), score=-1.1597

## Example 4

**DB:** `california_schools`

**Question:** What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?

**Gold labels:**

- `frpm`
- `schools`
- `frpm.CDSCode`
- `schools.CDSCode`
- `schools.MailStreet`

**Top predictions:**

- `schools.School` (column), score=1.6928
- `frpm` (table), score=1.2513
- `frpm.FRPM Count (K-12)` (column), score=0.9890
- `schools.Street` (column), score=0.9069
- `frpm.School Code` (column), score=0.0619
- `frpm.School Type` (column), score=-0.1284
- `satscores` (table), score=-0.2213
- `schools` (table), score=-0.2843
- `frpm.Charter School Number` (column), score=-0.3620
- `frpm.School Name` (column), score=-0.3652
- `frpm.Enrollment (K-12)` (column), score=-0.4544
- `schools.CDSCode` (column), score=-0.5718
- `frpm.Free Meal Count (K-12)` (column), score=-0.8267
- `frpm.Charter School (Y/N)` (column), score=-0.9112
- `frpm.District Code` (column), score=-1.0937

## Example 5

**DB:** `california_schools`

**Question:** Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.

**Evidence:** Charter schools refers to `Charter School (Y/N)` = 1 in the frpm

**Gold labels:**

- `frpm`
- `schools`
- `frpm.CDSCode`
- `schools.CDSCode`
- `schools.School`
- `schools.Phone`
- `schools.OpenDate`
- `schools.Charter`

**Top predictions:**

- `frpm.Charter School (Y/N)` (column), score=4.0332
- `schools.School` (column), score=3.6220
- `schools.Charter` (column), score=3.5158
- `schools.Phone` (column), score=2.8083
- `schools` (table), score=2.1465
- `frpm` (table), score=1.7017
- `frpm.Charter School Number` (column), score=1.2476
- `schools.CDSCode` (column), score=0.9399
- `frpm.School Type` (column), score=0.5212
- `frpm.School Code` (column), score=0.5056
- `frpm.School Name` (column), score=0.3140
- `schools.CharterNum` (column), score=-0.0450
- `satscores` (table), score=-0.0558
- `schools.District` (column), score=-0.2695
- `schools.AdmEmail2` (column), score=-0.4061

