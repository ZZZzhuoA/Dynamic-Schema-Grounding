## Example 1

**DB:** `california_schools`

**Question:** What is the highest eligible free rate for K-12 students in the schools in Alameda County?

**Evidence:** Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `frpm.Free Meal Count (K-12)` (column), score=3.3120
- `schools.County` (column), score=2.9778
- `frpm.Enrollment (K-12)` (column), score=2.9771
- `schools` (table), score=1.6943
- `schools.CDSCode` (column), score=1.1668
- `schools.District` (column), score=0.1364
- `schools.AdmEmail2` (column), score=-0.0133
- `schools.AdmEmail1` (column), score=-0.0595
- `frpm.FRPM Count (K-12)` (column), score=-0.2659
- `schools.School` (column), score=-0.3267
- `frpm` (table), score=-0.4535
- `schools.EILName` (column), score=-0.4802
- `frpm.Percent (%) Eligible Free (K-12)` (column), score=-0.5886
- `schools.Latitude` (column), score=-0.6807
- `schools.Ext` (column), score=-0.7029

## Example 2

**DB:** `california_schools`

**Question:** Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.

**Evidence:** Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `frpm.Free Meal Count (Ages 5-17)` (column), score=3.2289
- `frpm.Enrollment (Ages 5-17)` (column), score=2.9141
- `schools` (table), score=1.7829
- `schools.CDSCode` (column), score=0.8085
- `frpm` (table), score=-0.2495
- `schools.District` (column), score=-0.2763
- `frpm.FRPM Count (Ages 5-17)` (column), score=-0.3454
- `schools.AdmEmail2` (column), score=-0.4072
- `schools.AdmEmail1` (column), score=-0.4434
- `schools.School` (column), score=-0.7220
- `frpm.Percent (%) Eligible Free (Ages 5-17)` (column), score=-0.8108
- `schools.EILName` (column), score=-0.8170
- `schools.EILCode` (column), score=-1.0387
- `schools.Ext` (column), score=-1.0387
- `schools.Latitude` (column), score=-1.0745

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

- `schools.Charter` (column), score=3.3989
- `schools` (table), score=3.1735
- `schools.School` (column), score=3.1727
- `schools.County` (column), score=3.1329
- `frpm.Charter School (Y/N)` (column), score=2.9751
- `schools.Zip` (column), score=2.3679
- `schools.CDSCode` (column), score=1.3636
- `frpm` (table), score=1.2007
- `frpm.School Code` (column), score=0.7751
- `satscores` (table), score=0.6459
- `schools.District` (column), score=0.4951
- `schools.CharterNum` (column), score=0.4815
- `schools.AdmEmail2` (column), score=0.2670
- `schools.AdmEmail1` (column), score=0.2109
- `schools.EILName` (column), score=-0.0429

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

- `schools.School` (column), score=2.1183
- `frpm` (table), score=1.5344
- `frpm.FRPM Count (K-12)` (column), score=1.4965
- `schools.Street` (column), score=1.2372
- `frpm.School Code` (column), score=0.6215
- `frpm.School Name` (column), score=0.5618
- `frpm.School Type` (column), score=0.4393
- `frpm.Charter School Number` (column), score=0.4351
- `frpm.District Name` (column), score=0.4220
- `frpm.CDSCode` (column), score=0.3087
- `frpm.Enrollment (K-12)` (column), score=-0.2081
- `frpm.District Type` (column), score=-0.2211
- `frpm.District Code` (column), score=-0.3213
- `schools.CDSCode` (column), score=-0.4276
- `frpm.Free Meal Count (K-12)` (column), score=-0.4327

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

- `frpm.Charter School (Y/N)` (column), score=3.6463
- `schools.Charter` (column), score=3.4407
- `schools.School` (column), score=3.2135
- `schools.Phone` (column), score=2.2883
- `schools` (table), score=2.2186
- `frpm` (table), score=1.7885
- `frpm.Charter School Number` (column), score=1.6995
- `schools.CDSCode` (column), score=1.4900
- `frpm.School Code` (column), score=0.9343
- `frpm.School Name` (column), score=0.8772
- `frpm.School Type` (column), score=0.8358
- `schools.District` (column), score=0.6665
- `schools.CharterNum` (column), score=0.6420
- `frpm.District Name` (column), score=0.5618
- `frpm.CDSCode` (column), score=0.4832

