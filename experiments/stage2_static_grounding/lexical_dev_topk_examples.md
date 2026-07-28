## Example 1

**DB:** `california_schools`

**Question:** What is the highest eligible free rate for K-12 students in the schools in Alameda County?

**Evidence:** Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `schools.County` (column), score=7.8436
- `frpm.Free Meal Count (K-12)` (column), score=7.2443
- `frpm.Enrollment (K-12)` (column), score=6.7112
- `schools` (table), score=5.0952
- `frpm.Percent (%) Eligible Free (K-12)` (column), score=3.5777
- `frpm.FRPM Count (K-12)` (column), score=3.4193
- `schools.District` (column), score=3.0880
- `schools.Zip` (column), score=3.0404
- `frpm.Percent (%) Eligible FRPM (K-12)` (column), score=3.0364
- `schools.School` (column), score=2.9707
- `schools.State` (column), score=2.9519
- `schools.City` (column), score=2.9202
- `schools.Phone` (column), score=2.9202
- `schools.CDSCode` (column), score=2.9031
- `schools.CharterNum` (column), score=2.8642

## Example 2

**DB:** `california_schools`

**Question:** Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.

**Evidence:** Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`

**Gold labels:**

- `frpm`

**Top predictions:**

- `frpm.Free Meal Count (Ages 5-17)` (column), score=7.5213
- `frpm.Enrollment (Ages 5-17)` (column), score=6.8896
- `schools` (table), score=5.1061
- `frpm.Percent (%) Eligible Free (Ages 5-17)` (column), score=3.9338
- `frpm.FRPM Count (Ages 5-17)` (column), score=3.7784
- `frpm.Percent (%) Eligible FRPM (Ages 5-17)` (column), score=3.4493
- `schools.District` (column), score=3.0982
- `schools.Zip` (column), score=3.0506
- `schools.School` (column), score=2.9808
- `schools.County` (column), score=2.9653
- `schools.State` (column), score=2.9620
- `schools.City` (column), score=2.9304
- `schools.Phone` (column), score=2.9304
- `schools.CDSCode` (column), score=2.9133
- `schools.CharterNum` (column), score=2.8744

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

- `schools.School` (column), score=8.6398
- `schools.Charter` (column), score=8.5366
- `schools.County` (column), score=8.3168
- `schools.Zip` (column), score=8.2909
- `frpm.Charter School (Y/N)` (column), score=6.9909
- `schools` (table), score=6.3536
- `schools.District` (column), score=3.6104
- `frpm.School Code` (column), score=3.4644
- `schools.State` (column), score=3.4114
- `schools.City` (column), score=3.3651
- `schools.Phone` (column), score=3.3651
- `schools.CDSCode` (column), score=3.3402
- `schools.CharterNum` (column), score=3.2833
- `schools.Latitude` (column), score=3.2833
- `schools.AdmEmail1` (column), score=3.2798

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

- `frpm.FRPM Count (K-12)` (column), score=6.2912
- `frpm` (table), score=5.1460
- `schools.Street` (column), score=5.1244
- `schools.School` (column), score=4.9137
- `frpm.Enrollment (K-12)` (column), score=4.4614
- `frpm.Free Meal Count (K-12)` (column), score=4.4407
- `frpm.School Name` (column), score=4.2042
- `frpm.School Code` (column), score=4.1328
- `frpm.School Type` (column), score=4.1272
- `frpm.Percent (%) Eligible FRPM (K-12)` (column), score=4.0751
- `frpm.Percent (%) Eligible Free (K-12)` (column), score=3.7783
- `frpm.Charter School Number` (column), score=3.5806
- `frpm.FRPM Count (Ages 5-17)` (column), score=3.3416
- `frpm.Charter School (Y/N)` (column), score=3.2115
- `frpm.CDSCode` (column), score=3.0070

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

- `frpm.Charter School (Y/N)` (column), score=9.5403
- `schools.School` (column), score=8.6411
- `schools.Charter` (column), score=8.5385
- `schools.Phone` (column), score=8.3353
- `schools` (table), score=6.3545
- `frpm` (table), score=5.7337
- `frpm.Charter School Number` (column), score=5.2026
- `frpm.School Name` (column), score=4.5623
- `frpm.School Code` (column), score=4.4776
- `frpm.School Type` (column), score=4.4707
- `frpm.Charter Funding Type` (column), score=4.0299
- `schools.District` (column), score=3.6108
- `schools.Zip` (column), score=3.5412
- `schools.County` (column), score=3.4165
- `schools.State` (column), score=3.4118

## Example 6

**DB:** `california_schools`

**Question:** How many schools with an average score in Math greater than 400 in the SAT test are exclusively virtual?

**Evidence:** Exclusively virtual refers to Virtual = 'F'

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.AvgScrMath`
- `schools.CDSCode`
- `schools.School`
- `schools.Virtual`

**Top predictions:**

- `schools.Virtual` (column), score=9.0799
- `schools` (table), score=5.1622
- `schools.District` (column), score=3.1461
- `schools.Zip` (column), score=3.0984
- `schools.School` (column), score=3.0287
- `schools.County` (column), score=3.0131
- `schools.State` (column), score=3.0099
- `schools.City` (column), score=2.9782
- `schools.Phone` (column), score=2.9782
- `schools.CDSCode` (column), score=2.9611
- `schools.CharterNum` (column), score=2.9222
- `schools.Latitude` (column), score=2.9222
- `schools.AdmEmail1` (column), score=2.9198
- `schools.AdmEmail2` (column), score=2.9198
- `schools.AdmEmail3` (column), score=2.9198

## Example 7

**DB:** `california_schools`

**Question:** Among the schools with the SAT test takers of over 500, please list the schools that are magnet schools or offer a magnet program.

**Evidence:** Magnet schools or offer a magnet program means that Magnet = 1

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.NumTstTakr`
- `schools.CDSCode`
- `schools.School`
- `schools.Magnet`

**Top predictions:**

- `schools.Magnet` (column), score=9.4999
- `schools` (table), score=6.4856
- `schools.District` (column), score=3.7307
- `schools.Zip` (column), score=3.6610
- `schools.School` (column), score=3.5591
- `schools.County` (column), score=3.5364
- `schools.State` (column), score=3.5317
- `schools.City` (column), score=3.4854
- `schools.Phone` (column), score=3.4854
- `schools.CDSCode` (column), score=3.4604
- `schools.CharterNum` (column), score=3.4035
- `schools.Latitude` (column), score=3.4035
- `schools.AdmEmail1` (column), score=3.4000
- `schools.AdmEmail2` (column), score=3.4000
- `schools.AdmEmail3` (column), score=3.4000

## Example 8

**DB:** `california_schools`

**Question:** What is the phone number of the school that has the highest number of test takers with an SAT score of over 1500?

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.NumGE1500`
- `schools.CDSCode`
- `schools.Phone`

**Top predictions:**

- `schools.Phone` (column), score=5.0597
- `schools.School` (column), score=4.9584
- `frpm.Charter School Number` (column), score=1.9673
- `frpm.School Name` (column), score=1.4147
- `frpm.School Code` (column), score=1.3680
- `frpm.School Type` (column), score=1.3647
- `frpm.Charter School (Y/N)` (column), score=0.7848
- `frpm` (table), score=0.0000
- `satscores` (table), score=0.0000
- `schools` (table), score=0.0000
- `frpm.2013-14 CALPADS Fall 1 Certification Status` (column), score=0.0000
- `frpm.Academic Year` (column), score=0.0000
- `frpm.CDSCode` (column), score=0.0000
- `frpm.Charter Funding Type` (column), score=0.0000
- `frpm.County Code` (column), score=0.0000

## Example 9

**DB:** `california_schools`

**Question:** What is the number of SAT test takers of the schools with the highest FRPM count for K-12 students?

**Gold labels:**

- `frpm`
- `satscores`
- `frpm.CDSCode`
- `satscores.cds`
- `satscores.NumTstTakr`

**Top predictions:**

- `frpm.FRPM Count (K-12)` (column), score=6.3599
- `frpm` (table), score=5.1597
- `schools` (table), score=5.1430
- `frpm.Enrollment (K-12)` (column), score=4.5099
- `frpm.Free Meal Count (K-12)` (column), score=4.5076
- `frpm.Percent (%) Eligible FRPM (K-12)` (column), score=4.1255
- `frpm.Percent (%) Eligible Free (K-12)` (column), score=3.8268
- `frpm.Charter School Number` (column), score=3.5487
- `frpm.FRPM Count (Ages 5-17)` (column), score=3.3737
- `schools.District` (column), score=3.1285
- `schools.Zip` (column), score=3.0809
- `frpm.Free Meal Count (Ages 5-17)` (column), score=3.0210
- `frpm.CDSCode` (column), score=3.0189
- `schools.School` (column), score=3.0111
- `schools.County` (column), score=2.9956

## Example 10

**DB:** `california_schools`

**Question:** Among the schools with the average score in Math over 560 in the SAT test, how many schools are directly charter-funded?

**Gold labels:**

- `frpm`
- `satscores`
- `frpm.CDSCode`
- `satscores.cds`
- `satscores.AvgScrMath`

**Top predictions:**

- `schools.Charter` (column), score=7.7657
- `schools` (table), score=5.1974
- `schools.District` (column), score=3.1757
- `schools.Zip` (column), score=3.1280
- `schools.School` (column), score=3.0583
- `schools.County` (column), score=3.0427
- `schools.State` (column), score=3.0395
- `schools.City` (column), score=3.0078
- `schools.Phone` (column), score=3.0078
- `schools.CDSCode` (column), score=2.9907
- `schools.CharterNum` (column), score=2.9518
- `schools.Latitude` (column), score=2.9518
- `schools.AdmEmail1` (column), score=2.9494
- `schools.AdmEmail2` (column), score=2.9494
- `schools.AdmEmail3` (column), score=2.9494

## Example 11

**DB:** `california_schools`

**Question:** For the school with the highest average score in Reading in the SAT test, what is its FRPM count for students aged 5-17?

**Gold labels:**

- `frpm`
- `satscores`
- `frpm.CDSCode`
- `satscores.cds`
- `satscores.AvgScrRead`

**Top predictions:**

- `frpm` (table), score=5.1417
- `schools.School` (column), score=4.9080
- `frpm.FRPM Count (Ages 5-17)` (column), score=4.8434
- `frpm.School Name` (column), score=4.1948
- `frpm.Free Meal Count (Ages 5-17)` (column), score=4.1721
- `frpm.School Code` (column), score=4.1234
- `frpm.School Type` (column), score=4.1177
- `frpm.Enrollment (Ages 5-17)` (column), score=3.9831
- `frpm.Percent (%) Eligible FRPM (Ages 5-17)` (column), score=3.8058
- `frpm.Percent (%) Eligible Free (Ages 5-17)` (column), score=3.5800
- `frpm.FRPM Count (K-12)` (column), score=3.5738
- `frpm.Charter School Number` (column), score=3.5712
- `frpm.Charter School (Y/N)` (column), score=3.2021
- `frpm.Free Meal Count (K-12)` (column), score=3.0892
- `frpm.CDSCode` (column), score=3.0033

## Example 12

**DB:** `california_schools`

**Question:** Please list the codes of the schools with a total enrollment of over 500.

**Evidence:** Total enrollment can be represented by `Enrollment (K-12)` + `Enrollment (Ages 5-17)`

**Gold labels:**

- `frpm`
- `schools`
- `frpm.CDSCode`
- `schools.CDSCode`

**Top predictions:**

- `frpm.Enrollment (Ages 5-17)` (column), score=6.9869
- `frpm.Enrollment (K-12)` (column), score=6.7677
- `schools` (table), score=5.1490
- `schools.District` (column), score=3.1387
- `schools.Zip` (column), score=3.0911
- `schools.School` (column), score=3.0214
- `schools.County` (column), score=3.0058
- `schools.State` (column), score=3.0026
- `schools.City` (column), score=2.9709
- `schools.Phone` (column), score=2.9709
- `schools.CDSCode` (column), score=2.9538
- `schools.CharterNum` (column), score=2.9149
- `schools.Latitude` (column), score=2.9149
- `schools.AdmEmail1` (column), score=2.9125
- `schools.AdmEmail2` (column), score=2.9125

## Example 13

**DB:** `california_schools`

**Question:** Among the schools with an SAT excellence rate of over 0.3, what is the highest eligible free rate for students aged 5-17?

**Evidence:** Excellence rate = NumGE1500 / NumTstTakr; Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`

**Gold labels:**

- `frpm`
- `satscores`
- `frpm.CDSCode`
- `satscores.cds`
- `satscores.NumTstTakr`
- `satscores.NumGE1500`

**Top predictions:**

- `frpm.Free Meal Count (Ages 5-17)` (column), score=7.1229
- `frpm.Enrollment (Ages 5-17)` (column), score=6.6211
- `satscores.NumGE1500` (column), score=5.2126
- `satscores.NumTstTakr` (column), score=5.2126
- `schools` (table), score=5.0711
- `frpm.Percent (%) Eligible Free (Ages 5-17)` (column), score=3.5836
- `frpm.FRPM Count (Ages 5-17)` (column), score=3.5198
- `frpm.Percent (%) Eligible FRPM (Ages 5-17)` (column), score=3.1718
- `schools.District` (column), score=3.0675
- `schools.Zip` (column), score=3.0198
- `schools.School` (column), score=2.9501
- `schools.County` (column), score=2.9345
- `schools.State` (column), score=2.9313
- `schools.City` (column), score=2.8996
- `schools.Phone` (column), score=2.8996

## Example 14

**DB:** `california_schools`

**Question:** Please list the phone numbers of the schools with the top 3 SAT excellence rate.

**Evidence:** Excellence rate = NumGE1500 / NumTstTakr

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.NumTstTakr`
- `satscores.NumGE1500`
- `schools.CDSCode`
- `schools.Phone`

**Top predictions:**

- `schools.Phone` (column), score=7.9816
- `satscores.NumGE1500` (column), score=5.4522
- `satscores.NumTstTakr` (column), score=5.4522
- `schools` (table), score=5.1387
- `schools.District` (column), score=3.1264
- `schools.Zip` (column), score=3.0788
- `schools.School` (column), score=3.0090
- `schools.County` (column), score=2.9935
- `schools.State` (column), score=2.9902
- `schools.City` (column), score=2.9586
- `schools.CDSCode` (column), score=2.9415
- `schools.CharterNum` (column), score=2.9026
- `schools.Latitude` (column), score=2.9026
- `schools.AdmEmail1` (column), score=2.9002
- `schools.AdmEmail2` (column), score=2.9002

## Example 15

**DB:** `california_schools`

**Question:** List the top five schools, by descending order, from the highest to the lowest, the most number of Enrollment (Ages 5-17). Please give their NCES school identification number.

**Gold labels:**

- `frpm`
- `schools`
- `frpm.CDSCode`
- `schools.CDSCode`
- `schools.NCESSchool`

**Top predictions:**

- `schools.School` (column), score=7.8210
- `frpm.Enrollment (Ages 5-17)` (column), score=5.9472
- `schools` (table), score=5.0911
- `schools.District` (column), score=3.0835
- `schools.Zip` (column), score=3.0359
- `schools.County` (column), score=2.9506
- `schools.State` (column), score=2.9474
- `schools.City` (column), score=2.9157
- `schools.Phone` (column), score=2.9157
- `schools.CDSCode` (column), score=2.8986
- `schools.CharterNum` (column), score=2.8597
- `schools.Latitude` (column), score=2.8597
- `schools.AdmEmail1` (column), score=2.8573
- `schools.AdmEmail2` (column), score=2.8573
- `schools.AdmEmail3` (column), score=2.8573

## Example 16

**DB:** `california_schools`

**Question:** Which active district has the highest average score in Reading?

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.AvgScrRead`
- `schools.CDSCode`
- `schools.StatusType`
- `schools.District`

**Top predictions:**

- `schools.District` (column), score=4.9271
- `frpm.District Name` (column), score=1.3761
- `frpm.District Code` (column), score=1.3300
- `frpm.District Type` (column), score=1.3289
- `frpm` (table), score=0.0000
- `satscores` (table), score=0.0000
- `schools` (table), score=0.0000
- `frpm.2013-14 CALPADS Fall 1 Certification Status` (column), score=0.0000
- `frpm.Academic Year` (column), score=0.0000
- `frpm.CDSCode` (column), score=0.0000
- `frpm.Charter Funding Type` (column), score=0.0000
- `frpm.Charter School (Y/N)` (column), score=0.0000
- `frpm.Charter School Number` (column), score=0.0000
- `frpm.County Code` (column), score=0.0000
- `frpm.County Name` (column), score=0.0000

## Example 17

**DB:** `california_schools`

**Question:** How many schools in merged Alameda have number of test takers less than 100?

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.NumTstTakr`
- `schools.CDSCode`
- `schools.StatusType`
- `schools.County`

**Top predictions:**

- `schools` (table), score=5.2911
- `schools.District` (column), score=3.2569
- `schools.Zip` (column), score=3.2092
- `schools.School` (column), score=3.1395
- `schools.County` (column), score=3.1239
- `schools.State` (column), score=3.1207
- `schools.City` (column), score=3.0890
- `schools.Phone` (column), score=3.0890
- `schools.CDSCode` (column), score=3.0720
- `schools.CharterNum` (column), score=3.0330
- `schools.Latitude` (column), score=3.0330
- `schools.AdmEmail1` (column), score=3.0307
- `schools.AdmEmail2` (column), score=3.0307
- `schools.AdmEmail3` (column), score=3.0307
- `schools.AdmFName2` (column), score=3.0307

## Example 18

**DB:** `california_schools`

**Question:** Rank schools by their average score in Writing where the score is greater than 499, showing their charter numbers.

**Evidence:** Valid charter number means the number is not null

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.AvgScrWrite`
- `schools.CDSCode`
- `schools.CharterNum`

**Top predictions:**

- `schools.Charter` (column), score=8.3785
- `schools` (table), score=5.1409
- `schools.District` (column), score=3.1277
- `schools.Zip` (column), score=3.0800
- `schools.School` (column), score=3.0103
- `schools.County` (column), score=2.9947
- `schools.State` (column), score=2.9915
- `schools.City` (column), score=2.9598
- `schools.Phone` (column), score=2.9598
- `schools.CDSCode` (column), score=2.9428
- `schools.CharterNum` (column), score=2.9038
- `schools.Latitude` (column), score=2.9038
- `schools.AdmEmail1` (column), score=2.9015
- `schools.AdmEmail2` (column), score=2.9015
- `schools.AdmEmail3` (column), score=2.9015

## Example 19

**DB:** `california_schools`

**Question:** How many schools in Fresno (directly funded) have number of test takers not more than 250?

**Gold labels:**

- `frpm`
- `satscores`
- `frpm.CDSCode`
- `satscores.cds`
- `satscores.NumTstTakr`

**Top predictions:**

- `schools` (table), score=5.2672
- `schools.District` (column), score=3.2362
- `schools.Zip` (column), score=3.1886
- `schools.School` (column), score=3.1189
- `schools.County` (column), score=3.1033
- `schools.State` (column), score=3.1001
- `schools.City` (column), score=3.0684
- `schools.Phone` (column), score=3.0684
- `schools.CDSCode` (column), score=3.0513
- `schools.CharterNum` (column), score=3.0124
- `schools.Latitude` (column), score=3.0124
- `schools.AdmEmail1` (column), score=3.0100
- `schools.AdmEmail2` (column), score=3.0100
- `schools.AdmEmail3` (column), score=3.0100
- `schools.AdmFName2` (column), score=3.0100

## Example 20

**DB:** `california_schools`

**Question:** What is the phone number of the school that has the highest average score in Math?

**Gold labels:**

- `satscores`
- `schools`
- `satscores.cds`
- `satscores.AvgScrMath`
- `schools.CDSCode`
- `schools.Phone`

**Top predictions:**

- `schools.Phone` (column), score=5.0889
- `schools.School` (column), score=4.9845
- `frpm.Charter School Number` (column), score=2.0179
- `frpm.School Name` (column), score=1.4408
- `frpm.School Code` (column), score=1.3941
- `frpm.School Type` (column), score=1.3908
- `frpm.Charter School (Y/N)` (column), score=0.8109
- `frpm` (table), score=0.0000
- `satscores` (table), score=0.0000
- `schools` (table), score=0.0000
- `frpm.2013-14 CALPADS Fall 1 Certification Status` (column), score=0.0000
- `frpm.Academic Year` (column), score=0.0000
- `frpm.CDSCode` (column), score=0.0000
- `frpm.Charter Funding Type` (column), score=0.0000
- `frpm.County Code` (column), score=0.0000

